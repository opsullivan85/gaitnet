from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from typing import Sequence
from rsl_rl.modules import ActorCritic
from gaitnet import get_logger
from torch.distributions import Normal, Categorical
import gaitnet.constants as const

logger = get_logger()


def make_mlp(
    input_size, hidden_sizes, output_size, activation=nn.ReLU, output_activation=None
) -> nn.Sequential:
    """
    Creates an MLP (multi-layer perceptron) in PyTorch.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list[int]): Sizes of hidden layers.
        output_size (int): Number of output features.
        activation (nn.Module): Activation class for hidden layers (default: ReLU).
        output_activation (nn.Module or None): Optional activation for output layer.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    in_size = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(activation())
        in_size = h
    layers.append(nn.Linear(in_size, output_size))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)



def masked_option_logits(
    logits: torch.Tensor, observations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask invalid footstep options and put each leg's candidates on a per-step scale.

    The candidates are a Monte-Carlo sample of each leg's continuous foothold
    surface, so the raw aggregate ``sum_i exp(f_i)`` grows with the number of
    samples drawn, while the no-op is a single atom. Comparing the two directly
    makes the probability of stepping depend on ``num_footstep_options``.
    Subtracting ``log(N_valid)`` per leg turns that sum into a mean, so the
    policy weighs the *average quality* of a leg's available footholds against
    the value of waiting, independent of how many candidates were sampled.

    Args:
        logits: Raw per-option logits (num_envs, num_options), leg-major with the
            no-op option last.
        observations: Policy observations (num_envs, obs_dim), used to tell which
            options are real footsteps.

    Returns:
        masked_logits: (num_envs, num_options), invalid options set to -inf.
        op_mask: (num_envs, num_options), True where the option is a real footstep.
    """
    num_envs, num_options = logits.shape
    candidates = observations[:, const.gait_net.robot_state_dim :].view(
        num_envs, num_options, const.gait_net.footstep_option_dim
    )
    # the one hot encoding is [no_op, leg1, leg2, leg3, leg4]; options filtered out
    # by the sampler carry the no-op encoding, as does the trailing no-op itself
    no_op_mask = candidates[:, :, 0] == 1
    op_mask = ~no_op_mask

    # per-leg normalization over the leg-major candidate block (no-op is last)
    step_logits = logits[:, :-1].reshape(num_envs, const.robot.num_legs, -1)
    step_valid = op_mask[:, :-1].reshape(num_envs, const.robot.num_legs, -1)
    # legs with no valid candidate are fully masked below, so the count is a
    # placeholder there and only needs to keep the log finite
    valid_counts = step_valid.sum(dim=-1, keepdim=True).clamp(min=1)
    step_logits = step_logits - torch.log(valid_counts.float())

    masked_logits = torch.cat(
        [step_logits.reshape(num_envs, -1), logits[:, -1:]], dim=-1
    )
    masked_logits = masked_logits.masked_fill(no_op_mask, float("-inf"))
    # the no-op is an atom, not a sampled option: never normalized, always available
    masked_logits[:, -1] = logits[:, -1]
    return masked_logits, op_mask


class GaitnetActor(nn.Module):
    def __init__(
        self,
        shared_state_dim: int,
        shared_layer_sizes: Sequence[int],
        unique_state_dim: int,
        unique_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
        checkpoint_chunk_size: int | None = 1024,
        use_bf16: bool = True,
    ):
        """
        Args:
            checkpoint_chunk_size: When gradients are enabled, split the batch into chunks
                of this size and checkpoint each, so backward only holds one chunk's
                activations at a time. Memory would otherwise grow with
                batch_size * num_options. None disables checkpointing.
            use_bf16: Run the network under bf16 autocast on CUDA.
        """
        super().__init__()
        logger.info("GaitnetActor initializing")
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.use_bf16 = use_bf16

        self.shared_encoder = make_mlp(
            input_size=shared_state_dim,
            hidden_sizes=shared_layer_sizes[:-1],
            output_size=shared_layer_sizes[-1],
        )
        logger.info(f"shared_encoder: {self.shared_encoder}")

        self.unique_encoder = make_mlp(
            input_size=unique_state_dim,
            hidden_sizes=unique_layer_sizes[:-1],
            output_size=unique_layer_sizes[-1],
        )
        self.unique_embedding_size = unique_layer_sizes[-1]
        # random embedding to represent no-op
        self.no_op_embedding = nn.Parameter(torch.randn(unique_layer_sizes[-1]))
        logger.info(f"unique_encoder: {self.unique_encoder}")

        trunk_input_dim = shared_layer_sizes[-1] + unique_layer_sizes[-1]
        self.trunk = make_mlp(
            input_size=trunk_input_dim,
            hidden_sizes=trunk_layer_sizes[:-1],
            output_size=trunk_layer_sizes[-1],
        )
        logger.info(f"trunk: {self.trunk}")

        self.value_head = make_mlp(
            input_size=trunk_layer_sizes[-1], hidden_sizes=[], output_size=1
        )
        logger.info(f"value_head: {self.value_head}")

        self.duration_head = make_mlp(
            input_size=trunk_layer_sizes[-1],
            hidden_sizes=[],
            output_size=1,
        )
        logger.info(f"duration_head: {self.duration_head}")

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the GaitNet actor.

        Args:
            obs (torch.Tensor): Input observations.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - logits: Action selection logits (num_envs, num_options)
                - durations: Duration predictions for each option (num_envs, num_options)
        """
        num_envs = obs.shape[0]
        shared_state = obs[:, : const.gait_net.robot_state_dim]

        remaining_obs_size = obs.shape[1] - const.gait_net.robot_state_dim
        unique_states_dim = remaining_obs_size / const.gait_net.footstep_option_dim
        assert (
            unique_states_dim.is_integer()
        ), f"Expected unique_state_size ({const.gait_net.footstep_option_dim}) to evenly divide the remaining observation size ({remaining_obs_size}), got {unique_states_dim}"
        unique_states_dim = int(unique_states_dim)
        unique_states = obs[:, const.gait_net.robot_state_dim :].view(
            num_envs, unique_states_dim, const.gait_net.footstep_option_dim
        )

        with torch.autocast(
            device_type=obs.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16 and obs.is_cuda,
        ):
            chunk = self.checkpoint_chunk_size
            if chunk and torch.is_grad_enabled() and num_envs > chunk:
                outputs = [
                    checkpoint(self._forward, shared, unique, use_reentrant=False)
                    for shared, unique in zip(
                        shared_state.split(chunk), unique_states.split(chunk)
                    )
                ]
                logits = torch.cat([logits for logits, _ in outputs])
                duration = torch.cat([duration for _, duration in outputs])
            else:
                logits, duration = self._forward(shared_state, unique_states)

        return logits.float(), duration.float()

    def _forward(
        self, shared_state: torch.Tensor, unique_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on already split observations.

        Args:
            shared_state: (num_envs, robot_state_dim)
            unique_states: (num_envs, num_options, footstep_option_dim)

        Returns:
            logits and durations, both (num_envs, num_options)
        """
        # note that the one hot encoding is [no_op, leg1, leg2, leg3, leg4]
        no_op_mask = unique_states[:, :, 0] == 1  # (num_envs, num_options)

        shared_embedding: torch.Tensor = self.shared_encoder(shared_state)
        unique_embeddings: torch.Tensor = self.unique_encoder(unique_states)
        unique_embeddings = torch.where(
            no_op_mask.unsqueeze(-1),
            self.no_op_embedding.to(unique_embeddings.dtype),
            unique_embeddings,
        )

        trunk_input = torch.cat(
            [
                shared_embedding.unsqueeze(dim=1).expand(-1, unique_states.shape[1], -1),
                unique_embeddings,
            ],
            dim=-1,
        )
        trunk_output = self.trunk(trunk_input)

        logits = self.value_head(trunk_output).squeeze(-1)  # (num_envs, num_options)

        # Scale durations to reasonable range, no-op options have zero duration
        min_dur, max_dur = const.gait_net.valid_swing_duration_range
        scale = max_dur - min_dur
        duration = self.duration_head(trunk_output).squeeze(-1).float()
        duration = torch.sigmoid(duration) * scale + min_dur
        duration = torch.where(no_op_mask, 0.0, duration)  # (num_envs, num_options)

        return logits, duration

    @staticmethod
    def act_inference(
        actor: "GaitnetActor", observations: torch.Tensor
    ) -> torch.Tensor:
        """Deterministic action selection for inference.

        Args:
            observations: Observations (num_envs, obs_dim)

        Returns:
            actions: Deterministic actions (num_envs, 2) where:
                     - Column 0: discrete action index (0-16)
                     - Column 1: mean duration value
        """
        # Get logits and durations from actor
        logits, duration_means = actor(observations)

        # Same masking and per-leg normalization as training, so the deterministic
        # policy can't select a filtered-out option and weighs legs the same way.
        masked_logits, _ = masked_option_logits(logits, observations)

        # Select action with highest logit (deterministic)
        action_index = torch.argmax(masked_logits, dim=-1)  # (num_envs,)

        # Use mean durations for the selected action (deterministic)
        batch_size = action_index.shape[0]
        batch_indices = torch.arange(batch_size, device=action_index.device)
        selected_durations = duration_means[batch_indices, action_index]  # (num_envs,)

        # Combine action index and duration into a single tensor
        actions = torch.stack(
            [action_index.float(), selected_durations], dim=-1
        )  # (num_envs, 2)

        return actions


class GaitnetCritic(nn.Module):
    def __init__(
        self,
        shared_state_dim: int,
        shared_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
    ):
        super().__init__()
        logger.info("GaitnetCritic initializing")

        self.shared_encoder = make_mlp(
            input_size=shared_state_dim,
            hidden_sizes=shared_layer_sizes[:-1],
            output_size=shared_layer_sizes[-1],
        )
        logger.info(f"shared_encoder: {self.shared_encoder}")

        self.trunk = make_mlp(
            input_size=shared_layer_sizes[-1],
            hidden_sizes=trunk_layer_sizes,
            output_size=1,
        )
        logger.info(f"trunk: {self.trunk}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the critic.

        Only the shared robot state is used. The footstep-option candidates
        are i.i.d. noise conditional on the state (their identities and
        ordering come from the sampler's random tie-break), so they carry
        ~no information about V(s) and are dropped rather than fed through
        an order-sensitive combiner.

        Args:
            obs (torch.Tensor): Input observations.

        Returns:
            torch.Tensor: Value predictions (num_envs, 1).
        """
        shared_state = obs[:, : const.gait_net.robot_state_dim]
        shared_embedding = self.shared_encoder(shared_state)
        value = self.trunk(shared_embedding)  # (num_envs, 1)

        return value


class GaitnetActorCritic(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor: GaitnetActor,
        critic: GaitnetCritic,
        episode_info: dict[str, Any] | None = None,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        duration_std: float = 0.05,  # Initial standard deviation for duration noise (learned)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        **kwargs,
    ):
        """

        Args:
            num_actor_obs (_type_): _description_
            num_critic_obs (_type_): _description_
            num_actions (_type_): _description_
            actor (GaitnetActor): _description_
            critic (GaitnetCritic): _description_
            episode_info (dict[str, Any] | None, optional): Shared dictionary to dump episode data into
            init_noise_std (float, optional): _description_. Defaults to 1.0.
            noise_std_type (str, optional): _description_. Defaults to "scalar".
            duration_std (float, optional): Initial duration std, then learned. Defaults to 0.05.
        """
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.episode_info = episode_info
        nn.Module.__init__(self)
        logger.info("GaitnetActorCritic initializing")
        logger.info(
            "Note: num_actor_obs, num_critic_obs, and num_actions are not used in making the actor and critic."
        )
        logger.debug(
            f"num_actor_obs: {num_actor_obs}, num_critic_obs: {num_critic_obs}, num_actions: {num_actions}"
        )
        # warn that kwargs are ignored
        if len(kwargs) > 0:
            logger.warning(
                f"GaitnetActorCritic received unused kwargs: {kwargs}"
            )

        self.actor = actor
        self.critic = critic

        # Store the number of options (should match num_actions from environment)
        self.num_options = (
            const.gait_net.num_footstep_options * const.robot.num_legs + 1
        )  # +1 for no-op

        # Action distribution components (populated in update_distribution)
        self.discrete_distribution: Categorical | None = None  # For action selection
        self.duration_distribution: Normal | None = None  # For duration values

        # Duration standard deviation, learned. Log-parameterized so it can't
        # go negative and collapse the distribution during training.
        self.duration_log_std = nn.Parameter(
            torch.log(torch.tensor(duration_std, dtype=torch.float32))
        )

        # Cache for storing selected action indices and durations
        self._last_action_indices: torch.Tensor | None = None
        self._last_sampled_durations: torch.Tensor | None = None
        self._cached_duration_means: torch.Tensor | None = None
        # (num_envs, num_options) True where the option is a real footstep, not a no-op
        self._op_mask: torch.Tensor | None = None

    def reset(self, dones=None):
        """Reset recurrent states (no-op for non-recurrent policy)."""
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def duration_std(self) -> torch.Tensor:
        """Learned duration standard deviation (0-dim tensor, always positive)."""
        return self.duration_log_std.exp()

    @property
    def action_mean(self):
        """Return the mean of the distribution (for logging).

        Returns shape (num_envs, 2) with:
        - Column 0: discrete action index (mode of categorical)
        - Column 1: duration mean for the most likely action
        """
        if self.discrete_distribution is None or self.duration_distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution first."
            )

        # Get the mode (argmax) of the discrete distribution
        discrete_mode = torch.argmax(
            self.discrete_distribution.logits, dim=-1
        )  # (num_envs,)

        # Get duration mean for the most likely action
        batch_size = discrete_mode.shape[0]
        batch_indices = torch.arange(batch_size, device=discrete_mode.device)
        duration_mean = self.duration_distribution.mean[
            batch_indices, discrete_mode
        ]  # (num_envs,)

        return torch.stack(
            [discrete_mode.float(), duration_mean], dim=-1
        )  # (num_envs, 2)

    @property
    def action_std(self):
        """Return std for logging compatibility with PPO.

        Returns shape (num_envs, 2) with:
        - Column 0: dummy std for discrete action (set to 1.0)
        - Column 1: duration std
        """
        if self.discrete_distribution is None or self.duration_distribution is None:
            return torch.ones(1, 2)

        num_envs = self.discrete_distribution.logits.shape[0]
        device = self.discrete_distribution.logits.device

        # Dummy std for discrete action
        discrete_std = torch.ones(num_envs, 1, device=device)

        # Duration std (learned, shared across all options)
        duration_std = self.duration_std.detach().to(device).expand(num_envs, 1)

        return torch.cat([discrete_std, duration_std], dim=-1)

    @property
    def entropy(self):
        """Return the entropy used for the PPO entropy bonus.

        Only the discrete action selection is included. The duration std is learned,
        and a Normal's entropy grows with log(std) at a constant rate, so including it
        would steadily push the duration noise up regardless of performance.
        """
        if self.discrete_distribution is None or self.duration_distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution first."
            )

        return self.discrete_distribution.entropy()  # (num_envs,)

    def update_distribution(self, observations):
        """Update the action distribution based on observations.

        Creates a joint distribution over:
        1. Discrete action selection (Categorical)
        2. Continuous duration for each action (Normal)

        Args:
            observations: Observations (num_envs, obs_dim)
        """
        observations = observations["policy"]
        # Get logits and duration means from actor
        logits, duration_means = self.actor(observations)  # Access underlying actor
        # logits: (num_envs, num_options)
        # duration_means: (num_envs, num_options)

        if self.episode_info is not None:
            no_op_mask = duration_means == 0
            op_mask = ~no_op_mask
            if torch.any(op_mask):
                leg_logits = logits[:, :-1].view(
                    logits.shape[0], 4, -1
                )  # (num_envs, 4, num_options)
                leg_op_mask = (
                    torch.sum(
                        no_op_mask[:, :-1].view(no_op_mask.shape[0], 4, -1).long(),
                        dim=-1,
                    )
                    == 0
                )
                # Use unbiased=False for faster std computation
                per_leg_std = torch.std(leg_logits, dim=2, unbiased=False)[leg_op_mask]  # (num_ops,)
                option_std = torch.mean(per_leg_std)  # (1,)
                self.episode_info["leg_option_std"] = option_std.item()

                # Faster correlation approximation using dot product
                costs = observations[:, const.gait_net.robot_state_dim :].view(
                    logits.shape[0], -1, const.gait_net.footstep_option_dim
                )[:, :, -1]  # (num_envs, num_options, footstep_option_dim)
                # Use simple normalized dot product instead of full corrcoef
                logits_flat = logits[op_mask].flatten()
                costs_flat = -costs[op_mask].flatten()
                # Normalize
                logits_norm = logits_flat - logits_flat.mean()
                costs_norm = costs_flat - costs_flat.mean()
                logits_norm = logits_norm / (logits_norm.std(unbiased=False) + 1e-8)
                costs_norm = costs_norm / (costs_norm.std(unbiased=False) + 1e-8)
                correlation = (logits_norm * costs_norm).mean()
                self.episode_info["logit_cost_correlation"] = correlation.item()

            else:
                self.episode_info["leg_option_std"] = 0
                self.episode_info["logit_cost_correlation"] = 0

            # Learned duration distribution std (the parameter itself, not the
            # empirical std of sampled durations logged in act()).
            self.episode_info["duration_std_param"] = self.duration_std.item()

        # Cache duration means for later use
        self._cached_duration_means = duration_means

        # mask out options the sampler filtered, and scale each leg's candidates
        # by its valid count so stepping doesn't inflate with the option count
        masked_logits, op_mask = masked_option_logits(logits, observations)
        self._op_mask = op_mask

        self.discrete_distribution = Categorical(logits=masked_logits)

        if self.episode_info is not None:
            # probability of stepping rather than holding. With the per-leg
            # normalization this is comparable across option counts.
            self.episode_info["step_prob"] = (
                (1.0 - self.discrete_distribution.probs[:, -1]).mean().item()
            )
            self.episode_info["valid_options"] = (
                op_mask[:, :-1].sum(dim=-1).float().mean().item()
            )

        # Create continuous distribution for durations. Keep this as the learned
        # parameter (not detached) so gradients from the duration log-prob flow
        # back into duration_log_std, letting the policy widen/narrow it as needed.
        duration_std = self.duration_std.expand_as(duration_means)
        self.duration_distribution = Normal(duration_means, duration_std)

    def act(self, observations, **kwargs):
        """Sample actions from the policy.

        Args:
            observations: Observations (num_envs, obs_dim)

        Returns:
            actions: Sampled actions (num_envs, 2) where:
                     - Column 0: discrete action index (0-16)
                     - Column 1: sampled duration value
        """
        self.update_distribution(observations)
        observations = observations["policy"]

        # Sample discrete action
        action_index = self.discrete_distribution.sample()  # (num_envs,)

        # Sample duration for the selected action
        batch_size = action_index.shape[0]
        batch_indices = torch.arange(batch_size, device=action_index.device)

        # Sample from the duration distribution for the selected action
        sampled_durations = self.duration_distribution.sample()[
            batch_indices, action_index
        ]  # (num_envs,)

        # Combine action index and duration into a single tensor
        actions = torch.stack(
            [action_index.float(), sampled_durations], dim=-1
        )  # (num_envs, 2)

        if self.episode_info is not None:
            no_op_index = const.gait_net.num_footstep_options * const.robot.num_legs
            op_mask = action_index != no_op_index
            if torch.any(op_mask):
                # Use unbiased=False for faster computation
                self.episode_info["duration_mean"] = torch.mean(
                    sampled_durations[op_mask]
                ).item()
                self.episode_info["duration_std"] = torch.std(
                    sampled_durations[op_mask], unbiased=False
                ).item()
                self.episode_info["ops_per_step"] = torch.mean(op_mask.float()).item()
            else:
                self.episode_info["duration_mean"] = 0
                self.episode_info["duration_std"] = 0
                self.episode_info["ops_per_step"] = 0

        # Cache for log probability computation
        self._last_action_indices = action_index
        self._last_sampled_durations = sampled_durations

        return actions

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution.

        This computes the JOINT log probability of:
        1. Selecting the discrete action
        2. Sampling the duration for that action

        Args:
            actions: Actions (num_envs, 2) where:
                     - Column 0: discrete action index (0-16)
                     - Column 1: sampled duration value

        Returns:
            log_probs: Joint log probabilities (num_envs,)
        """
        if self.discrete_distribution is None or self.duration_distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution first."
            )

        # Extract action indices and durations
        action_indices = actions[:, 0].long()  # (num_envs,)
        sampled_durations = actions[:, 1]  # (num_envs,)

        # Get log probability of discrete action selection
        discrete_log_prob = self.discrete_distribution.log_prob(
            action_indices
        )  # (num_envs,)

        # Get log probability of duration for the selected action
        batch_size = action_indices.shape[0]
        batch_indices = torch.arange(batch_size, device=action_indices.device)

        # Evaluate log probability of these durations under the Normal distribution
        # duration_distribution has shape (num_envs, num_options)
        # We need to evaluate sampled_durations under the distribution for the selected action
        duration_log_prob = self.duration_distribution.log_prob(
            sampled_durations.unsqueeze(-1)
        )[batch_indices, action_indices]
        # a no-op's duration is never used, so it isn't part of the action
        is_op = self._op_mask[batch_indices, action_indices]  # type: ignore
        duration_log_prob = torch.where(is_op, duration_log_prob, 0.0)

        # Joint log probability is the sum (since they're independent given the action)
        joint_log_prob = discrete_log_prob + duration_log_prob

        return joint_log_prob

    def act_inference(self, observations):
        """Deterministic action selection for inference.

        Args:
            observations: Observations (num_envs, obs_dim)

        Returns:
            actions: Deterministic actions (num_envs, 2) where:
                     - Column 0: discrete action index (0-16)
                     - Column 1: mean duration value
        """
        return GaitnetActor.act_inference(self.actor, observations)

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate the value function.

        Args:
            critic_observations: Observations (num_envs, obs_dim)

        Returns:
            values: State values (num_envs, 1)
        """
        critic_observations = critic_observations["policy"]
        return self.critic(critic_observations)
