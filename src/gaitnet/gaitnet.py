from typing import Any
import torch
import torch.nn as nn

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


class GaitnetActor(nn.Module):
    def __init__(
        self,
        shared_state_dim: int,
        shared_layer_sizes: Sequence[int],
        unique_state_dim: int,
        unique_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
    ):
        super().__init__()
        logger.info("GaitnetActor initializing")

        self.shared_encoder = make_mlp(
            input_size=shared_state_dim,
            hidden_sizes=shared_layer_sizes[:-1],
            output_size=shared_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        logger.info(f"shared_encoder: {self.shared_encoder}")

        self.unique_encoder = make_mlp(
            input_size=unique_state_dim,
            hidden_sizes=unique_layer_sizes[:-1],
            output_size=unique_layer_sizes[-1],
            output_activation=nn.ReLU,
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
            output_activation=nn.ReLU,
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
        unique_states_iter = torch.split(unique_states, 1, dim=1)

        shared_embedding: torch.Tensor = self.shared_encoder(shared_state)

        unique_embeddings = []
        for unique_state in unique_states_iter:
            # remove the extra dimension
            unique_state = unique_state.squeeze(1)
            # check if this is a no-op state
            # note that the one hot encoding is [no_op, leg1, leg2, leg3, leg4]
            no_op_mask = unique_state[:, 0] == 1

            unique_embedding = torch.zeros(
                (num_envs, self.unique_embedding_size), device=obs.device
            )
            if (~no_op_mask).any():
                unique_embedding[~no_op_mask] = self.unique_encoder(
                    unique_state[~no_op_mask]
                )
            if no_op_mask.any():
                unique_embedding[no_op_mask] = self.no_op_embedding
            unique_embeddings.append(unique_embedding)

        unique_embeddings = torch.stack(unique_embeddings, dim=1)
        trunk_input = torch.cat(
            [
                shared_embedding.unsqueeze(dim=1).expand(-1, unique_states_dim, -1),
                unique_embeddings,
            ],
            dim=-1,
        )
        trunk_output = self.trunk(trunk_input)

        logits = self.value_head(trunk_output).squeeze(-1)  # (num_envs, num_options)

        # only calculate durations for non no-op options
        duration = torch.zeros((num_envs, unique_states_dim), device=obs.device)
        no_op_mask = unique_states[:, :, 0] == 1
        duration[~no_op_mask] = self.duration_head(trunk_output[~no_op_mask]).squeeze(
            -1
        )  # (num_envs, num_options)

        # Scale durations to reasonable range
        min_dur, max_dur = const.gait_net.valid_swing_duration_range
        scale = max_dur - min_dur
        duration[~no_op_mask] = torch.sigmoid(duration[~no_op_mask]) * scale + min_dur

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

        # Select action with highest logit (deterministic)
        action_index = torch.argmax(logits, dim=-1)  # (num_envs,)

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
        num_unique_states: int,
        unique_state_dim: int,
        unique_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
        trunk_combiner_head_sizes: Sequence[int] = [32, 32],
    ):
        super().__init__()
        logger.info("GaitnetCritic initializing")

        self.shared_encoder = make_mlp(
            input_size=shared_state_dim,
            hidden_sizes=shared_layer_sizes[:-1],
            output_size=shared_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        logger.info(f"shared_encoder: {self.shared_encoder}")

        self.unique_encoder = make_mlp(
            input_size=unique_state_dim,
            hidden_sizes=unique_layer_sizes[:-1],
            output_size=unique_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        self.unique_embedding_size = unique_layer_sizes[-1]
        # random embedding to represent no-op
        self.no_op_embedding = nn.Parameter(torch.randn(unique_layer_sizes[-1]))
        logger.info(f"unique_encoder: {self.unique_encoder}")

        trunk_input_dim = shared_layer_sizes[-1] + unique_layer_sizes[-1]
        self.trunk = make_mlp(
            input_size=trunk_input_dim,
            hidden_sizes=trunk_layer_sizes,
            output_size=1,
        )
        logger.info(f"trunk: {self.trunk}")

        self.trunk_combiner_head = make_mlp(
            input_size=num_unique_states,
            hidden_sizes=trunk_combiner_head_sizes,
            output_size=1,
        )
        logger.info(f"trunk_combiner_head: {self.trunk_combiner_head}")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the MiMo network.

        Args:
            obs (torch.Tensor): Input observations.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Value and duration predictions.
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
        unique_states_iter = torch.split(unique_states, 1, dim=1)

        shared_embedding: torch.Tensor = self.shared_encoder(shared_state)

        unique_embeddings = []
        for unique_state in unique_states_iter:
            # remove the extra dimension
            unique_state = unique_state.squeeze(1)
            # check if this is a no-op state
            no_op_mask = unique_state[:, 4] == 1
            # also treat high cost as no-op
            no_op_mask = no_op_mask | (unique_state[:, -1] >= 2.0)

            unique_embedding = torch.zeros(
                (num_envs, self.unique_embedding_size), device=obs.device
            )
            if (~no_op_mask).any():
                unique_embedding[~no_op_mask] = self.unique_encoder(
                    unique_state[~no_op_mask]
                )
            if no_op_mask.any():
                unique_embedding[no_op_mask] = self.no_op_embedding
            unique_embeddings.append(unique_embedding)

        unique_embeddings = torch.stack(unique_embeddings, dim=1)
        trunk_input = torch.cat(
            [
                shared_embedding.unsqueeze(dim=1).expand(-1, unique_states_dim, -1),
                unique_embeddings,
            ],
            dim=-1,
        )
        values = self.trunk(trunk_input).squeeze(-1)  # (num_envs, num_unique_states)
        # collapse the values across the unique states to a single value
        # in theory this part should learn how we are using the output of the actor (single action selection)
        value = self.trunk_combiner_head(values)  # (num_envs, 1)

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
        duration_std: float = 0.01,  # Standard deviation for duration noise
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
            duration_std (float, optional): _description_. Defaults to 0.01.
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

        # Duration standard deviation (fixed for now, could be learned)
        self.duration_std = duration_std

        # Cache for storing selected action indices and durations
        self._last_action_indices: torch.Tensor | None = None
        self._last_sampled_durations: torch.Tensor | None = None
        self._cached_duration_means: torch.Tensor | None = None

    def reset(self, dones=None):
        """Reset recurrent states (no-op for non-recurrent policy)."""
        pass

    def forward(self):
        raise NotImplementedError

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

        # Duration std (constant for all options)
        duration_std = torch.full((num_envs, 1), self.duration_std, device=device)

        return torch.cat([discrete_std, duration_std], dim=-1)

    @property
    def entropy(self):
        """Return the entropy of the joint distribution.

        For independent discrete and continuous components, total entropy is the sum.
        """
        if self.discrete_distribution is None or self.duration_distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution first."
            )

        discrete_entropy = self.discrete_distribution.entropy()  # (num_envs,)
        # For duration, sum entropy across all options (since we condition on action selection)
        duration_entropy = self.duration_distribution.entropy().mean(
            dim=-1
        )  # (num_envs,)

        return discrete_entropy + duration_entropy

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

        # Cache duration means for later use
        self._cached_duration_means = duration_means

        # remove all but the last no-op so it doesn't affect probabilities
        num_options = logits.shape[1]
        # Reshape to get per-option state
        action_candidates = observations[:, const.gait_net.robot_state_dim :].view(
            observations.shape[0], num_options, const.gait_net.footstep_option_dim
        )
        # Mask: True for valid actions, False for no-ops or high cost
        no_op_mask = action_candidates[:, :, 0] == 1
        # Apply mask to logits (set invalid actions to -inf so they get 0 probability)
        masked_logits = logits.clone()
        masked_logits[no_op_mask] = float("-inf")
        masked_logits[:, -1] = logits[:, -1]  # always allow the last no-op option

        self.discrete_distribution = Categorical(logits=masked_logits)

        # Create continuous distribution for durations
        duration_std = torch.full_like(duration_means, self.duration_std)
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
