"""The GaitNet actor as an RSL-RL model.

RSL-RL (5.x) builds each model as `class_name(obs, obs_groups, obs_set, output_dim, **cfg)`
and PPO uses only the model's forward pass, its log-probability, entropy and distribution
parameters. This model wraps any `gaitnet_core` scoring network and owns the candidate
distribution (`gaitnet_core.selection.FootstepDistribution`), which needs the candidate set
from the observation and so can't be one of RSL-RL's output distributions.

Actions follow `gaitnet_core.action_layout`: the choice (candidate index, duration) that
log-probabilities are computed from, and the footstep it resolves to, which the environment
executes. The nudge fields are zero; feedback observers add theirs outside the policy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from gaitnet_core import action_layout
from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.candidates import Candidates
from gaitnet_core.networks import build_network
from gaitnet_core.selection import FootstepDistribution, Selection


class GaitNetActor(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        network: dict,
        candidates_group: str = "candidates",
        terrain_group: str | None = None,
        duration_std: float = 0.05,
        distribution_cfg: dict | None = None,
    ):
        """
        Args:
            obs_groups: `obs_groups[obs_set]` are the groups concatenated into the network's
                state vector.
            network: `{"class_name": <key of gaitnet_core.networks.NETWORKS>, **kwargs}`.
                `state_dim` is taken from the observations if not given.
            candidates_group: the group holding packed candidates, (N, L, K, 5)
            terrain_group: the group holding terrain patches, for networks that use them
            duration_std: initial swing duration noise (s), then learned
            distribution_cfg: must be None. Isaac Lab's runner cfg gives every model this key;
                this model's distribution is fixed by the candidates.
        """
        super().__init__()
        if distribution_cfg is not None:
            raise ValueError("GaitNetActor owns its distribution, leave distribution_cfg as None")
        if output_dim != action_layout.DIM:
            raise ValueError(f"GaitNetActor emits {action_layout.DIM}-dim actions, the environment expects {output_dim}")
        self.state_groups = list(obs_groups[obs_set])
        for group in self.state_groups:
            if obs[group].dim() != 2:
                raise ValueError(f"state group '{group}' must be (N, D), got {tuple(obs[group].shape)}")
        state_dim = sum(obs[group].shape[-1] for group in self.state_groups)

        config = dict(network)
        self.network_class = config.pop("class_name")
        if config.setdefault("state_dim", state_dim) != state_dim:
            raise ValueError(f"network state_dim {config['state_dim']} != {state_dim} from groups {self.state_groups}")
        self.network = build_network(self.network_class, config)
        self.candidates_group = candidates_group
        self.terrain_group = terrain_group
        # log-parameterized so it stays positive
        self.duration_log_std = nn.Parameter(torch.tensor(math.log(duration_std)))
        self.distribution: FootstepDistribution | None = None

    @property
    def duration_std(self) -> torch.Tensor:
        return self.duration_log_std.exp()

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """(N, action_layout.DIM) actions, sampled or deterministic (two-stage select)."""
        state = torch.cat([obs[group] for group in self.state_groups], dim=-1)
        candidates = Candidates.unpack(obs[self.candidates_group])
        terrain = obs[self.terrain_group] if self.terrain_group is not None else None
        scores = self.network(state, candidates, terrain)
        self.distribution = FootstepDistribution(scores, candidates, self.duration_std)
        selection = self.distribution.sample() if stochastic_output else self.distribution.deterministic()
        return encode_selection(selection, candidates)

    def _require_distribution(self) -> FootstepDistribution:
        if self.distribution is None:
            raise RuntimeError("call the model on an observation first")
        return self.distribution

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        action = EnvAction.decode(outputs)
        return self._require_distribution().log_prob(Selection(index=action.choice_index, duration=action.duration))

    @property
    def output_entropy(self) -> torch.Tensor:
        return self._require_distribution().entropy()

    @property
    def output_mean(self) -> torch.Tensor:
        distribution = self._require_distribution()
        return encode_selection(distribution.deterministic(), distribution.candidates)

    @property
    def output_std(self) -> torch.Tensor:
        """(N, 1) the swing duration noise; the discrete choice has no std."""
        n = self._require_distribution().candidates.num_robots
        return self.duration_std.detach().expand(n, 1)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """(log-probabilities over the flat action index, per-entry duration mean, duration std)."""
        distribution = self._require_distribution()
        n = distribution.candidates.num_robots
        return (
            distribution.categorical.logits,
            distribution._duration_mean,
            self.duration_std.expand(n, 1),
        )

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """(N,) KL(old || new): the categorical choice, plus each step's duration KL weighted by
        the old probability of taking it."""
        old_logp, old_mean, old_std = old_params
        new_logp, new_mean, new_std = new_params
        old_p = old_logp.exp()
        # entries the old policy never takes contribute nothing; avoid -inf - -inf
        categorical = torch.where(old_p > 0, old_p * (old_logp - new_logp), torch.zeros_like(old_p)).sum(-1)
        duration = (
            torch.log(new_std / old_std) + (old_std**2 + (old_mean - new_mean) ** 2) / (2 * new_std**2) - 0.5
        )
        # the last entry is the no-op, which has no duration
        duration = (old_p[:, :-1] * duration[:, :-1]).sum(-1)
        return categorical + duration

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("export a policy bundle instead, see gaitnet_sim.scripts.export_bundle")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("export a policy bundle instead, see gaitnet_sim.scripts.export_bundle")


def encode_selection(selection: Selection, candidates: Candidates) -> torch.Tensor:
    """The action vector for a selection: the choice plus the footstep it resolves to."""
    is_step, leg, target = candidates.gather(selection.index)
    return EnvAction(
        choice_index=selection.index,
        duration=selection.duration,
        leg=torch.where(is_step, leg, torch.full_like(leg, NO_STEP_LEG)),
        target=target,
        nudge=torch.zeros_like(target),
    ).encode()
