"""Named robot-state features, computed from a RobotState.

The simulator's observation terms and the deployment runtime both build the policy's
state vector through this registry, so a feature means the same thing in both places.
An experiment picks its state vector as a list of feature names; the list is saved in the
checkpoint bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from gaitnet_core.state import RobotState

MAX_STANCE_TIME_OBS = 0.5
"""Time since touchdown is clipped to this (s), since it is otherwise unbounded."""


@dataclass(frozen=True)
class Feature:
    fn: Callable[[RobotState], torch.Tensor]
    """RobotState -> (N, dim)"""
    dim_per_leg: int = 0
    dim_fixed: int = 0

    def dim(self, num_legs: int) -> int:
        return self.dim_fixed + self.dim_per_leg * num_legs


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)


def _gait_timing(state: RobotState) -> torch.Tensor:
    timing = state.gait_timing.clone()
    timing[..., 2] = timing[..., 2].clamp(max=MAX_STANCE_TIME_OBS)
    # feature grouped: [swing phase x L, remaining swing x L, time since touchdown x L]
    return _flat(timing.transpose(1, 2))


FEATURES: dict[str, Feature] = {
    # leg grouped (FL xyz, FR xyz, ...) unless noted
    "foot_pos": Feature(lambda s: _flat(s.foot_pos), dim_per_leg=3),
    "foot_pos_xy": Feature(lambda s: _flat(s.foot_pos[..., :2]), dim_per_leg=2),
    "foot_pos_z": Feature(lambda s: s.foot_pos[..., 2], dim_per_leg=1),
    "foot_vel": Feature(lambda s: _flat(s.foot_vel), dim_per_leg=3),
    "base_lin_vel": Feature(lambda s: s.base_lin_vel, dim_fixed=3),
    "base_ang_vel": Feature(lambda s: s.base_ang_vel, dim_fixed=3),
    "projected_gravity": Feature(lambda s: s.projected_gravity, dim_fixed=3),
    "contact": Feature(lambda s: s.contact.float(), dim_per_leg=1),
    "gait_timing": Feature(_gait_timing, dim_per_leg=3),
    "command": Feature(lambda s: s.command, dim_fixed=3),
}

DEFAULT_FEATURES: tuple[str, ...] = (
    "foot_pos",
    "base_lin_vel",
    "base_ang_vel",
    "command",
    "contact",
    "projected_gravity",
    "foot_vel",
    "gait_timing",
)


def feature_dim(names: tuple[str, ...] | list[str], num_legs: int) -> int:
    return sum(FEATURES[name].dim(num_legs) for name in names)


def state_vector(state: RobotState, names: tuple[str, ...] | list[str]) -> torch.Tensor:
    """(N, feature_dim) concatenation of the named features, in order."""
    return torch.cat([FEATURES[name].fn(state).float() for name in names], dim=-1)
