"""Named robot-state features, computed from a RobotState.

The simulator's observation terms and the deployment runtime both build the policy's
state vector through this registry, so a feature means the same thing in both places.
An experiment picks its state vector as a list of feature names; the list is saved in the
checkpoint bundle.

Every feature also says how it transforms under the robot's left/right mirror, which
`gaitnet_core.symmetry` uses to mirror a state vector for symmetry-augmented training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    mirror: tuple[float, ...] = field(kw_only=True)
    """Sign of each component under the left/right mirror (y -> -y): one per dimension of a
    leg for per-leg features, whose legs also trade places with their mirror image, else one
    per dimension."""
    leg_major: bool = True
    """Per-leg layout: leg-major (FL xyz, FR xyz, ...), or feature-major (every leg's first
    component, then every leg's second, ...)."""

    def dim(self, num_legs: int) -> int:
        return self.dim_fixed + self.dim_per_leg * num_legs


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)


def _gait_timing(state: RobotState) -> torch.Tensor:
    timing = state.gait_timing.clone()
    timing[..., 2] = timing[..., 2].clamp(max=MAX_STANCE_TIME_OBS)
    # feature grouped: [swing phase x L, remaining swing x L, time since touchdown x L]
    return _flat(timing.transpose(1, 2))


_VECTOR = (1.0, -1.0, 1.0)
"""A vector's mirror flips its y. An angular velocity, a pseudovector, flips x and z instead."""
_ANGULAR = (-1.0, 1.0, -1.0)

FEATURES: dict[str, Feature] = {
    # leg grouped (FL xyz, FR xyz, ...) unless noted
    "foot_pos": Feature(lambda s: _flat(s.foot_pos), dim_per_leg=3, mirror=_VECTOR),
    "foot_pos_xy": Feature(lambda s: _flat(s.foot_pos[..., :2]), dim_per_leg=2, mirror=_VECTOR[:2]),
    "foot_pos_z": Feature(lambda s: s.foot_pos[..., 2], dim_per_leg=1, mirror=(1.0,)),
    "foot_vel": Feature(lambda s: _flat(s.foot_vel), dim_per_leg=3, mirror=_VECTOR),
    "base_lin_vel": Feature(lambda s: s.base_lin_vel, dim_fixed=3, mirror=_VECTOR),
    "base_ang_vel": Feature(lambda s: s.base_ang_vel, dim_fixed=3, mirror=_ANGULAR),
    "projected_gravity": Feature(lambda s: s.projected_gravity, dim_fixed=3, mirror=_VECTOR),
    "contact": Feature(lambda s: s.contact.float(), dim_per_leg=1, mirror=(1.0,)),
    "gait_timing": Feature(_gait_timing, dim_per_leg=3, mirror=(1.0, 1.0, 1.0), leg_major=False),
    # (vx, vy, yaw rate)
    "command": Feature(lambda s: s.command, dim_fixed=3, mirror=(1.0, -1.0, -1.0)),
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
