"""Observation noise for sim2real: the planner's view of the robot, corrupted the way a real
robot's state estimate and elevation map are.

Noise is applied once per planning step to the whole `gaitnet_core.state.Observation`, so
the state vector, the candidate footholds (their validity and heights) and the terrain
group all see the same corrupted world, as on hardware. Isaac Lab's per-term observation
noise can't do that: each group would draw its own. Terminations, rewards and privileged
observations keep reading the truth.

Contact, the gait timing (the controller's own schedule) and the commands are exact.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from isaaclab.utils import configclass

from gaitnet_core.state import Observation


@configclass
class ObservationNoiseCfg:
    """Half-widths of uniform noise, drawn fresh every planning step."""

    foot_pos: float = 0.005
    """Feet relative to the base (m): leg kinematics from joint encoders."""
    foot_vel: float = 0.1
    """Feet relative to the base (m/s): kinematics from differentiated encoders."""
    base_lin_vel: float = 0.1
    """Base velocity (m/s), from the state estimator."""
    base_ang_vel: float = 0.2
    """Base angular velocity (rad/s), from the IMU."""
    projected_gravity: float = 0.05
    """Gravity direction in the base frame (unitless), from the IMU."""
    terrain_offset: float = 0.01
    """One offset per leg's terrain patch (m): the elevation map drifting relative to the hip."""
    terrain_cell: float = 0.002
    """Per cell (m). Keep it well under half the foothold rules' step threshold (0.02 m), or
    neighbouring cells read as edges and footholds vanish."""


def _uniform(like: torch.Tensor, half_width: float) -> torch.Tensor:
    return (torch.rand_like(like) * 2 - 1) * half_width


def corrupt(observation: Observation, cfg: ObservationNoiseCfg) -> Observation:
    """`observation` with noise added (a new Observation; the input is left as it was)."""
    state = observation.state
    state = replace(
        state,
        foot_pos=state.foot_pos + _uniform(state.foot_pos, cfg.foot_pos),
        foot_vel=state.foot_vel + _uniform(state.foot_vel, cfg.foot_vel),
        base_lin_vel=state.base_lin_vel + _uniform(state.base_lin_vel, cfg.base_lin_vel),
        base_ang_vel=state.base_ang_vel + _uniform(state.base_ang_vel, cfg.base_ang_vel),
        projected_gravity=state.projected_gravity + _uniform(state.projected_gravity, cfg.projected_gravity),
    )
    heights = observation.terrain.heights
    offset = _uniform(heights[..., :1, :1], cfg.terrain_offset)  # (N, L, 1, 1)
    # unknown cells (-inf) stay unknown
    heights = heights + offset + _uniform(heights, cfg.terrain_cell)
    return Observation(state, replace(observation.terrain, heights=heights))
