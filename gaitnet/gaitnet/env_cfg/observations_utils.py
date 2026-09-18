from __future__ import annotations
import numpy as np
import gaitnet.constants as const


import torch


def get_terrain_mask(
    valid_height_range: tuple[float, float], obs: torch.Tensor
) -> torch.Tensor:
    """Get a mask for the terrain observations.

    0 indicates invalid terrain (too high or too low)
    1 indicates valid terrain
    """
    terrain_terms = (
        const.footstep_scanner.grid_size[0] * const.footstep_scanner.grid_size[1] * const.robot.num_legs
    )
    terrain_obs = obs[:, -terrain_terms:]
    # reshape to (N, 4, H, W)
    terrain_obs = terrain_obs.reshape(
        terrain_obs.shape[0],
        const.robot.num_legs,
        const.footstep_scanner.grid_size[0],
        const.footstep_scanner.grid_size[1],
    )
    # mask out values outside of allowed height range
    min_height, max_height = valid_height_range
    terrain_mask = (terrain_obs > min_height) & (terrain_obs < max_height)
    return terrain_mask


robot_state_layout: dict[str, tuple[int, int]] = {
    "foot_position_xy_b": (0, 8),  # leg grouped xy
    "base_pos_z": (8, 9),
    "base_lin_vel": (9, 12),
    "base_ang_vel": (12, 15),
    "control": (15, 18),
    "contact_state_sensor": (18, 22),  # measured in sim
    "projected_gravity": (22, 25),
    "foot_position_z_b": (25, 29),
    "foot_velocity_b": (29, 41),  # relative to base, leg grouped xyz
    "gait_timing_controller": (41, 53),  # swing phase, remaining swing time, time since touchdown
}
"""(start, end) of each policy observation term in the robot state, in term order.
Per-leg values are in FL, FR, RL, RR order. The footstep scanner terms follow.
GaitNetObservationManager checks this against the configured terms."""

footstep_scanner_terms = ["FL_foot_scanner", "FR_foot_scanner", "RL_foot_scanner", "RR_foot_scanner"]
"""Policy observation terms after the robot state. Their order is the leg order of the
terrain channels, and so of the footstep options: FL, FR, RL, RR."""

contact_state_indices = np.arange(*robot_state_layout["contact_state_sensor"])
"""Measured contact state. For whether the controller considers a leg in stance, use `scheduled_contact`."""
_gait_timing_start = robot_state_layout["gait_timing_controller"][0]
swing_phase_indices = np.arange(_gait_timing_start, _gait_timing_start + 4)
swing_time_remaining_indices = np.arange(_gait_timing_start + 4, _gait_timing_start + 8)
stance_time_indices = np.arange(_gait_timing_start + 8, _gait_timing_start + 12)


def scheduled_contact(obs: torch.Tensor) -> torch.Tensor:
    """Whether the controller's schedule has each leg in stance, (num_envs, 4) bool.

    This is what decides whether a leg can be given a new footstep, and can
    disagree with the measured contact state (e.g. early touchdown on rough terrain).
    """
    return obs[:, swing_time_remaining_indices] <= 0