"""Reward terms that depend on the footstep action term (the nudged command, the step
taken) or that isaaclab's stock mdp doesn't provide."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from gaitnet_sim.env.observations import footstep_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: "ManagerBasedRLEnv", command: Literal["effective", "base"], action_name: str) -> torch.Tensor:
    term = footstep_action(env, action_name)
    return term.effective_command() if command == "effective" else term.base_command()


def track_lin_vel_xy_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command: Literal["effective", "base"] = "effective",
    action_name: str = "footstep",
) -> torch.Tensor:
    """exp(-|v_xy - command_xy|^2 / std^2), base frame.

    Args:
        command: track the nudged command the controller follows ("effective") or the
            command before the nudge ("base").
    """
    velocity = footstep_action(env, action_name).io.robot.data.root_link_lin_vel_b.torch[:, :2]
    error = torch.sum(torch.square(_command(env, command, action_name)[:, :2] - velocity), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command: Literal["effective", "base"] = "effective",
    action_name: str = "footstep",
) -> torch.Tensor:
    """exp(-(yaw rate - commanded yaw rate)^2 / std^2)."""
    yaw_rate = footstep_action(env, action_name).io.robot.data.root_link_ang_vel_b.torch[:, 2]
    error = torch.square(_command(env, command, action_name)[:, 2] - yaw_rate)
    return torch.exp(-error / std**2)


def step_taken(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """1 where the policy started a footstep this env step (rather than the no-op)."""
    return footstep_action(env, action_name).footsteps.active.float()


def foot_slip(env: "ManagerBasedRLEnv", threshold: float = 1.0, action_name: str = "footstep") -> torch.Tensor:
    """Sum over feet in contact of their horizontal speed (m/s)."""
    io = footstep_action(env, action_name).io
    forces = io.contact_sensor.data.net_normal_forces_w.torch[:, io.contact_ids]
    in_contact = forces.norm(dim=-1) > threshold
    speed = io.robot.data.body_link_lin_vel_w.torch[:, io.foot_ids, :2].norm(dim=-1)
    return torch.sum(in_contact * speed, dim=1)
