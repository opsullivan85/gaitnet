"""Where feet land against where the planner sent them: a `LowLevelController` wrapper that
follows every footstep from its command to touchdown, for `gaitnet_sim.scripts.landing_error`.

It sits between the footstep action term and its controller, forwarding every call, and reads
the scene through the term's `RobotIO` once per control step (250 Hz, not the 25 Hz planning
rate). Each footstep is recorded against the foothold as commanded, frozen in the world at the
moment of the command, so the error is what the planner experiences: it picked a patch of
ground, did the foot end up on it. Both controllers pin the foothold in the world from that
same state and aim the swing at it, so what is left is how well the swing tracks it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils

from gaitnet_core.interfaces import FootstepCommand, LowLevelController
from gaitnet_sim.robot import HIP_NAMES
from gaitnet_sim.robot_io import RobotIO

Rows = list[dict]
"""One dict per footstep; the columns are listed in `LandingProbe`'s field docstrings, vectors
split into `_x`, `_y`, `_z`."""

_POINTS = ("target", "commanded", "touchdown", "end", "settled")
_SCALARS = (
    "t",
    "duration",
    "yaw",
    "vx",
    "vy",
    "wz",
    "command_vx",
    "command_vy",
    "command_wz",
    "touchdown_t",
    "end_contact",
    "dyaw",
    "roll_end",
    "pitch_end",
    "vx_end",
    "vy_end",
)


class LandingProbe:
    """Implements `LowLevelController` by forwarding to `controller`, and records each footstep.

    Per footstep, points are (x, y, z) in the world frame, on the ground (the foot sphere's
    bottom), except `target`:

    - `target`: as commanded, in the leg's hip yaw frame (m)
    - `commanded`: the target in the world, at the hip and heading of the command
    - `touchdown`: the foot at its first contact after lifting off, from half the swing on
    - `end`: the foot at the scheduled touchdown, contact or not
    - `settled`: the foot `settle_s` after the scheduled touchdown

    and scalars: `t` command time (s since the probe started), `duration` (s); at the command,
    `yaw` (rad), base velocity `vx`, `vy` (m/s, base frame) and yaw rate `wz` (rad/s), and the
    velocity command `command_*`; `touchdown_t` contact time relative to the scheduled
    touchdown (s, negative is early); `end_contact` (0/1) at the scheduled touchdown; at the
    scheduled touchdown, `dyaw` the heading change since the command (rad), `roll_end`,
    `pitch_end` (rad) and base velocity `vx_end`, `vy_end`. `lifted` (0/1) whether the foot
    lost contact at all; `interrupted` (0/1) whether the leg was sent again before its record
    finished. Missing values are NaN.
    """

    def __init__(
        self,
        controller: LowLevelController,
        io: RobotIO,
        dt: float,
        foot_radius: float,
        device: torch.device | str,
        settle_s: float = 0.04,
        grace_s: float = 0.1,
    ):
        """
        Args:
            dt: control period (s), the interval between `compute_torques` calls
            foot_radius: the controllers' foot sphere radius (m); points are its bottom
            settle_s: how long after the scheduled touchdown `settled` is read (s)
            grace_s: how long after the scheduled touchdown a late contact still counts (s)
        """
        self.controller = controller
        self.io = io
        self.dt = dt
        self.device = torch.device(device)
        self.settle_s = settle_s
        self.grace_s = grace_s
        self.rows: Rows = []

        self._hip_ids, _ = io.robot.find_bodies(list(HIP_NAMES), preserve_order=True)
        self._down = torch.tensor([0.0, 0.0, foot_radius], device=self.device)
        n, legs = controller.num_robots, len(HIP_NAMES)
        self._step = 0
        self._start = torch.zeros(n, legs, dtype=torch.long, device=self.device)
        self._pending = torch.zeros(n, legs, dtype=torch.bool, device=self.device)
        self._lifted = torch.zeros(n, legs, dtype=torch.bool, device=self.device)
        self._points = {name: torch.full((n, legs, 3), math.nan, device=self.device) for name in _POINTS}
        self._scalars = {name: torch.full((n, legs), math.nan, device=self.device) for name in _SCALARS}
        self._velocity_command = torch.zeros(n, 3, device=self.device)

    @property
    def num_robots(self) -> int:
        return self.controller.num_robots

    def __getattr__(self, name: str):
        # anything beyond the protocol (estimated_rpy, ...) is the controller's
        return getattr(self.controller, name)

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        self.controller.reset(robot_ids)
        # a reset teleports the robot: whatever was in flight is not a landing
        self._pending[slice(None) if robot_ids is None else robot_ids] = False

    def gait_timing(self) -> torch.Tensor:
        return self.controller.gait_timing()

    def close(self) -> None:
        self.controller.close()

    def command_footsteps(self, footsteps: FootstepCommand) -> None:
        self.controller.command_footsteps(footsteps)
        legs = len(HIP_NAMES)
        sent = torch.nn.functional.one_hot(footsteps.leg.clamp(0, legs - 1), legs).bool()
        sent &= footsteps.active.unsqueeze(1)
        if not sent.any():
            return
        self._emit(self._pending & sent, interrupted=True)

        pose, vel = self.io.base_pose(), self.io.base_vel()
        quat = pose[:, 3:7]
        heading = math_utils.yaw_quat(quat).unsqueeze(1).expand(-1, legs, -1)
        target = footsteps.target.unsqueeze(1).expand(-1, legs, -1)
        commanded = self._hips() + math_utils.quat_apply(heading, target)
        base_vel = math_utils.quat_apply_inverse(quat, vel[:, :3])
        at_command = {
            "t": torch.full_like(footsteps.duration, self._step * self.dt),
            "duration": footsteps.duration,
            "yaw": math_utils.euler_xyz_from_quat(quat)[2],
            "vx": base_vel[:, 0],
            "vy": base_vel[:, 1],
            "wz": vel[:, 5],
            "command_vx": self._velocity_command[:, 0],
            "command_vy": self._velocity_command[:, 1],
            "command_wz": self._velocity_command[:, 2],
        }

        at_command_points = {"target": target, "commanded": commanded}
        for name, value in self._points.items():
            new = at_command_points.get(name, torch.full_like(value, math.nan))
            self._points[name] = torch.where(sent.unsqueeze(-1), new, value)
        for name, value in self._scalars.items():
            new = at_command[name].unsqueeze(1) if name in at_command else torch.full_like(value, math.nan)
            self._scalars[name] = torch.where(sent, new, value)
        self._start[sent] = self._step
        self._lifted[sent] = False
        self._pending |= sent

    def compute_torques(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_pose: torch.Tensor,
        base_vel: torch.Tensor,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        if self._pending.any():
            self._observe(base_pose, base_vel)
        self._velocity_command = velocity_command.clone()
        torques = self.controller.compute_torques(joint_pos, joint_vel, base_pose, base_vel, velocity_command)
        self._step += 1
        return torques

    def _hips(self) -> torch.Tensor:
        """(N, L, 3) hip positions, world frame, now (the scanners only refresh per plan)."""
        return self.io.robot.data.body_link_pos_w.torch[:, self._hip_ids]

    def _observe(self, base_pose: torch.Tensor, base_vel: torch.Tensor) -> None:
        pending = self._pending
        elapsed = (self._step - self._start).float() * self.dt
        duration = self._scalars["duration"]

        data = self.io.robot.data
        feet = data.body_link_pos_w.torch[:, self.io.foot_ids] - self._down
        forces = self.io.contact_sensor.data.net_normal_forces_w.torch[:, self.io.contact_ids]
        contact = forces.norm(dim=-1) > self.io.contact_threshold
        quat = base_pose[:, 3:7]
        base_vel_b = math_utils.quat_apply_inverse(quat, base_vel[:, :3])

        touchdown = (
            pending
            & self._lifted
            & contact
            & self._scalars["touchdown_t"].isnan()
            & (elapsed >= 0.5 * duration)
            & (elapsed < duration + self.grace_s)
        )
        self._points["touchdown"] = torch.where(touchdown.unsqueeze(-1), feet, self._points["touchdown"])
        self._scalars["touchdown_t"] = torch.where(touchdown, elapsed - duration, self._scalars["touchdown_t"])
        self._lifted |= pending & ~contact

        end = pending & self._scalars["end_contact"].isnan() & (elapsed >= duration)
        if end.any():
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
            at_end = {
                "end_contact": contact.float(),
                "dyaw": _wrap(yaw.unsqueeze(1) - self._scalars["yaw"]),
                "roll_end": _wrap(roll).unsqueeze(1),
                "pitch_end": _wrap(pitch).unsqueeze(1),
                "vx_end": base_vel_b[:, :1],
                "vy_end": base_vel_b[:, 1:2],
            }
            for name, value in at_end.items():
                self._scalars[name] = torch.where(end, value, self._scalars[name])
            self._points["end"] = torch.where(end.unsqueeze(-1), feet, self._points["end"])

        settled = pending & self._points["settled"][..., 0].isnan() & (elapsed >= duration + self.settle_s)
        self._points["settled"] = torch.where(settled.unsqueeze(-1), feet, self._points["settled"])

        self._emit(pending & (elapsed >= duration + max(self.settle_s, self.grace_s)), interrupted=False)

    def _emit(self, done: torch.Tensor, interrupted: bool) -> None:
        """Turn the `done` (N, L) records into rows and stop following them."""
        if not done.any():
            return
        env, leg = done.nonzero(as_tuple=True)
        columns: dict[str, torch.Tensor] = {"env": env, "leg": leg}
        for name, value in self._points.items():
            for axis, component in zip("xyz", value[env, leg].unbind(-1)):
                columns[f"{name}_{axis}"] = component
        for name, value in self._scalars.items():
            columns[name] = value[env, leg]
        columns["lifted"] = self._lifted[env, leg].float()
        columns["interrupted"] = torch.full_like(env, int(interrupted))
        host = {name: value.cpu().tolist() for name, value in columns.items()}
        self.rows.extend(dict(zip(host, values)) for values in zip(*host.values()))
        self._pending &= ~done


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def write_csv(rows: Rows, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Rows) -> list[str]:
    """Report lines: each error's size and its bias along and across the heading at the
    command (along is forward, lateral is to the left)."""
    if not rows:
        return ["no footsteps recorded"]

    def column(name: str) -> torch.Tensor:
        return torch.tensor([row[name] for row in rows], dtype=torch.float64)

    def point(name: str) -> torch.Tensor:
        return torch.stack([column(f"{name}_{axis}") for axis in "xyz"], dim=-1)

    yaw = column("yaw")
    cos, sin = yaw.cos(), yaw.sin()
    lines = [
        (
            f"{len(rows)} footsteps; lifted off {column('lifted').mean():.1%}, contact at the scheduled"
            f" touchdown {column('end_contact').nanmean():.1%}, interrupted {column('interrupted').mean():.1%}"
        )
    ]
    timing = column("touchdown_t")
    found = timing.isfinite()
    if found.any():
        q = torch.quantile(timing[found], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
        lines.append(
            f"first contact vs scheduled touchdown (s, <0 early), {found.float().mean():.1%} found:"
            f" 5% {q[0]:+.3f}, median {q[1]:+.3f}, 95% {q[2]:+.3f}"
        )
    lines.append(f"{'error (m)':<24}{'n':>6}{'|xy| med':>10}{'p90':>8}{'p99':>8}{'along':>9}{'lateral':>9}{'z med':>9}")
    for label, a, b in (
        ("touchdown - commanded", "touchdown", "commanded"),
        ("end - commanded", "end", "commanded"),
        ("settled - commanded", "settled", "commanded"),
    ):
        d = point(a) - point(b)
        ok = d.isfinite().all(dim=-1)
        if not ok.any():
            lines.append(f"{label:<24}{0:>6}")
            continue
        d, c, s = d[ok], cos[ok], sin[ok]
        along = c * d[:, 0] + s * d[:, 1]
        lateral = -s * d[:, 0] + c * d[:, 1]
        size = d[:, :2].norm(dim=-1)
        q = torch.quantile(size, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
        lines.append(
            f"{label:<24}{int(ok.sum()):>6}{q[0]:>10.4f}{q[1]:>8.4f}{q[2]:>8.4f}"
            f"{along.mean():>+9.4f}{lateral.mean():>+9.4f}{d[:, 2].median():>+9.4f}"
        )
    return lines
