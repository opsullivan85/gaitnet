"""The controller's contact schedule, batched over robots.

There is no gait here, periodic or otherwise: a leg is in swing exactly when a footstep
the planner commanded is still running, and in stance the rest of the time. That is the
whole point of the planner, and it is why the schedule can say when a swing *ends* but
never when the next one begins - nothing has decided yet. The MPC's contact table over
the horizon therefore assumes every stance foot stays down, which is what
`CalculatedGait` in `gaitnet_mpc` does.

This is the controller's plan, not a measurement. A foot that hits the ground early is
still scheduled as swinging until its commanded touchdown, and the planner's step
eligibility rules depend on that (see `gaitnet_core.state.RobotState.gait_timing`).
"""

from __future__ import annotations

import torch


class GaitSchedule:
    """Per-leg swing timing for N robots.

    A leg is in swing over `[swing_start, swing_start + swing_duration)`. Both are zero
    for a leg that has never stepped, which reads as stance from time zero.
    """

    def __init__(
        self,
        num_robots: int,
        num_legs: int,
        control_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.control_dt = control_dt
        self.swing_start = torch.zeros(num_robots, num_legs, device=device, dtype=dtype)
        """(N, L) time each leg's current or last swing began (s)."""
        self.swing_duration = torch.zeros(num_robots, num_legs, device=device, dtype=dtype)
        """(N, L) how long that swing lasts, kept after touchdown (s)."""
        self.iteration = torch.zeros(num_robots, device=device, dtype=torch.long)
        """(N,) control steps completed since this robot was last reset."""
        self.time = torch.zeros(num_robots, device=device, dtype=dtype)
        """(N,) the clock as of the start of the control step now running (s).

        Held rather than derived from `iteration` because a footstep commanded between
        two steps is stamped with the step that has already run, not the one about to,
        and the CPU controller's swing phases are built on that. Counted in whole steps
        so a long episode cannot drift away from the steps the swings were timed with.
        """

    @property
    def touchdown(self) -> torch.Tensor:
        """(N, L) scheduled end of each leg's swing (s)."""
        return self.swing_start + self.swing_duration

    def begin_step(self) -> None:
        """Move the clock to the control step that is about to run."""
        self.time = self.iteration.to(self.swing_start.dtype) * self.control_dt

    def end_step(self) -> None:
        """Count the control step that has just run."""
        self.iteration += 1

    def initiate(self, active: torch.Tensor, leg: torch.Tensor, duration: torch.Tensor) -> None:
        """Start a swing of `leg` lasting `duration` s, for the robots where `active`.

        Args:
            active: (N,) bool, False where this robot takes no new step.
            leg: (N,) long leg index, ignored where inactive.
            duration: (N,) swing duration (s), ignored where inactive.
        """
        index = leg.unsqueeze(-1)
        now = self.time
        held_start = self.swing_start.gather(1, index).squeeze(-1)
        # `now` is the previous step's clock, which is what the CPU controller stamps
        # with: its gait clock only moves when a control step starts
        held_duration = self.swing_duration.gather(1, index).squeeze(-1)
        started = torch.where(active, now, held_start)
        lasting = torch.where(active, duration.to(held_duration.dtype), held_duration)
        self.swing_start.scatter_(1, index, started.unsqueeze(-1))
        self.swing_duration.scatter_(1, index, lasting.unsqueeze(-1))

    def in_contact(self) -> torch.Tensor:
        """(N, L) bool, whether the schedule has each leg on the ground now."""
        return self.touchdown <= self.time.unsqueeze(-1)

    def swing_phase(self) -> torch.Tensor:
        """(N, L) progress through the current swing in [0, 1), 0 for a leg in stance."""
        duration = self.swing_duration
        elapsed = self.time.unsqueeze(-1) - self.swing_start
        ratio = elapsed / duration.clamp_min(1e-9)
        phase = torch.where(duration > 0, ratio, torch.zeros_like(elapsed)).clamp(0.0, 1.0)
        # a finished swing reads as 0, the same value as a leg that never stepped
        return torch.where(phase >= 1.0, torch.zeros_like(phase), phase)

    def contact_table(self, horizon: int, timestep: float) -> torch.Tensor:
        """(N, H, L) scheduled contact over the MPC's horizon, 1 in stance.

        Legs already down stay down: no future liftoff is known when the table is built.
        """
        steps = torch.arange(horizon, device=self.swing_start.device, dtype=self.swing_start.dtype)
        times = self.time.unsqueeze(-1) + steps * timestep
        return (self.touchdown.unsqueeze(1) <= times.unsqueeze(-1)).to(self.swing_start.dtype)

    def timing(self) -> torch.Tensor:
        """(N, L, 3) the scheduled timing the planner observes.

        Columns are swing phase in [0, 1] (0 in stance), remaining swing time (s, 0 in
        stance) and time since scheduled touchdown (s, 0 in swing).
        """
        now = self.time.unsqueeze(-1)
        touchdown = self.touchdown
        stance = touchdown <= now
        zero = torch.zeros_like(touchdown)
        elapsed = now - self.swing_start
        phase = torch.where(
            self.swing_duration > 0, elapsed / self.swing_duration.clamp_min(1e-9), zero
        ).clamp(0.0, 1.0)
        return torch.stack(
            [
                torch.where(stance, zero, phase),
                torch.where(stance, zero, (touchdown - now).clamp_min(0.0)),
                torch.where(stance, (now - touchdown).clamp_min(0.0), zero),
            ],
            dim=-1,
        )

    def reset(self, robot_ids: torch.Tensor) -> None:
        """Put the named robots back to "standing, never stepped, time zero"."""
        self.swing_start[robot_ids] = 0.0
        self.swing_duration[robot_ids] = 0.0
        self.iteration[robot_ids] = 0
        self.time[robot_ids] = 0.0
