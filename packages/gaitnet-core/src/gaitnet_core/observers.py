"""Planner -> controller feedback outside the learned policy.

An observer looks at a plan (the scores of every candidate and leg, not just the chosen
footstep, and how much valid terrain each leg has) and may nudge the controller, e.g. slow down when no leg has good footholds
ahead. Observers run in the deployment runtime and, optionally, during training (inside the
RSL-RL actor), where the nudge is part of the environment's dynamics rather than the
policy's action. Both places can give them the plan and the operator's command before any
nudge, so that is their whole input. In training the plan's `foothold_fraction` comes from
an observation group, and is None unless a preset turns that group on.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from gaitnet_core.interfaces import Nudge
from gaitnet_core.planner import PlanResult
from gaitnet_core.robot_spec import ROBOTS


class Observer(Protocol):
    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        """Clear per-robot memory (all robots if None), e.g. on episode reset."""
        ...

    def observe(self, plan: PlanResult, base_command: torch.Tensor) -> Nudge | None:
        """Return a nudge for this tick, or None for no change.

        Args:
            base_command: (N, 3) the operator's velocity command, before any nudge
        """
        ...


def combined_nudge(observers: Sequence[Observer], plan: PlanResult, base_command: torch.Tensor) -> Nudge:
    """Sum of every observer's nudge (zero if none respond)."""
    total = Nudge.zeros(base_command.shape[0], device=base_command.device)
    for observer in observers:
        nudge = observer.observe(plan, base_command)
        if nudge is not None:
            total = total + nudge
    return total


class StepConfidenceSlowdown:
    """Slow down while the policy keeps preferring to wait over every foothold it may take.

    On a tick where some leg may step, waiting wins if the no-op's score beats the best
    leg's marginal by more than `margin`. After `patience` such ticks in a row, the command
    is scaled by `scale` until a tick where a step is preferred. Ticks where no leg may step
    (the gait's own swings) neither count nor break the run.
    """

    def __init__(self, patience: int = 10, margin: float = 0.0, scale: float = 0.5):
        """
        Args:
            patience: planning ticks (25 Hz in the sim) of waiting before slowing down
            margin: log-odds by which waiting must win for a tick to count
            scale: factor on the operator's command while slowed
        """
        self.patience = patience
        self.margin = margin
        self.scale = scale
        self._waiting: torch.Tensor | None = None
        """(N,) long, consecutive ticks waiting won"""

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        if self._waiting is None:
            return
        if robot_ids is None:
            self._waiting = torch.zeros_like(self._waiting)
        else:
            # out of place, so memory made under torch.inference_mode can be reset outside it
            cleared = torch.zeros_like(self._waiting, dtype=torch.bool)
            cleared[robot_ids.to(self._waiting.device)] = True
            self._waiting = torch.where(cleared, 0, self._waiting)

    def observe(self, plan: PlanResult, base_command: torch.Tensor) -> Nudge:
        n = base_command.shape[0]
        if self._waiting is None or self._waiting.shape[0] != n or self._waiting.device != base_command.device:
            self._waiting = torch.zeros(n, dtype=torch.long, device=base_command.device)
        best = plan.leg_marginals.max(dim=-1).values  # -inf where no leg may step
        can_step = torch.isfinite(best)
        waits = can_step & (plan.scores.noop_logit - best > self.margin)
        steps = can_step & ~waits
        self._waiting = torch.where(waits, self._waiting + 1, torch.where(steps, 0, self._waiting))
        slowed = (self._waiting >= self.patience).unsqueeze(-1)
        return Nudge(command_delta=torch.where(slowed, (self.scale - 1.0) * base_command, 0.0))


class BlockedLegRedirect:
    """Steer the commanded xy velocity away from legs that are running out of footholds.

    A leg's blockage is how far the valid fraction of its foothold grid
    (`PlanResult.foothold_fraction`, terrain rules only) falls short of `full_fraction`,
    scaled to [0, 1]. It ignores whether the leg may step this tick, so a leg in swing still
    reports the terrain under it, and it doesn't depend on the sampler.

    Each leg pulls along the unit direction of its hip in the base frame, weighted by its
    blockage. The resultant gives one direction d and a strength s = min(|resultant|, 1). A
    fraction s of the command's component towards d is removed, so the robot slides past
    the blocked side instead of walking into it, plus an optional `push` away from it. Two
    blocked front legs stop forward motion but not a sideways walk; the yaw rate is left alone.
    """

    def __init__(self, robot: str = "go1", full_fraction: float = 1.0, push: float = 0.0):
        """
        Args:
            robot: key of `gaitnet_core.robot_spec.ROBOTS`, for the hip directions
            full_fraction: valid fraction of a leg's grid at or above which it is not blocked
            push: speed away from the blocked side at full strength (m/s)
        """
        hips = torch.tensor(ROBOTS[robot].hip_offsets)[:, :2]
        self.hip_directions = hips / hips.norm(dim=-1, keepdim=True)
        """(L, 2) unit direction of each hip from the base, base frame"""
        self.full_fraction = full_fraction
        self.push = push

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        pass

    def observe(self, plan: PlanResult, base_command: torch.Tensor) -> Nudge:
        if plan.foothold_fraction is None:
            raise ValueError(
                "blocked_leg_redirect needs the plan's foothold_fraction; in training, turn on the"
                " 'footholds' observation group (presets=redirect)"
            )
        fraction = plan.foothold_fraction.to(base_command.dtype)
        blocked = (1.0 - fraction / self.full_fraction).clamp(0.0, 1.0)
        resultant = blocked @ self.hip_directions.to(device=base_command.device, dtype=base_command.dtype)
        length = resultant.norm(dim=-1, keepdim=True)
        direction = resultant / length.clamp_min(1e-6)
        strength = length.clamp(max=1.0)
        towards = (base_command[:, :2] * direction).sum(dim=-1, keepdim=True).clamp_min(0.0)
        delta_xy = -strength * (towards + self.push) * direction
        return Nudge(command_delta=torch.cat([delta_xy, torch.zeros_like(base_command[:, 2:])], dim=-1))


OBSERVERS: dict[str, type] = {
    "step_confidence_slowdown": StepConfidenceSlowdown,
    "blocked_leg_redirect": BlockedLegRedirect,
}


def make_observers(specs: dict[str, dict]) -> list[Observer]:
    """Observers from `{<key of OBSERVERS>: kwargs}`, as bundles and cfgs store them. Keyed by
    name so a cfg override can reach one parameter (`...observers.<name>.<param>=...`)."""
    return [OBSERVERS[name](**(kwargs or {})) for name, kwargs in specs.items()]
