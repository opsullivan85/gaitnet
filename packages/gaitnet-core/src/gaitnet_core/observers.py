"""Planner -> controller feedback outside the learned policy.

An observer looks at a plan (the scores of every candidate and leg, not just the chosen
footstep) and may nudge the controller, e.g. slow down when no leg has good footholds
ahead. Observers run in the deployment runtime and, optionally, during training (inside the
RSL-RL actor), where the nudge is part of the environment's dynamics rather than the
policy's action. Both places can give them the plan and the operator's command before any
nudge, so that is their whole input.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from gaitnet_core.interfaces import Nudge
from gaitnet_core.planner import PlanResult


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


OBSERVERS: dict[str, type] = {
    "step_confidence_slowdown": StepConfidenceSlowdown,
}


def make_observers(specs: dict[str, dict]) -> list[Observer]:
    """Observers from `{<key of OBSERVERS>: kwargs}`, as bundles and cfgs store them. Keyed by
    name so a cfg override can reach one parameter (`...observers.<name>.<param>=...`)."""
    return [OBSERVERS[name](**(kwargs or {})) for name, kwargs in specs.items()]
