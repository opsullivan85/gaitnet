"""Planner -> controller feedback outside the learned policy.

An observer looks at a plan (the scores of every candidate and leg, not just the chosen
footstep) and may nudge the controller, e.g. slow down when no leg has good footholds
ahead. Observers run in the deployment runtime and, optionally, during training, where
the nudge is part of the environment's dynamics rather than the policy's action.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from gaitnet_core.interfaces import Nudge
from gaitnet_core.planner import PlanResult
from gaitnet_core.state import Observation


class Observer(Protocol):
    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        """Clear any per-robot memory (e.g. on episode reset)."""
        ...

    def observe(self, plan: PlanResult, observation: Observation) -> Nudge | None:
        """Return a nudge for this tick, or None for no change."""
        ...


def combined_nudge(observers: Sequence[Observer], plan: PlanResult, observation: Observation) -> Nudge:
    """Sum of every observer's nudge (zero if none respond)."""
    total = Nudge.zeros(observation.num_robots, device=plan.target.device)
    for observer in observers:
        nudge = observer.observe(plan, observation)
        if nudge is not None:
            total = total + nudge
    return total
