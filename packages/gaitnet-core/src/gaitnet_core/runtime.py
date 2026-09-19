"""The deployment loop: observe, plan, run observers, command, at a fixed rate."""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

import torch

from gaitnet_core.interfaces import RobotInterface
from gaitnet_core.observers import Observer, combined_nudge
from gaitnet_core.planner import FootstepPlanner, PlanResult
from gaitnet_core.state import Observation

logger = logging.getLogger(__name__)


class PlannerRuntime:
    def __init__(
        self,
        robot: RobotInterface,
        planner: FootstepPlanner,
        observers: Sequence[Observer] = (),
        rate_hz: float = 25.0,
        deterministic: bool = True,
        postprocess: Callable[[PlanResult, Observation], PlanResult] | None = None,
    ):
        """
        Args:
            postprocess: applied to each plan (with the observation it was planned from)
                before the observers see it, e.g. continuous refinement
                (`gaitnet_core.refine.Refiner`)
        """
        self.robot = robot
        self.planner = planner
        self.observers = list(observers)
        self.period = 1.0 / rate_hz
        self.deterministic = deterministic
        self.postprocess = postprocess
        self.overruns = 0

    def step(self) -> PlanResult:
        """One planning tick."""
        observation = self.robot.observe().to(next(self.planner.network.parameters()).device)
        plan = self.planner.plan(observation, deterministic=self.deterministic)
        if self.postprocess is not None:
            plan = self.postprocess(plan, observation)
        nudge = combined_nudge(self.observers, plan, observation.state.base_command)
        self.robot.command(plan.footstep_command(), nudge)
        return plan

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        """Clear the observers' per-robot memory, for robots starting over."""
        for observer in self.observers:
            observer.reset(robot_ids)

    def run(self, max_ticks: int | None = None, should_stop: Callable[[], bool] = lambda: False) -> int:
        """Tick at the configured rate until `max_ticks` or `should_stop()`. Returns ticks run."""
        ticks = 0
        next_tick = time.monotonic()
        while (max_ticks is None or ticks < max_ticks) and not should_stop():
            self.step()
            ticks += 1
            next_tick += self.period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self.overruns += 1
                logger.warning(f"planning tick overran its {self.period * 1e3:.0f} ms period by {-delay * 1e3:.1f} ms")
                next_tick = time.monotonic()
        return ticks
