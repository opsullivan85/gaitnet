"""The per-episode gait statistics of the training log, on hand-made ticks with a fake
`RobotIO`. Without the simulator."""

from __future__ import annotations

import pytest
import torch

from gaitnet_core.interfaces import FootstepCommand
from gaitnet_core.robot_spec import GO1
from gaitnet_sim.env.metrics import GaitMetrics, RatioMeter

DT = 0.04
FL, RR = 0, 3


class FakeIO:
    """Two robots: feet at their hips, robot 0 on four feet and robot 1 on two."""

    def foot_pos_hip(self) -> torch.Tensor:
        return torch.zeros(2, 4, 3)

    def foot_contact(self) -> torch.Tensor:
        return torch.tensor([[True, True, True, True], [True, False, False, True]])


def _footsteps(leg: int, target_xy: tuple[float, float], duration: float) -> FootstepCommand:
    """Robot 0 steps, robot 1 doesn't."""
    return FootstepCommand(
        active=torch.tensor([True, False]),
        leg=torch.tensor([leg, -1]),
        target=torch.tensor([[*target_xy, -0.3], [0.0, 0.0, 0.0]]),
        duration=torch.tensor([duration, 0.0]),
    )


def _nudge(robot_1_xy: tuple[float, float]) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 0.0], [*robot_1_xy, 0.0]])


@pytest.fixture
def metrics() -> GaitMetrics:
    low, high = GO1.swing_duration_range
    metrics = GaitMetrics(FakeIO(), GO1, DT)
    metrics.record(_footsteps(FL, (0.1, 0.0), 0.5 * (low + high)), _nudge((0.3, 0.4)))
    metrics.record(_footsteps(RR, (0.0, 0.2), high), _nudge((0.0, 0.0)))
    return metrics


def test_the_stepping_robot(metrics):
    low, high = GO1.swing_duration_range
    log = {key: value.item() for key, value in metrics.pop(torch.tensor([0])).items()}
    assert log == pytest.approx(
        {
            "Gait/steps_per_s": 1 / DT,
            "Gait/leg_share_FL": 0.5,
            "Gait/leg_share_FR": 0.0,
            "Gait/leg_share_RL": 0.0,
            "Gait/leg_share_RR": 0.5,
            "Gait/swing_duration": 0.5 * (0.5 * (low + high) + high),
            "Gait/duration_at_limit": 0.5,
            "Gait/step_length": 0.15,
            "Gait/low_support_frac": 0.0,
            "Observer/nudged_frac": 0.0,
            "Observer/nudge_xy": 0.0,
        }
    )


def test_the_nudged_robot_on_two_feet(metrics):
    log = {key: value.item() for key, value in metrics.pop(torch.tensor([1])).items()}
    # no steps: every per-step ratio reads 0 rather than nan
    assert log["Gait/steps_per_s"] == 0.0
    assert log["Gait/swing_duration"] == 0.0
    assert log["Gait/low_support_frac"] == 1.0
    assert log["Observer/nudged_frac"] == 0.5
    assert log["Observer/nudge_xy"] == pytest.approx(0.5)


def test_pop_pools_the_ended_envs_and_restarts_only_them():
    meter = RatioMeter()
    assert meter.pop(torch.tensor([0])) == {}
    meter.add({"r": (torch.tensor([1.0, 3.0, 5.0]), torch.tensor([1.0, 1.0, 1.0]))})
    meter.add({"r": (torch.tensor([1.0, 3.0, 5.0]), torch.tensor([1.0, 1.0, 1.0]))})
    # (2 + 6) / (2 + 2), not the mean of per-env ratios
    assert meter.pop(torch.tensor([0, 1]))["r"].item() == pytest.approx(2.0)
    assert meter.pop(torch.tensor([0, 1]))["r"].item() == 0.0
    assert meter.pop(torch.tensor([2]))["r"].item() == pytest.approx(5.0)


def test_recorded_in_inference_mode_popped_outside():
    """RSL-RL steps the env under inference mode; play and evaluation reset outside it."""
    meter = RatioMeter()
    with torch.inference_mode():
        meter.add({"r": (torch.ones(2), torch.ones(2))})
    assert meter.pop(torch.tensor([0]))["r"].item() == 1.0
