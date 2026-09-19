"""Messages and Ros1Robot without ROS: conversions, the contract checks, and the deployment
loop over an in-memory transport."""

from __future__ import annotations

import json
import threading
import time

import pytest
import torch

from gaitnet_core.features import DEFAULT_FEATURES, feature_dim
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.networks import CandidateScorer
from gaitnet_core.planner import FootstepPlanner
from gaitnet_core.robot_spec import GO1
from gaitnet_core.runtime import PlannerRuntime
from gaitnet_core.samplers import Dense
from gaitnet_core.state import Observation, RobotState, TerrainPatch
from gaitnet_ros1.messages import (
    UNKNOWN_HEIGHT,
    ContractError,
    command_from_msg,
    command_to_msg,
    observation_from_msg,
    observation_to_msg,
    stamp_now,
)
from gaitnet_ros1.robot import Ros1Robot, StaleObservation

GRID = FootholdGrid(resolution=0.015, size=(9, 9), border=2)


def observation() -> Observation:
    torch.manual_seed(0)
    foot_pos = torch.randn(1, 4, 3) * 0.1
    foot_pos[..., 2] = -0.26
    state = RobotState(
        foot_pos=foot_pos,
        foot_vel=torch.randn(1, 4, 3),
        base_lin_vel=torch.randn(1, 3),
        base_ang_vel=torch.randn(1, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        contact=torch.tensor([[True, False, True, True]]),
        gait_timing=torch.tensor([[[0.0, 0.0, 0.3], [0.5, 0.1, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]]),
        command=torch.tensor([[0.1, 0.0, 0.0]]),
        base_command=torch.tensor([[0.2, 0.0, 0.0]]),
    )
    heights = -0.26 + 0.01 * torch.randn(1, 4, *GRID.patch_size)
    heights[0, 2, :2] = float("-inf")
    return Observation(state, TerrainPatch(heights, GRID))


def test_observation_round_trip_through_json():
    original = observation()
    message = json.loads(json.dumps(observation_to_msg(original), allow_nan=False))
    assert min(message["terrain"]["heights"]) == UNKNOWN_HEIGHT
    back = observation_from_msg(message, GRID)
    for name in ("foot_pos", "foot_vel", "base_lin_vel", "base_ang_vel", "projected_gravity", "gait_timing", "command", "base_command"):
        assert torch.allclose(getattr(back.state, name), getattr(original.state, name)), name
    assert torch.equal(back.state.contact, original.state.contact)
    assert torch.equal(torch.isinf(back.terrain.heights), torch.isinf(original.terrain.heights))
    known = torch.isfinite(original.terrain.heights)
    assert torch.allclose(back.terrain.heights[known], original.terrain.heights[known])


def test_contract_violations_are_refused():
    message = observation_to_msg(observation())
    with pytest.raises(ContractError, match="terrain patch"):
        observation_from_msg(message, FootholdGrid(resolution=0.015, size=(11, 11), border=2))
    with pytest.raises(ContractError, match="terrain patch"):
        observation_from_msg(message, FootholdGrid(resolution=0.02, size=(9, 9), border=2))
    message["state"]["foot_pos"] = message["state"]["foot_pos"][:9]
    with pytest.raises(ContractError, match="foot_pos"):
        observation_from_msg(message, GRID)


def test_command_round_trip():
    footsteps = [
        FootstepCommand(torch.tensor([True]), torch.tensor([2]), torch.tensor([[0.05, -0.02, -0.25]]), torch.tensor([0.2])),
        FootstepCommand.none(1),
    ]
    stamp = {"secs": 5, "nsecs": 7}
    message = json.loads(json.dumps(command_to_msg(footsteps, Nudge(torch.tensor([[-0.1, 0.0, 0.0]])), stamp)))
    assert message["observation_stamp"] == stamp and len(message["footsteps"]) == 1
    steps, nudge = command_from_msg(message)
    assert int(steps[0].leg) == 2 and torch.allclose(steps[0].target, footsteps[0].target)
    assert torch.allclose(nudge.command_delta, torch.tensor([[-0.1, 0.0, 0.0]]))


class LoopbackTransport:
    def __init__(self):
        self.callbacks: dict[str, object] = {}
        self.published: list[tuple[str, dict]] = []

    def subscribe(self, topic, message_type, callback):
        self.callbacks[topic] = callback

    def publish(self, topic, message_type, message):
        self.published.append((topic, message))

    def close(self):
        pass


def planner() -> FootstepPlanner:
    torch.manual_seed(0)
    network = CandidateScorer(
        feature_dim(DEFAULT_FEATURES, 4), shared_sizes=[16], candidate_sizes=[8], trunk_sizes=[16], use_bf16=False
    )
    return FootstepPlanner(network, GO1, GRID, DEFAULT_FEATURES, Dense())


def test_runtime_answers_each_observation_the_robot_sends():
    transport = LoopbackTransport()
    robot = Ros1Robot(transport, GO1, GRID, timeout=1.0)
    runtime = PlannerRuntime(robot, planner(), rate_hz=None)
    stamps = []

    def robot_side():
        # publish, then wait for the answer before the next
        for i in range(5):
            stamps.append(stamp_now())
            transport.callbacks["/gaitnet/observation"](observation_to_msg(observation(), stamps[-1]))
            deadline = time.monotonic() + 2.0
            while len(transport.published) <= i and time.monotonic() < deadline:
                time.sleep(0.001)

    publisher = threading.Thread(target=robot_side)
    publisher.start()
    with torch.no_grad():
        ticks = runtime.run(max_ticks=5)
    publisher.join()
    assert ticks == 5 and len(transport.published) == 5
    answered = [message["observation_stamp"] for _, message in transport.published]
    assert answered == stamps and robot.skipped == 0
    for topic, message in transport.published:
        assert topic == "/gaitnet/command"
        for step in message["footsteps"]:
            assert 0 <= step["leg"] < 4 and 0.1 <= step["duration"] <= 0.3

    # the robot goes quiet: the planner stops instead of re-planning on old data
    with pytest.raises(StaleObservation):
        robot.timeout = 0.05
        robot.observe()
