"""A real robot behind ROS 1, as `gaitnet_core.interfaces.RobotInterface`.

The robot side publishes gaitnet_msgs/Observation and executes gaitnet_msgs/PlannerCommand
with its own controller (see the package README). One robot, so everything is N = 1.
"""

from __future__ import annotations

import threading
from typing import Sequence

import torch

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.robot_spec import RobotSpec
from gaitnet_core.state import Observation
from gaitnet_ros1.messages import COMMAND_TYPE, OBSERVATION_TYPE, command_to_msg, observation_from_msg
from gaitnet_ros1.transport import Transport


class StaleObservation(TimeoutError):
    """No new observation arrived in time: the robot, its bridge or the network stalled. The
    planner stops rather than plan on old data; the robot side has to cope with commands
    ceasing (see the README)."""


class Ros1Robot:
    def __init__(
        self,
        transport: Transport,
        spec: RobotSpec,
        grid: FootholdGrid,
        observation_topic: str = "/gaitnet/observation",
        command_topic: str = "/gaitnet/command",
        timeout: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            grid: the policy's foothold grid, which the robot's terrain patches must match
            timeout: longest wait for a new observation (s) before `observe` gives up
        """
        self.transport = transport
        self.spec = spec
        self.grid = grid
        self.command_topic = command_topic
        self.timeout = timeout
        self.device = torch.device(device)

        self._arrived = threading.Condition()
        self._latest: dict | None = None
        self._received = 0
        self._returned = 0
        self._stamp: dict | None = None
        self.skipped = 0
        """Observations that arrived but were never planned on (a newer one came first)."""
        transport.subscribe(observation_topic, OBSERVATION_TYPE, self._on_observation)

    @property
    def num_robots(self) -> int:
        return 1

    def _on_observation(self, message: dict) -> None:
        with self._arrived:
            self._latest = message
            self._received += 1
            self._arrived.notify_all()

    def observe(self) -> Observation:
        """The first observation newer than the last one returned, waiting up to `timeout`."""
        with self._arrived:
            if not self._arrived.wait_for(lambda: self._received > self._returned, self.timeout):
                raise StaleObservation(f"no observation for {self.timeout} s")
            message = self._latest
            self.skipped += self._received - self._returned - 1
            self._returned = self._received
        self._stamp = message["header"]["stamp"]
        return observation_from_msg(message, self.grid, self.spec.num_legs, self.device)

    def command(self, footsteps: FootstepCommand | Sequence[FootstepCommand], nudge: Nudge | None = None) -> None:
        """Answer the last observation `observe` returned."""
        if self._stamp is None:
            raise RuntimeError("command before any observation")
        if isinstance(footsteps, FootstepCommand):
            footsteps = [footsteps]
        self.transport.publish(self.command_topic, COMMAND_TYPE, command_to_msg(footsteps, nudge, self._stamp))

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        """Nothing to do: the robot has no episodes to restart."""

    def close(self) -> None:
        self.transport.close()
