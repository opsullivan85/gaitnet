"""GaitNet on a real robot over ROS 1: the planner runs off-board and talks to the robot
through rosbridge, so it needs no ROS install of its own. See the package README for the
contract the robot side implements."""

from gaitnet_ros1.robot import Ros1Robot, StaleObservation

__all__ = ["Ros1Robot", "StaleObservation"]
