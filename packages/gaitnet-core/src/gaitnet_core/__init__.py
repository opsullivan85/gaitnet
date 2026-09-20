"""GaitNet core: the planner and its contract with simulators and robots.

Depends only on torch and numpy, so the same code runs in training, in simulated
evaluation, and on the robot.

`gaitnet_core.control` is the other half: a batched low-level controller that turns this
planner's footsteps into joint torques for a whole batch of robots at once. It is
imported on its own rather than re-exported here, since the planner never needs it.
"""

from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.interfaces import FootstepCommand, LowLevelController, Nudge, RobotInterface
from gaitnet_core.planner import FootholdRules, FootstepPlanner, PlanResult
from gaitnet_core.robot_spec import GO1, RobotSpec
from gaitnet_core.state import Observation, RobotState, TerrainPatch

__all__ = [
    "Candidates",
    "FootholdGrid",
    "FootholdRules",
    "FootstepCommand",
    "FootstepPlanner",
    "GO1",
    "LowLevelController",
    "Nudge",
    "Observation",
    "PlanResult",
    "RobotInterface",
    "RobotSpec",
    "RobotState",
    "TerrainPatch",
]
