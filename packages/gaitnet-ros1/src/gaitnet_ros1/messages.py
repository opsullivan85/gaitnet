"""gaitnet_msgs as rosbridge carries them (JSON objects, here dicts) <-> gaitnet_core types.

The message definitions are in ros/gaitnet_msgs/msg; the robot side builds that package.
Frames and units are already the core contract's, so conversion is only reshaping, plus
checks that the robot's terrain patch is the policy's grid.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.state import Observation, RobotState, TerrainPatch

OBSERVATION_TYPE = "gaitnet_msgs/Observation"
COMMAND_TYPE = "gaitnet_msgs/PlannerCommand"
UNKNOWN_HEIGHT = -1000.0
"""TerrainPatch.UNKNOWN: the wire value for a cell with no data."""
_UNKNOWN_BELOW = UNKNOWN_HEIGHT / 2


class ContractError(ValueError):
    """A message the robot sent doesn't fit the contract (sizes, grid)."""


def stamp_now() -> dict:
    now = time.time()
    return {"secs": int(now), "nsecs": int((now % 1) * 1e9)}


def stamp_seconds(stamp: dict) -> float:
    return stamp["secs"] + stamp["nsecs"] * 1e-9


def _tensor(values: Sequence[float], shape: tuple[int, ...], name: str, device, dtype=torch.float32) -> torch.Tensor:
    expected = 1
    for size in shape:
        expected *= size
    if len(values) != expected:
        raise ContractError(f"{name}: expected {expected} values, got {len(values)}")
    return torch.tensor(values, dtype=dtype, device=device).reshape(shape)


def observation_from_msg(
    msg: dict, grid: FootholdGrid, num_legs: int = 4, device: torch.device | str = "cpu"
) -> Observation:
    """A gaitnet_msgs/Observation -> a one-robot Observation on the policy's grid."""
    state, terrain = msg["state"], msg["terrain"]
    size = (int(terrain["size_x"]), int(terrain["size_y"]))
    if size != grid.patch_size or abs(float(terrain["resolution"]) - grid.resolution) > 1e-6:
        raise ContractError(
            f"terrain patch of {size} cells at {terrain['resolution']} m, the policy's is"
            f" {grid.patch_size} at {grid.resolution} m"
        )
    heights = _tensor(terrain["heights"], (1, num_legs, *size), "terrain.heights", device)
    # the wire's UNKNOWN (and anything non-finite that got through) is core's -inf
    heights = torch.where(torch.isfinite(heights) & (heights > _UNKNOWN_BELOW), heights, float("-inf"))

    timing = state["gait_timing"]
    gait_timing = torch.stack(
        [
            _tensor(timing[name], (1, num_legs), f"gait_timing.{name}", device)
            for name in ("swing_phase", "swing_remaining", "time_since_touchdown")
        ],
        dim=-1,
    )
    robot_state = RobotState(
        foot_pos=_tensor(state["foot_pos"], (1, num_legs, 3), "foot_pos", device),
        foot_vel=_tensor(state["foot_vel"], (1, num_legs, 3), "foot_vel", device),
        base_lin_vel=_tensor(state["base_lin_vel"], (1, 3), "base_lin_vel", device),
        base_ang_vel=_tensor(state["base_ang_vel"], (1, 3), "base_ang_vel", device),
        projected_gravity=_tensor(state["projected_gravity"], (1, 3), "projected_gravity", device),
        contact=_tensor(state["contact"], (1, num_legs), "contact", device, dtype=torch.bool),
        gait_timing=gait_timing,
        command=_tensor(state["command"], (1, 3), "command", device),
        base_command=_tensor(state["base_command"], (1, 3), "base_command", device),
    )
    return Observation(robot_state, TerrainPatch(heights=heights, grid=grid))


def _values(t: torch.Tensor) -> list:
    return t.detach().cpu().reshape(-1).tolist()


def observation_to_msg(observation: Observation, stamp: dict | None = None, robot: int = 0) -> dict:
    """One robot of an Observation -> a gaitnet_msgs/Observation, e.g. for a Python robot side
    or a test."""
    state, heights = observation.state, observation.terrain.heights[robot]
    grid = observation.terrain.grid
    heights = torch.where(torch.isfinite(heights), heights, torch.full_like(heights, UNKNOWN_HEIGHT))
    timing = state.gait_timing[robot]
    return {
        "header": {"seq": 0, "stamp": stamp or stamp_now(), "frame_id": ""},
        "state": {
            "foot_pos": _values(state.foot_pos[robot]),
            "foot_vel": _values(state.foot_vel[robot]),
            "base_lin_vel": _values(state.base_lin_vel[robot]),
            "base_ang_vel": _values(state.base_ang_vel[robot]),
            "projected_gravity": _values(state.projected_gravity[robot]),
            "contact": [bool(c) for c in _values(state.contact[robot])],
            "gait_timing": {
                "swing_phase": _values(timing[:, 0]),
                "swing_remaining": _values(timing[:, 1]),
                "time_since_touchdown": _values(timing[:, 2]),
            },
            "command": _values(state.command[robot]),
            "base_command": _values(state.base_command[robot]),
        },
        "terrain": {
            "resolution": grid.resolution,
            "size_x": grid.patch_size[0],
            "size_y": grid.patch_size[1],
            "heights": _values(heights),
        },
    }


def command_to_msg(
    footsteps: Sequence[FootstepCommand],
    nudge: Nudge | None,
    observation_stamp: dict,
    stamp: dict | None = None,
    robot: int = 0,
) -> dict:
    """One robot's footsteps (the active ones) and nudge -> a gaitnet_msgs/PlannerCommand."""
    steps = [
        {
            "leg": int(footstep.leg[robot]),
            "target": _values(footstep.target[robot]),
            "duration": float(footstep.duration[robot]),
        }
        for footstep in footsteps
        if bool(footstep.active[robot])
    ]
    delta = _values(nudge.command_delta[robot]) if nudge is not None else [0.0, 0.0, 0.0]
    return {
        "header": {"seq": 0, "stamp": stamp or stamp_now(), "frame_id": ""},
        "observation_stamp": observation_stamp,
        "footsteps": steps,
        "nudge": {"command_delta": delta},
    }


def command_from_msg(msg: dict, device: torch.device | str = "cpu") -> tuple[list[FootstepCommand], Nudge]:
    """A gaitnet_msgs/PlannerCommand -> one-robot footsteps and nudge."""
    footsteps = [
        FootstepCommand(
            active=torch.ones(1, dtype=torch.bool, device=device),
            leg=torch.tensor([int(step["leg"])], device=device),
            target=torch.tensor([step["target"]], dtype=torch.float32, device=device),
            duration=torch.tensor([float(step["duration"])], device=device),
        )
        for step in msg["footsteps"]
    ]
    nudge = Nudge(command_delta=torch.tensor([msg["nudge"]["command_delta"]], dtype=torch.float32, device=device))
    return footsteps, nudge
