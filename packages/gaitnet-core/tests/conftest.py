import pytest
import torch

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import GO1
from gaitnet_core.state import Observation, RobotState, TerrainPatch

GROUND = -0.26
"""Flat ground relative to the hips at the nominal height."""


def make_state(n: int, num_legs: int = 4) -> RobotState:
    foot_pos = torch.zeros(n, num_legs, 3)
    foot_pos[..., 2] = GROUND
    return RobotState(
        foot_pos=foot_pos,
        foot_vel=torch.zeros(n, num_legs, 3),
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]).expand(n, 3).clone(),
        contact=torch.ones(n, num_legs, dtype=torch.bool),
        gait_timing=torch.zeros(n, num_legs, 3),
        command=torch.zeros(n, 3),
        base_command=torch.zeros(n, 3),
    )


def make_observation(n: int = 3, grid: FootholdGrid | None = None, height: float = GROUND) -> Observation:
    grid = grid or FootholdGrid()
    heights = torch.full((n, GO1.num_legs, *grid.patch_size), height)
    return Observation(make_state(n), TerrainPatch(heights, grid))


@pytest.fixture(params=[(0.0, 0.0), (0.02, 0.08)], ids=["on_hip", "offset"])
def grid(request) -> FootholdGrid:
    return FootholdGrid(center=request.param)


@pytest.fixture
def observation() -> Observation:
    torch.manual_seed(0)
    return make_observation()
