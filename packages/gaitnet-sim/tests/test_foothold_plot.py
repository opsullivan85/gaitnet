"""The foothold map figure renders, for both logit kinds, and FootholdPlot writes its files."""

from __future__ import annotations

import pytest
import torch

from gaitnet_core.features import DEFAULT_FEATURES, feature_dim
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.networks import DenseSpatialCNN
from gaitnet_core.planner import FootstepPlanner
from gaitnet_core.robot_spec import GO1
from gaitnet_core.samplers import Dense
from gaitnet_core.state import Observation, RobotState, TerrainPatch
from gaitnet_sim.viz.foothold_plot import FootholdPlot


def _observation(grid: FootholdGrid) -> Observation:
    n, legs = 2, GO1.num_legs
    foot_pos = torch.zeros(n, legs, 3)
    foot_pos[..., 2] = -0.26
    state = RobotState(
        foot_pos=foot_pos,
        foot_vel=torch.zeros(n, legs, 3),
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]).expand(n, 3).clone(),
        contact=torch.ones(n, legs, dtype=torch.bool),
        gait_timing=torch.zeros(n, legs, 3),
        command=torch.zeros(n, 3),
        base_command=torch.zeros(n, 3),
    )
    state.gait_timing[1, 3, 1] = 0.1
    heights = torch.full((n, legs, *grid.patch_size), -0.26)
    heights[:, :, 5:12, 8:20] = float("-inf")
    heights[:, 1] -= 0.3
    return Observation(state, TerrainPatch(heights, grid))


@pytest.mark.parametrize("kind", ["raw", "corrected"])
def test_plot_writes_latest_and_frames(tmp_path, kind):
    torch.manual_seed(0)
    grid = FootholdGrid()
    net = DenseSpatialCNN(feature_dim(DEFAULT_FEATURES, GO1.num_legs), grid.to_dict(), use_bf16=False)
    planner = FootstepPlanner(net, GO1, grid, DEFAULT_FEATURES, Dense())
    obs = _observation(grid)
    plot = FootholdPlot(planner, [1, 0], tmp_path, kind=kind, every=2, keep_frames=True)
    for _ in range(3):
        plot(planner.plan(obs), obs)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["robot0.png", "robot1.png"]
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == ["robot0_000000.png", "robot0_000002.png", "robot1_000000.png", "robot1_000002.png"]
