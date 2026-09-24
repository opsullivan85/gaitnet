"""The foothold scanners' rays fall on the foothold grid's cells, for every leg and grid
centre. Built from the cfgs alone, without the simulator."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_core.grid import FootholdGrid  # noqa: E402
from gaitnet_sim.env.contract import GaitNetCfg  # noqa: E402
from gaitnet_sim.env.scene import SCANNER_NAMES, GaitNetSceneCfg, foothold_scanner_cfg  # noqa: E402
from gaitnet_sim.robot import HIP_NAMES  # noqa: E402


def ray_cells(scanner, grid: FootholdGrid) -> torch.Tensor:
    """(X, Y, 2) where the scanner's rays start over the inner grid, in its hip's frame."""
    pattern = scanner.pattern_cfg
    starts, _ = pattern.func(pattern, "cpu")
    starts = starts[:, :2] + torch.tensor(scanner.offset.pos[:2])
    b = grid.border
    return starts.reshape(*grid.patch_size, 2)[b : b + grid.size[0], b : b + grid.size[1]]


@pytest.mark.parametrize("center", [(0.0, 0.0), (0.02, 0.08)])
def test_scanner_rays_fall_on_the_grid_cells(center):
    grid = FootholdGrid(center=center)
    for leg, hip in enumerate(HIP_NAMES):
        scanner = foothold_scanner_cfg(hip, grid, leg)
        assert torch.allclose(ray_cells(scanner, grid), grid.cell_centers()[leg], atol=1e-5)


def test_default_scene_scans_the_contract_grid():
    grid = GaitNetCfg().foothold_grid()
    scene = GaitNetSceneCfg(num_envs=1, env_spacing=1.0)
    for leg, name in enumerate(SCANNER_NAMES):
        assert torch.allclose(ray_cells(getattr(scene, name), grid), grid.cell_centers()[leg], atol=1e-5)
