"""Which cells of a terrain patch a foot may be placed on."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import RobotSpec


def _window_max(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Max over a (2 radius + 1)^2 window, same size output (edges padded with -inf)."""
    if radius == 0:
        return x
    n, l, h, w = x.shape
    out = F.max_pool2d(
        x.reshape(n * l, 1, h, w), kernel_size=2 * radius + 1, stride=1, padding=radius
    )
    return out.reshape(n, l, h, w)


def valid_footholds(
    heights: torch.Tensor,
    spec: RobotSpec,
    grid: FootholdGrid,
    step_threshold: float = 0.02,
    edge_margin: int = 2,
) -> torch.Tensor:
    """Cells within the leg's reach and at least `edge_margin` cells from any height step.

    Args:
        heights: (N, L, *grid.patch_size) terrain heights relative to each hip (m), -inf
            where unknown.
        step_threshold: A height difference above this between neighbouring cells is an
            edge (m). Both cells of the step count as edge cells.
        edge_margin: Cells within this Chebyshev distance of an edge cell are invalid.
            Needs `grid.border >= edge_margin + 1` to see edges just outside the grid.

    Returns:
        (N, L, *grid.size) bool, True where a foot may be placed.
    """
    return reachable_cells(heights, spec, grid) & away_from_edges(heights, grid, step_threshold, edge_margin)


def _check_patch(heights: torch.Tensor, grid: FootholdGrid) -> None:
    if tuple(heights.shape[-2:]) != grid.patch_size:
        raise ValueError(f"expected terrain patches of {grid.patch_size}, got {tuple(heights.shape[-2:])}")


def reachable_cells(heights: torch.Tensor, spec: RobotSpec, grid: FootholdGrid) -> torch.Tensor:
    """(N, L, *grid.size) bool, cells whose height is within the leg's reach band. Half of
    `valid_footholds`; heights as there."""
    _check_patch(heights, grid)
    lowest, highest = spec.reach_band
    return inner_heights((heights >= lowest) & (heights <= highest), grid)


def away_from_edges(
    heights: torch.Tensor, grid: FootholdGrid, step_threshold: float = 0.02, edge_margin: int = 2
) -> torch.Tensor:
    """(N, L, *grid.size) bool, cells more than `edge_margin` cells from any height step. The
    other half of `valid_footholds`; arguments as there."""
    _check_patch(heights, grid)
    # height range over each cell's 3x3 neighbourhood; unknown (-inf) cells make it inf
    finite = torch.where(torch.isfinite(heights), heights, torch.full_like(heights, -1e6))
    local_max = _window_max(finite, 1)
    local_min = -_window_max(-finite, 1)
    edge = (local_max - local_min) > step_threshold
    near_edge = _window_max(edge.float(), edge_margin) > 0
    return inner_heights(~near_edge, grid)


def inner_heights(heights: torch.Tensor, grid: FootholdGrid) -> torch.Tensor:
    """Crop a (…, *patch_size) terrain patch to the (…, *size) candidate grid."""
    b = grid.border
    return heights[..., b : b + grid.size[0], b : b + grid.size[1]]


UNKNOWN_HEIGHT = -1.0
"""What networks read where the terrain height is unknown (-inf): far below any foothold,
so it reads as a drop."""


def fill_unknown(heights: torch.Tensor, value: float = UNKNOWN_HEIGHT) -> torch.Tensor:
    """`heights` with unknown (non-finite) entries set to `value`, for networks to read."""
    return torch.where(torch.isfinite(heights), heights, torch.full_like(heights, value))


def sample_patch(maps: torch.Tensor, xy: torch.Tensor, grid: FootholdGrid) -> torch.Tensor:
    """Bilinear interpolation of per-leg maps over the grid's cells, differentiable in `xy`.

    Args:
        maps: (N, L, C, X, Y) values at the centres of an X x Y block of cells centred on the
            hip at `grid.resolution`: the terrain patch, the candidate grid, or a size between
        xy: (N, L, K, 2) points in each leg's hip yaw frame (m). Points beyond the outermost
            cell centres take the value at the map's edge.

    Returns:
        (N, L, K, C), in `xy`'s dtype
    """
    n, l, c, size_x, size_y = maps.shape
    if (size_x - grid.size[0]) % 2 or (size_y - grid.size[1]) % 2:
        raise ValueError(f"maps of {(size_x, size_y)} cells can't be centred like the {grid.size} grid")
    k = xy.shape[2]
    # grid_sample's coordinates run from -1 to 1 between the first and last cell centres
    # (align_corners=True), and its first coordinate indexes the last dimension (our y)
    half = torch.tensor([size_x - 1, size_y - 1], device=xy.device, dtype=xy.dtype) * (grid.resolution / 2)
    normalized = (xy / half).flip(-1).reshape(n * l, k, 1, 2)
    sampled = F.grid_sample(
        maps.reshape(n * l, c, size_x, size_y).to(xy.dtype),
        normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(n, l, c, k).transpose(-1, -2)
