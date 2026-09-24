"""The per-leg foothold grid: a rectangle of cells near each hip, the same for every leg up
to a mirror in y.

Each leg's grid is centred at `center` from its hip, in the hip's gravity-aligned yaw frame,
with y mirrored on the right legs so that positive `center[1]` is outboard on every leg. Cell
(i, j) of a leg has its centre at `leg_centers()[leg] + (-half_x + i * resolution, -half_y +
j * resolution)`, so the first index runs along x (forward) and the second along y (left) on
both sides. A patch tensor is (..., num_legs, size_x, size_y) in that order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from gaitnet_core.robot_spec import LEG_NAMES


@dataclass(frozen=True)
class FootholdGrid:
    resolution: float = 0.015
    """Cell size (m)."""
    size: tuple[int, int] = (25, 25)
    """Number of cells along (x, y). Odd sizes put a cell centre on `center`."""
    border: int = 3
    """Extra cells of terrain on every side of the grid. Terrain patches are
    `patch_size`, so rules that look at neighbouring cells (edge margins) see real
    terrain at the grid's edge. Candidates only ever come from the inner grid."""
    center: tuple[float, float] = (0.0, 0.0)
    """(x, y) of a left leg's grid centre from its hip, in the hip's yaw frame (m); right
    legs mirror y. The default, on the hip, is what bundles from before the field had."""

    @property
    def patch_size(self) -> tuple[int, int]:
        return (self.size[0] + 2 * self.border, self.size[1] + 2 * self.border)

    @property
    def num_cells(self) -> int:
        return self.size[0] * self.size[1]

    @property
    def half_extent(self) -> tuple[float, float]:
        """Distance from the grid's centre to the outermost cell centres along (x, y)."""
        return (
            (self.size[0] - 1) * self.resolution / 2,
            (self.size[1] - 1) * self.resolution / 2,
        )

    def leg_centers(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """(L, 2) each leg's grid centre from its hip, in the hip's yaw frame (m)."""
        side = [1.0 if name.endswith("L") else -1.0 for name in LEG_NAMES]
        centers = [(self.center[0], s * self.center[1]) for s in side]
        return torch.tensor(centers, device=device, dtype=dtype or torch.get_default_dtype())

    def cell_centers(self, device: torch.device | str | None = None) -> torch.Tensor:
        """(L, size_x, size_y, 2) each leg's cell centre (x, y) coordinates."""
        half_x, half_y = self.half_extent
        x = torch.linspace(-half_x, half_x, self.size[0], device=device)
        y = torch.linspace(-half_y, half_y, self.size[1], device=device)
        local = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
        return local + self.leg_centers(device, local.dtype)[:, None, None]

    def cell_to_xy(self, cell: torch.Tensor, leg: torch.Tensor | int) -> torch.Tensor:
        """(..., 2) integer (i, j) cell indices to (..., 2) cell centre (x, y).

        Args:
            leg: which leg each cell is on, broadcastable to `cell.shape[:-1]`
        """
        half = torch.tensor(self.half_extent, device=cell.device)
        return cell.to(half.dtype) * self.resolution - half + self._center_of(leg, cell.device, half.dtype)

    def xy_to_cell(self, xy: torch.Tensor, leg: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., 2) (x, y) to the (..., 2) index of the cell containing it.

        Args:
            leg: which leg each point is on, broadcastable to `xy.shape[:-1]`

        Returns:
            cell: clamped to the grid
            in_bounds: (...) whether the point lies inside the grid's outer cell edges
        """
        half = torch.tensor(self.half_extent, device=xy.device, dtype=xy.dtype)
        cell_float = (xy - self._center_of(leg, xy.device, xy.dtype) + half) / self.resolution
        cell = torch.round(cell_float).long()
        upper = torch.tensor(self.size, device=xy.device) - 1
        in_bounds = ((cell >= 0) & (cell <= upper)).all(dim=-1)
        return torch.minimum(cell.clamp(min=0), upper), in_bounds

    def _center_of(self, leg: torch.Tensor | int, device, dtype) -> torch.Tensor:
        """(..., 2) the grid centre of each of `leg`'s entries."""
        leg = torch.as_tensor(leg, device=device)
        return self.leg_centers(device, dtype)[leg]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FootholdGrid":
        return cls(
            resolution=float(data["resolution"]),
            size=tuple(int(n) for n in data["size"]),
            border=int(data["border"]),
            center=tuple(float(c) for c in data.get("center", (0.0, 0.0))),
        )
