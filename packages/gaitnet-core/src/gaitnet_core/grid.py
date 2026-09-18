"""The per-leg foothold grid: a rectangle of cells centred on each hip.

Cell (i, j) has its centre at (x, y) = (-half_x + i * resolution, -half_y + j * resolution)
in the hip's gravity-aligned yaw frame, so the first index runs along x (forward). A
patch tensor is (..., num_legs, size_x, size_y) in that order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class FootholdGrid:
    resolution: float = 0.015
    """Cell size (m)."""
    size: tuple[int, int] = (25, 25)
    """Number of cells along (x, y). Odd sizes put a cell centre on the hip."""
    border: int = 3
    """Extra cells of terrain on every side of the grid. Terrain patches are
    `patch_size`, so rules that look at neighbouring cells (edge margins) see real
    terrain at the grid's edge. Candidates only ever come from the inner grid."""

    @property
    def patch_size(self) -> tuple[int, int]:
        return (self.size[0] + 2 * self.border, self.size[1] + 2 * self.border)

    @property
    def num_cells(self) -> int:
        return self.size[0] * self.size[1]

    @property
    def half_extent(self) -> tuple[float, float]:
        """Distance from the hip to the outermost cell centres along (x, y)."""
        return (
            (self.size[0] - 1) * self.resolution / 2,
            (self.size[1] - 1) * self.resolution / 2,
        )

    def cell_centers(self, device: torch.device | str | None = None) -> torch.Tensor:
        """(size_x, size_y, 2) cell centre (x, y) coordinates."""
        half_x, half_y = self.half_extent
        x = torch.linspace(-half_x, half_x, self.size[0], device=device)
        y = torch.linspace(-half_y, half_y, self.size[1], device=device)
        return torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)

    def cell_to_xy(self, cell: torch.Tensor) -> torch.Tensor:
        """(..., 2) integer (i, j) cell indices to (..., 2) cell centre (x, y)."""
        half = torch.tensor(self.half_extent, device=cell.device)
        return cell.to(half.dtype) * self.resolution - half

    def xy_to_cell(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., 2) (x, y) to the (..., 2) index of the cell containing it.

        Returns:
            cell: clamped to the grid
            in_bounds: (...) whether the point lies inside the grid's outer cell edges
        """
        half = torch.tensor(self.half_extent, device=xy.device, dtype=xy.dtype)
        cell_float = (xy + half) / self.resolution
        cell = torch.round(cell_float).long()
        upper = torch.tensor(self.size, device=xy.device) - 1
        in_bounds = ((cell >= 0) & (cell <= upper)).all(dim=-1)
        return torch.minimum(cell.clamp(min=0), upper), in_bounds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FootholdGrid":
        return cls(
            resolution=float(data["resolution"]),
            size=tuple(int(n) for n in data["size"]),
            border=int(data["border"]),
        )
