"""Foothold candidates: the finite set of places the policy scores on one planning tick.

A candidate is a point (x, y, z) in its leg's hip yaw frame. Every leg has the same
number of slots K. Slots a sampler couldn't fill are invalid. The action is an index
into the flattened, leg-major candidate list, with the no-op at index L * K.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Candidates:
    xyz: torch.Tensor
    """(N, L, K, 3) candidate positions in each leg's hip yaw frame (m)."""
    valid: torch.Tensor
    """(N, L, K) bool, False for slots the sampler couldn't fill."""
    log_q: torch.Tensor
    """(N, L, K) log of the sampler's proposal density relative to uniform over the leg's
    valid footholds. 0 for uniform and exhaustive samplers."""

    PACKED_DIM = 5
    """Channels of the packed form: x, y, z, valid, log_q."""

    @property
    def num_robots(self) -> int:
        return self.xyz.shape[0]

    @property
    def num_legs(self) -> int:
        return self.xyz.shape[1]

    @property
    def per_leg(self) -> int:
        return self.xyz.shape[2]

    @property
    def noop_index(self) -> int:
        return self.num_legs * self.per_leg

    def num_valid(self) -> torch.Tensor:
        """(N, L) valid candidates per leg."""
        return self.valid.sum(dim=-1)

    def pack(self) -> torch.Tensor:
        """(N, L, K, 5) single tensor form, e.g. for an observation group."""
        return torch.cat(
            [self.xyz, self.valid.unsqueeze(-1).to(self.xyz.dtype), self.log_q.unsqueeze(-1)],
            dim=-1,
        )

    @classmethod
    def unpack(cls, packed: torch.Tensor) -> "Candidates":
        return cls(xyz=packed[..., :3], valid=packed[..., 3] > 0.5, log_q=packed[..., 4])

    def gather(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve flat action indices.

        Args:
            index: (N,) flat index, leg * K + slot, or `noop_index`. Anything outside
                [0, noop_index) is treated as the no-op.

        Returns:
            is_step: (N,) bool, False for the no-op
            leg: (N,) long, 0 for the no-op
            xyz: (N, 3), zeros for the no-op
        """
        index = index.long()
        is_step = (index >= 0) & (index < self.noop_index)
        safe = torch.where(is_step, index, torch.zeros_like(index))
        leg = safe // self.per_leg
        slot = safe % self.per_leg
        rows = torch.arange(index.shape[0], device=index.device)
        xyz = self.xyz[rows, leg, slot]
        xyz = torch.where(is_step.unsqueeze(-1), xyz, torch.zeros_like(xyz))
        return is_step, leg, xyz

    def __getitem__(self, index) -> "Candidates":
        """Select a subset of robots."""
        return Candidates(self.xyz[index], self.valid[index], self.log_q[index])
