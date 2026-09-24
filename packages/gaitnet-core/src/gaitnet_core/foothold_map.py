"""What the planner saw on one tick, for a few watched robots, laid out on the foothold grid:
the network's score at every cell, the masks that decided which cells it could take, and
what it chose. For looking at, not for planning; `gaitnet_sim.viz` draws it, fed through
`PlannerRuntime(on_plan=...)`.

The scores come from a dense pass of their own over every cell of the watched robots, with
the masks ignored. So the map also shows what the network thinks of cells the rules forbid,
and it looks the same whichever sampler the planner used. The leg and no-op probabilities
and the choice are the plan's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from gaitnet_core.eligibility import step_eligible
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.planner import FootstepPlanner, PlanResult
from gaitnet_core.samplers import Dense
from gaitnet_core.state import Observation
from gaitnet_core.terrain import away_from_edges, fill_unknown, inner_heights, reachable_cells

LogitKind = Literal["raw", "corrected"]
LOGIT_KINDS: tuple[str, ...] = ("raw", "corrected")
"""`raw` is f(s, l, x), what the deterministic policy's argmax within a leg compares.
`corrected` is f - log N_valid, the logit the stochastic policy samples from, on the same
scale as the no-op logit (see `gaitnet_core.selection`)."""


@dataclass
class FootholdMap:
    grid: FootholdGrid
    robot_ids: torch.Tensor
    """(R,) long, which robots of the batch these are."""
    raw: torch.Tensor
    """(R, L, *grid.size) network score f(s, l, x) at every cell centre, masked or not."""
    duration: torch.Tensor
    """(R, L, *grid.size) swing duration the network gives each cell centre (s)."""
    heights: torch.Tensor
    """(R, L, *grid.size) terrain height at each cell centre relative to the leg's hip (m),
    -inf where unknown."""
    reachable: torch.Tensor
    """(R, L, *grid.size) bool, the height is within the leg's reach band."""
    away_from_edges: torch.Tensor
    """(R, L, *grid.size) bool, far enough from any height step."""
    eligible: torch.Tensor
    """(R, L) bool, the leg may lift off this tick."""
    leg_logit: torch.Tensor
    """(R, L) the plan's log unnormalized step probability of each leg, -inf for legs that
    can't step. Comparable to `noop_logit`."""
    noop_logit: torch.Tensor
    """(R,) the plan's no-op logit."""
    leg: torch.Tensor
    """(R,) long, the leg the plan steps, -1 for the no-op."""
    target: torch.Tensor
    """(R, 3) the chosen foothold in that leg's hip yaw frame (m), after any refinement;
    zeros for the no-op."""
    step_duration: torch.Tensor
    """(R,) the chosen swing duration (s), 0 for the no-op."""

    @property
    def terrain_ok(self) -> torch.Tensor:
        """(R, L, *grid.size) bool, the terrain allows the cell: reachable and away from edges."""
        return self.reachable & self.away_from_edges

    @property
    def allowed(self) -> torch.Tensor:
        """(R, L, *grid.size) bool, cells the planner may step to this tick."""
        return self.terrain_ok & self.eligible[..., None, None]

    def logits(self, kind: LogitKind = "raw") -> torch.Tensor:
        """(R, L, *grid.size) per-cell logits of `kind` (see `LOGIT_KINDS`), at every cell.

        The correction is the dense sampler's: log q is 0, and N_valid counts the cells the
        terrain allows the leg, whether or not it may step this tick, so an ineligible leg
        shows what its logits would be if it could."""
        if kind == "raw":
            return self.raw
        if kind == "corrected":
            num_valid = self.terrain_ok.flatten(2).sum(-1).clamp(min=1).to(self.raw.dtype)
            return self.raw - torch.log(num_valid)[..., None, None]
        raise ValueError(f"unknown logit kind {kind!r}, expected one of {LOGIT_KINDS}")

    def step_probabilities(self) -> torch.Tensor:
        """(R, L + 1) the plan's probability of stepping each leg, and of the no-op last."""
        return torch.softmax(torch.cat([self.leg_logit, self.noop_logit.unsqueeze(-1)], dim=-1), dim=-1)

    def best_cells(self) -> torch.Tensor:
        """(R, L, 2) long, each leg's highest scoring terrain-allowed cell, the one the
        deterministic policy would take if it stepped that leg. Arbitrary for legs with none."""
        masked = self.raw.masked_fill(~self.terrain_ok, float("-inf")).flatten(2)
        flat = masked.argmax(dim=-1)
        return torch.stack(torch.unravel_index(flat, self.grid.size), dim=-1)


@torch.no_grad()
def foothold_map(
    planner: FootstepPlanner,
    plan: PlanResult,
    observation: Observation,
    robot_ids: Sequence[int] | torch.Tensor,
) -> FootholdMap:
    """The foothold map of `robot_ids`, from the plan `planner` made from `observation`."""
    ids = torch.as_tensor(robot_ids, dtype=torch.long, device=observation.state.device)
    observation = observation[ids]
    grid, rules = planner.grid, planner.rules
    patch = observation.terrain.heights
    heights = inner_heights(patch, grid)

    every_cell = torch.ones(heights.shape, dtype=torch.bool, device=heights.device)
    candidates = Dense().sample(every_cell, grid, heights=fill_unknown(heights))
    scores = planner.score(observation, candidates)
    shape = heights.shape

    is_step = plan.is_step[ids]
    return FootholdMap(
        grid=grid,
        robot_ids=ids,
        raw=scores.step_logits.reshape(shape),
        duration=scores.duration.reshape(shape),
        heights=heights,
        reachable=reachable_cells(patch, planner.spec, grid),
        away_from_edges=away_from_edges(patch, grid, rules.step_threshold, rules.edge_margin),
        eligible=step_eligible(observation.state.gait_timing, rules.min_stance_after_step),
        leg_logit=plan.leg_marginals[ids],
        noop_logit=plan.scores.noop_logit[ids],
        leg=torch.where(is_step, plan.leg[ids], torch.full_like(plan.leg[ids], -1)),
        target=plan.target[ids],
        step_duration=plan.selection.duration[ids],
    )
