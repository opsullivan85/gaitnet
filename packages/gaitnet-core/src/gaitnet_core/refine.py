"""Continuous refinement of a chosen footstep by gradient ascent on the network's score.

The planner picks a leg and a foothold from a finite candidate set. Since the network
scores any point, the foothold can then be moved off the candidate set, uphill in f,
while staying on valid footholds. This replaces the old per-leg projected gradient
ascent (gaitnet/util/pga.py): it refines only the leg already chosen, starts from the
best candidates rather than random points, and checks validity with a cell lookup on the
GPU instead of a per-robot scipy distance transform.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.networks.candidate_scorer import CandidateScorer
from gaitnet_core.planner import PlanResult


def _lookup(cells_valid: torch.Tensor, heights: torch.Tensor | None, grid: FootholdGrid, rows, leg, xy):
    """Validity and height of the cell under each point. xy: (N, S, 2)"""
    cell, in_bounds = grid.xy_to_cell(xy)
    r = rows.unsqueeze(-1)
    l = leg.unsqueeze(-1)
    ok = cells_valid[r, l, cell[..., 0], cell[..., 1]] & in_bounds
    z = heights[r, l, cell[..., 0], cell[..., 1]] if heights is not None else torch.zeros_like(xy[..., 0])
    return ok, z


def refine_plan(
    network: CandidateScorer,
    state: torch.Tensor,
    plan: PlanResult,
    valid: torch.Tensor,
    grid: FootholdGrid,
    heights: torch.Tensor | None = None,
    starts: int = 4,
    steps: int = 4,
    step_length: float | None = None,
) -> PlanResult:
    """Move each stepping robot's foothold uphill in the network's score.

    Args:
        state: (N, state_dim) the state vector the plan was scored with
        valid: (N, L, *grid.size) valid footholds the plan was sampled from
        heights: (N, L, *grid.size) cell heights for the refined foothold's z
        starts: best candidates of the chosen leg to start ascent from
        steps: ascent steps, each of `step_length` along the normalized gradient
        step_length: (m), default a quarter cell. A step onto an invalid cell is rejected.

    Returns:
        The plan with `target` and `selection.duration` replaced for stepping robots.
        `selection.index` still names the starting candidate.
    """
    step_length = grid.resolution / 4 if step_length is None else step_length
    candidates = plan.candidates
    n, l = candidates.num_robots, candidates.num_legs
    device = state.device
    rows = torch.arange(n, device=device)
    leg = plan.leg

    # start from the chosen leg's best candidates
    leg_logits = plan.scores.step_logits[rows, leg].masked_fill(~candidates.valid[rows, leg], float("-inf"))
    starts = min(starts, candidates.per_leg)
    top_logits, top_slots = torch.topk(leg_logits, starts, dim=-1)  # (N, S)
    xy = candidates.xyz[rows.unsqueeze(-1), leg.unsqueeze(-1), top_slots][..., :2].clone()
    alive = torch.isfinite(top_logits) & plan.is_step.unsqueeze(-1)

    def score(points: torch.Tensor):
        ok, z = _lookup(valid, heights, grid, rows, leg, points)
        xyz = torch.zeros(n, l, starts, 3, device=device, dtype=points.dtype)
        mask = torch.zeros(n, l, starts, dtype=torch.bool, device=device)
        xyz[rows, leg] = torch.cat([points, z.unsqueeze(-1)], dim=-1)
        mask[rows, leg] = True
        cands = Candidates(xyz=xyz, valid=mask, log_q=torch.zeros_like(xyz[..., 0]))
        scores = network(state, cands)
        return scores.step_logits[rows, leg], scores.duration[rows, leg], ok, z

    with torch.enable_grad():
        best_f, best_duration, _, best_z = score(xy)
        best_f = best_f.detach().masked_fill(~alive, float("-inf"))
        best_duration, best_z = best_duration.detach(), best_z.detach()
        best_xy = xy.clone()
        for _ in range(steps):
            points = xy.detach().requires_grad_(True)
            f, _, _, _ = score(points)
            (grad,) = torch.autograd.grad(f.sum(), points)
            direction = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            proposal = (points + step_length * direction).detach()
            f_new, duration_new, ok, z_new = score(proposal)
            f_new, duration_new = f_new.detach(), duration_new.detach()
            accept = ok & alive
            xy = torch.where(accept.unsqueeze(-1), proposal, points.detach())
            better = accept & (f_new > best_f)
            best_f = torch.where(better, f_new, best_f)
            best_duration = torch.where(better, duration_new, best_duration)
            best_z = torch.where(better, z_new, best_z)
            best_xy = torch.where(better.unsqueeze(-1), proposal, best_xy)

    pick = torch.argmax(best_f, dim=-1)
    new_target = torch.cat([best_xy[rows, pick], best_z[rows, pick].unsqueeze(-1)], dim=-1)
    target = torch.where(plan.is_step.unsqueeze(-1), new_target, plan.target)
    duration = torch.where(plan.is_step, best_duration[rows, pick], plan.selection.duration)
    return replace(plan, target=target, selection=replace(plan.selection, duration=duration))
