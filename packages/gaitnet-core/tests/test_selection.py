import math

import torch

from gaitnet_core.candidates import Candidates
from gaitnet_core.samplers import Dense, UniformJitter, UniformLattice
from gaitnet_core.selection import (
    FootstepDistribution,
    Scores,
    Selection,
    action_logits,
    leg_marginals,
    select_deterministic,
)


def smooth_scores(cands: Candidates, noop: float, peak: float = 2.0, width: float = 0.06, leg_bonus=(0.0, 0.3, 0.0, 0.0)):
    """A smooth score field peaking at (0.05, 0.02) in every leg's frame."""
    x, y = cands.xyz[..., 0], cands.xyz[..., 1]
    f = peak - ((x - 0.05) ** 2 + (y - 0.02) ** 2) / (2 * width**2)
    f = f + torch.tensor(leg_bonus).view(1, -1, 1)
    n = cands.num_robots
    return Scores(step_logits=f, noop_logit=torch.full((n,), noop), duration=torch.full_like(f, 0.2))


def step_probability(scores, cands):
    return 1 - torch.softmax(action_logits(scores, cands), dim=-1)[:, -1]


def test_flat_field_prefers_stepping(grid):
    """Every foothold scores 2 above the no-op, so the policy steps with probability
    4e^2 / (4e^2 + 1) and the deterministic policy must step. A plain argmax over candidate
    logits compares one foothold, minus log(625), against the no-op and holds."""
    valid = torch.ones(1, 4, *grid.size, dtype=torch.bool)
    cands = Dense().sample(valid, grid)
    scores = Scores(
        step_logits=torch.full((1, 4, grid.num_cells), 2.0),
        noop_logit=torch.zeros(1),
        duration=torch.full((1, 4, grid.num_cells), 0.2),
    )
    assert torch.allclose(leg_marginals(scores, cands), torch.full((1, 4), 2.0))
    expected = 4 * math.exp(2) / (4 * math.exp(2) + 1)
    assert torch.allclose(step_probability(scores, cands), torch.tensor([expected]))
    assert select_deterministic(scores, cands).index.item() != cands.noop_index
    # the old rule
    assert torch.argmax(action_logits(scores, cands), dim=-1).item() == cands.noop_index


def test_step_probability_is_sampler_invariant(grid):
    valid = torch.ones(1, 4, *grid.size, dtype=torch.bool)
    valid[..., :, :6] = False  # some terrain masked out
    dense = Dense().sample(valid, grid)
    p_dense = step_probability(smooth_scores(dense, noop=1.0), dense).item()
    generator = torch.Generator().manual_seed(0)
    for sampler in (UniformLattice(64), UniformJitter(64), UniformJitter(16)):
        ps = []
        for _ in range(200):
            cands = sampler.sample(valid, grid, generator=generator)
            ps.append(step_probability(smooth_scores(cands, noop=1.0), cands).item())
        assert abs(sum(ps) / len(ps) - p_dense) < 0.02, (type(sampler).__name__, sum(ps) / len(ps), p_dense)


def test_deterministic_choice_agrees_across_samplers(grid):
    """With a clear best leg, sampled candidates reach the same decision as exhaustive
    scoring: step, with the same leg, near the same foothold."""
    valid = torch.ones(1, 4, *grid.size, dtype=torch.bool)
    field = dict(noop=-0.5, leg_bonus=(0.0, 1.0, 0.0, 0.0))
    dense = Dense().sample(valid, grid)
    dense_pick = select_deterministic(smooth_scores(dense, **field), dense)
    _, dense_leg, dense_xyz = dense.gather(dense_pick.index)
    assert dense_pick.index.item() != dense.noop_index and dense_leg.item() == 1  # the bonus leg

    generator = torch.Generator().manual_seed(1)
    agree = 0
    for _ in range(100):
        cands = UniformJitter(128).sample(valid, grid, generator=generator)
        pick = select_deterministic(smooth_scores(cands, **field), cands)
        is_step, leg, xyz = cands.gather(pick.index)
        agree += bool(is_step.item() and leg.item() == 1 and (xyz - dense_xyz).norm() < 3 * grid.resolution)
    assert agree >= 95, agree


def test_invalid_candidates_are_never_chosen(grid):
    valid = torch.zeros(2, 4, *grid.size, dtype=torch.bool)
    valid[1, 2, 5, 5] = True  # robot 1: exactly one legal foothold
    cands = UniformLattice(8).sample(valid, grid)
    scores = Scores(torch.full((2, 4, 8), 10.0), torch.zeros(2), torch.full((2, 4, 8), 0.2))
    pick = select_deterministic(scores, cands)
    assert pick.index[0].item() == cands.noop_index
    is_step, leg, _ = cands.gather(pick.index)
    assert is_step[1] and leg[1] == 2
    dist = FootstepDistribution(scores, cands, torch.tensor(0.05))
    for _ in range(20):
        s = dist.sample()
        assert s.index[0] == cands.noop_index
        assert cands.gather(s.index)[1][1] == 2 or s.index[1] == cands.noop_index


def test_distribution_log_prob(grid):
    valid = torch.ones(3, 4, *grid.size, dtype=torch.bool)
    cands = UniformJitter(16).sample(valid, grid)
    scores = smooth_scores(cands, noop=1.5)
    std = torch.tensor(0.05, requires_grad=True)
    dist = FootstepDistribution(scores, cands, std)
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    assert torch.isfinite(log_prob).all()
    # a no-op's duration doesn't enter the log-probability
    noop = torch.full((3,), cands.noop_index)
    a = dist.log_prob(Selection(noop, torch.zeros(3)))
    b = dist.log_prob(Selection(noop, torch.full((3,), 123.0)))
    assert torch.equal(a, b)
    # duration std is learnable through the step log-probability
    step = Selection(torch.zeros(3, dtype=torch.long), torch.full((3,), 0.25))
    dist.log_prob(step).sum().backward()
    assert std.grad is not None and std.grad != 0
    assert torch.isfinite(dist.entropy()).all()


def test_fixed_duration_is_never_sampled(grid):
    valid = torch.ones(2, 4, *grid.size, dtype=torch.bool)
    cands = Dense().sample(valid, grid)
    scores = smooth_scores(cands, noop=-5.0)
    scores.duration = torch.full_like(scores.duration, 0.25)
    dist = FootstepDistribution(scores, cands, None)
    selection = dist.sample()
    assert torch.all(selection.duration[selection.index < cands.noop_index] == 0.25)
    assert torch.allclose(dist.log_prob(selection), dist.categorical.log_prob(selection.index))
