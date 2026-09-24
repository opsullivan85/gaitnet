"""The experimentation pieces: terrain sampling, the local-crop encoder, the dense spatial
CNN, refinement through it, and feedback observers."""

from types import SimpleNamespace

import pytest
import torch

from conftest import GROUND, make_observation
from gaitnet_core.bundle import BundleError, PolicyBundle, check_manifest, load_bundle, save_bundle
from gaitnet_core.candidates import Candidates
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim, state_vector
from gaitnet_core.networks import CANDIDATE_FEATURES, CandidateScorer, DenseSpatialCNN
from gaitnet_core.networks.candidate_scorer import EncoderInputs
from gaitnet_core.observers import BlockedLegRedirect, StepConfidenceSlowdown, combined_nudge
from gaitnet_core.planner import FootholdRules, FootstepPlanner
from gaitnet_core.refine import Refiner
from gaitnet_core.robot_spec import GO1
from gaitnet_core.samplers import Dense, UniformJitter
from gaitnet_core.terrain import sample_patch

STATE_DIM = feature_dim(DEFAULT_FEATURES, 4)


def patch_centres(grid) -> torch.Tensor:
    """(L, X, Y, 2) each leg's cell centres of the whole terrain patch, border included."""
    size_x, size_y = grid.patch_size
    x = (torch.arange(size_x) - (size_x - 1) / 2) * grid.resolution
    y = (torch.arange(size_y) - (size_y - 1) / 2) * grid.resolution
    local = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
    return local + grid.leg_centers()[:, None, None]


def per_leg(points: torch.Tensor, n: int) -> torch.Tensor:
    """(L, 2) one point per leg -> (n, L, 1, 2) sample_patch points."""
    return points.view(1, -1, 1, 2).expand(n, -1, 1, 2)


def test_sample_patch_reads_cells_and_interpolates(grid):
    torch.manual_seed(0)
    maps = torch.randn(2, 4, 3, *grid.patch_size)
    centres = patch_centres(grid)
    i, j = 5, 17
    xy = per_leg(centres[:, i, j], 2)
    assert torch.allclose(sample_patch(maps, xy, grid)[:, :, 0], maps[..., i, j], atol=1e-5)
    # halfway to the next cell in x (the first index)
    midpoint = per_leg((centres[:, i, j] + centres[:, i + 1, j]) / 2, 2)
    expected = (maps[..., i, j] + maps[..., i + 1, j]) / 2
    assert torch.allclose(sample_patch(maps, midpoint, grid)[:, :, 0], expected, atol=1e-5)
    # beyond the patch: the edge
    middle = grid.patch_size[1] // 2
    far = per_leg(centres[:, -1, middle] + torch.tensor([10.0, 0.0]), 2)
    assert torch.allclose(sample_patch(maps, far, grid)[:, :, 0], maps[..., -1, middle], atol=1e-5)
    # maps over the candidate grid only
    inner = torch.randn(2, 4, 3, *grid.size)
    xy = per_leg(grid.cell_centers()[:, 3, 20], 2)
    assert torch.allclose(sample_patch(inner, xy, grid)[:, :, 0], inner[..., 3, 20], atol=1e-5)


def test_crop_encoding_reads_terrain_around_the_candidate(grid):
    terrain = torch.full((1, 4, *grid.patch_size), GROUND)
    step_x = grid.patch_size[0] // 2 + 1
    terrain[..., step_x:, :] = GROUND + 0.05  # a step up just ahead of the grid's centre
    centres = patch_centres(grid)
    xy = centres[:, grid.patch_size[0] // 2, grid.patch_size[1] // 2]  # the centre cell, on the lower side
    xyz = torch.cat([xy, torch.full((4, 1), GROUND)], dim=-1).view(1, 4, 1, 3)
    encoding = CANDIDATE_FEATURES["xyz_crop"]
    features = encoding.encode(EncoderInputs(xyz, terrain, grid, crop_radius=2))
    assert features.shape == (1, 4, 1, encoding.dim(4, 2)) == (1, 4, 1, 4 + 3 + 25)
    crop = features[0, 0, 0, 7:].reshape(5, 5)
    # rows are x offsets -2..2: the step starts one cell ahead
    assert torch.allclose(crop[:3], torch.zeros(3, 5), atol=1e-6)
    assert torch.allclose(crop[3:], torch.full((2, 5), 0.05), atol=1e-6)


def test_terrain_networks_need_terrain_and_grid(grid):
    with pytest.raises(ValueError):
        CandidateScorer(STATE_DIM, candidate_features="xyz_crop")
    net = CandidateScorer(STATE_DIM, candidate_features="xyz_crop", grid=grid.to_dict(), use_bf16=False)
    obs = make_observation(2)
    cands = UniformJitter(4).sample(torch.ones(2, 4, *grid.size, dtype=torch.bool), grid)
    with pytest.raises(ValueError):
        net(state_vector(obs.state, DEFAULT_FEATURES), cands)
    assert net(state_vector(obs.state, DEFAULT_FEATURES), cands, obs.terrain.heights).step_logits.shape == (2, 4, 4)


def spatial(grid, **kwargs) -> DenseSpatialCNN:
    torch.manual_seed(0)
    return DenseSpatialCNN(STATE_DIM, grid.to_dict(), channels=[8, 8], state_sizes=[32, 16], noop_sizes=[16], use_bf16=False, **kwargs)


def test_spatial_cnn_shapes_chunking_and_unknown_terrain(grid):
    obs = make_observation(5)
    obs.terrain.heights[:, :, :4] = float("-inf")  # unscanned cells
    state = state_vector(obs.state, DEFAULT_FEATURES)
    valid = torch.ones(5, 4, *grid.size, dtype=torch.bool)
    cands = Dense().sample(valid, grid)
    net = spatial(grid, checkpoint_chunk_size=None)
    full = net(state, cands, obs.terrain.heights)
    assert full.step_logits.shape == (5, 4, grid.num_cells) and full.noop_logit.shape == (5,)
    assert torch.isfinite(full.step_logits).all() and torch.isfinite(full.noop_logit).all()
    assert ((full.duration >= 0.1) & (full.duration <= 0.3)).all()
    chunked = spatial(grid, checkpoint_chunk_size=2)
    chunked.load_state_dict(net.state_dict())
    with torch.enable_grad():
        out = chunked(state, cands, obs.terrain.heights)
    assert torch.allclose(out.step_logits, full.step_logits, atol=1e-5)
    assert torch.allclose(out.noop_logit, full.noop_logit, atol=1e-5)
    with pytest.raises(ValueError):
        net(state, cands)


def test_spatial_cnn_is_differentiable_in_the_foothold(grid):
    obs = make_observation(2)
    obs.terrain.heights[..., 15:, :] = GROUND + 0.04  # something for the score to vary with
    cands = UniformJitter(8).sample(torch.ones(2, 4, *grid.size, dtype=torch.bool), grid)
    xyz = cands.xyz.clone().requires_grad_(True)
    scores = spatial(grid)(state_vector(obs.state, DEFAULT_FEATURES), Candidates(xyz, cands.valid, cands.log_q), obs.terrain.heights)
    scores.step_logits.sum().backward()
    assert torch.isfinite(xyz.grad).all() and xyz.grad[..., :2].abs().sum() > 0


def test_refiner_through_spatial_cnn_under_inference_mode(grid):
    obs = make_observation(3)
    obs.terrain.heights[..., 15:, :] = GROUND + 0.04
    planner = FootstepPlanner(spatial(grid), GO1, grid, DEFAULT_FEATURES, Dense())
    refiner = Refiner(planner, steps=3)
    with torch.inference_mode():
        plan = planner.plan(obs)
        refined = refiner(plan, obs)
    assert torch.equal(refined.is_step, plan.is_step) and torch.equal(refined.leg, plan.leg)
    stepping = refined.is_step
    valid = planner.rules.valid(obs, GO1)
    cell, in_bounds = grid.xy_to_cell(refined.target[stepping, :2], refined.leg[stepping])
    assert in_bounds.all() and valid[stepping.nonzero().squeeze(-1), refined.leg[stepping], cell[:, 0], cell[:, 1]].all()


def fake_plan(marginals, noop):
    return SimpleNamespace(leg_marginals=torch.tensor(marginals), scores=SimpleNamespace(noop_logit=torch.tensor(noop)))


def test_slowdown_after_patient_waiting():
    observer = StepConfidenceSlowdown(patience=3, scale=0.5)
    base = torch.tensor([[0.2, 0.0, 0.1]] * 3)
    ninf = float("-inf")
    # robot 0 keeps waiting; robot 1 can't step at all (its gait's swings); robot 2 waits,
    # then steps
    waiting = fake_plan([[0.0, -1.0, ninf, ninf], [ninf] * 4, [0.0] * 4], [1.0, 1.0, 1.0])
    stepping = fake_plan([[0.0, -1.0, ninf, ninf], [ninf] * 4, [2.0] * 4], [1.0, 1.0, 1.0])
    for _ in range(2):
        assert (observer.observe(waiting, base).command_delta == 0).all()
    nudge = observer.observe(waiting, base).command_delta
    assert torch.allclose(nudge[0], -0.5 * base[0]) and torch.allclose(nudge[2], -0.5 * base[2])
    assert (nudge[1] == 0).all()
    nudge = observer.observe(stepping, base).command_delta
    assert torch.allclose(nudge[0], -0.5 * base[0]) and (nudge[2] == 0).all()
    observer.reset(torch.tensor([0]))
    assert (observer.observe(waiting, base).command_delta[0] == 0).all()
    total = combined_nudge([observer, StepConfidenceSlowdown(patience=1, scale=0.0)], waiting, base)
    assert torch.allclose(total.command_delta[2], -1.0 * base[2])


def test_redirect_away_from_blocked_legs():
    observer = BlockedLegRedirect()
    forward = torch.tensor([[0.2, 0.0, 0.1]] * 3)
    # robot 0: both front legs mostly blocked; robot 1: FL half blocked; robot 2: open ground
    plan = SimpleNamespace(foothold_fraction=torch.tensor([[0.2, 0.2, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0], [1.0] * 4]))
    delta = observer.observe(plan, forward).command_delta
    effective = forward + delta
    assert (delta[:, 2] == 0).all() and (delta[2] == 0).all()
    assert abs(effective[0, 0]) < 1e-6 and abs(effective[0, 1]) < 1e-6
    fl = observer.hip_directions[0]
    assert (effective[1, :2] @ fl) < (forward[1, :2] @ fl) and effective[1, 1] < 0  # slides towards FR
    # moving away from the blocked side is left alone
    assert (observer.observe(plan, -forward).command_delta == 0).all()
    with pytest.raises(ValueError):
        observer.observe(SimpleNamespace(foothold_fraction=None), forward)


def test_plan_reports_terrain_fraction_regardless_of_eligibility(grid):
    obs = make_observation(2)
    obs.state.gait_timing[..., 1] = 0.1  # every leg in swing: nothing may step
    planner = FootstepPlanner(spatial(grid), GO1, grid, DEFAULT_FEATURES, UniformJitter(8))
    plan = planner.plan(obs)
    assert not plan.candidates.valid.any()
    cells = planner.rules.valid_cells(obs, GO1)
    assert torch.allclose(plan.foothold_fraction, cells.flatten(2).float().mean(-1)) and (plan.foothold_fraction > 0).all()

def test_bundle_with_spatial_network_and_observers(tmp_path, grid):
    net = spatial(grid)
    bundle = PolicyBundle(
        actor=net,
        robot=GO1,
        grid=grid,
        features=DEFAULT_FEATURES,
        rules=FootholdRules(),
        train_sampler={"name": "uniform_jitter", "per_leg": 64},
        duration_std=0.05,
        extra={},
        observers={"step_confidence_slowdown": {"patience": 4}},
    )
    loaded = load_bundle(save_bundle(tmp_path / "policy.pt", bundle))
    (observer,) = loaded.make_observers()
    assert isinstance(observer, StepConfidenceSlowdown) and observer.patience == 4
    obs = make_observation(2)
    assert torch.allclose(bundle.planner().plan(obs).scores.step_logits, loaded.planner().plan(obs).scores.step_logits)

    manifest = torch.load(tmp_path / "policy.pt", weights_only=True)["manifest"]
    check_manifest(manifest)
    with pytest.raises(BundleError):
        check_manifest(dict(manifest, grid=dict(manifest["grid"], resolution=0.02)))
    with pytest.raises(BundleError):
        check_manifest(dict(manifest, observers={"no_such_observer": {}}))
