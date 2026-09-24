import pytest
import torch

from conftest import make_observation
from gaitnet_core.bundle import BundleError, PolicyBundle, check_manifest, load_bundle, save_bundle
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim, state_vector
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.networks import CandidateScorer
from gaitnet_core.planner import FootholdRules
from gaitnet_core.robot_spec import GO1
from gaitnet_core.samplers import UniformJitter


def _network(**kwargs):
    torch.manual_seed(0)
    return CandidateScorer(
        state_dim=feature_dim(DEFAULT_FEATURES, 4),
        shared_sizes=[32, 32],
        candidate_sizes=[16],
        trunk_sizes=[32],
        use_bf16=False,
        **kwargs,
    )


def test_feature_layout():
    obs = make_observation(2)
    obs.state.gait_timing[0, 1] = torch.tensor([0.5, 0.1, 0.0])
    obs.state.gait_timing[0, 2] = torch.tensor([0.0, 0.0, 9.0])  # long stance, clipped
    v = state_vector(obs.state, ["gait_timing"])
    assert v.shape == (2, 12)
    assert v[0].tolist()[:4] == [0.0, 0.5, 0.0, 0.0]  # swing phase of FL, FR, RL, RR
    assert v[0, 4 + 1] == pytest.approx(0.1)  # remaining swing, FR
    assert v[0, 8 + 2] == pytest.approx(0.5)  # time since touchdown, RL, clipped
    assert state_vector(obs.state, DEFAULT_FEATURES).shape == (2, feature_dim(DEFAULT_FEATURES, 4))


def test_scorer_shapes_and_chunking(grid):
    obs = make_observation(5)
    valid = torch.ones(5, 4, *grid.size, dtype=torch.bool)
    cands = UniformJitter(8).sample(valid, grid)
    state = state_vector(obs.state, DEFAULT_FEATURES)
    net = _network(checkpoint_chunk_size=None)
    full = net(state, cands)
    assert full.step_logits.shape == (5, 4, 8) and full.noop_logit.shape == (5,)
    assert ((full.duration >= 0.1) & (full.duration <= 0.3)).all()
    chunked = _network(checkpoint_chunk_size=2)
    chunked.load_state_dict(net.state_dict())
    with torch.enable_grad():
        out = chunked(state, cands)
    assert torch.allclose(out.step_logits, full.step_logits, atol=1e-6)
    assert torch.allclose(out.noop_logit, full.noop_logit, atol=1e-6)


def _bundle(net):
    return PolicyBundle(
        actor=net,
        robot=GO1,
        grid=make_observation(1).terrain.grid,
        features=DEFAULT_FEATURES,
        rules=FootholdRules(),
        train_sampler={"name": "uniform_jitter", "per_leg": 64},
        duration_std=0.05,
        extra={"git_commit": "test"},
    )


def test_bundle_round_trip(tmp_path):
    net = _network()
    path = save_bundle(tmp_path / "policy.pt", _bundle(net))
    loaded = load_bundle(path)
    assert loaded.features == DEFAULT_FEATURES and loaded.extra["git_commit"] == "test"
    obs = make_observation(2)
    a = _bundle(net).planner().plan(obs)
    b = loaded.planner().plan(obs)
    assert torch.equal(a.selection.index, b.selection.index)
    assert torch.allclose(a.scores.step_logits, b.scores.step_logits)


def test_bundle_keeps_the_grid_centre_and_reads_format_2(tmp_path):
    bundle = _bundle(_network())
    bundle.grid = FootholdGrid(center=(0.01, 0.08))
    loaded = load_bundle(save_bundle(tmp_path / "policy.pt", bundle))
    assert loaded.grid == bundle.grid

    # format 2 had no grid centre: its grids were centred on the hip
    data = torch.load(save_bundle(tmp_path / "old.pt", _bundle(_network())), weights_only=True)
    del data["manifest"]["grid"]["center"]
    data["manifest"]["format_version"] = 2
    torch.save(data, tmp_path / "old.pt")
    assert load_bundle(tmp_path / "old.pt").grid.center == (0.0, 0.0)


def test_bundle_rejects_mismatches(tmp_path):
    net = _network()
    data_path = save_bundle(tmp_path / "policy.pt", _bundle(net))
    manifest = torch.load(data_path, weights_only=True)["manifest"]
    check_manifest(manifest)
    for key, value in [
        ("features", ["foot_pos", "no_such_feature"]),
        ("features", ["foot_pos"]),  # state_dim no longer matches
        ("robot", "spot"),
        ("format_version", 0),
    ]:
        bad = dict(manifest, **{key: value})
        with pytest.raises(BundleError):
            check_manifest(bad)
    low, high = GO1.swing_duration_range
    for network_args in [{"duration_range": [low, high + 0.1]}, {"fixed_duration": low - 0.05}]:
        actor = dict(manifest["actor"], config={**manifest["actor"]["config"], **network_args})
        with pytest.raises(BundleError):
            check_manifest(dict(manifest, actor=actor))


def test_fixed_duration_needs_no_duration_head(grid):
    obs = make_observation(3)
    valid = torch.ones(3, 4, *grid.size, dtype=torch.bool)
    cands = UniformJitter(8).sample(valid, grid)
    state = state_vector(obs.state, DEFAULT_FEATURES)
    net = _network(fixed_duration=0.25)
    assert net.duration_head is None
    assert torch.all(net(state, cands).duration == 0.25)
