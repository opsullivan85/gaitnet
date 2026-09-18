"""A run directory as Isaac Lab and RSL-RL write it -> a policy bundle that scores like the
trained actor. Needs isaaclab's configclass (the env contract) but not the simulator."""

from __future__ import annotations

import pytest
import torch
import yaml

pytest.importorskip("isaaclab")

from tensordict import TensorDict  # noqa: E402

from gaitnet_core import action_layout  # noqa: E402
from gaitnet_core.bundle import BundleError, load_bundle, save_bundle  # noqa: E402
from gaitnet_core.candidates import Candidates  # noqa: E402
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim  # noqa: E402
from gaitnet_core.robot_spec import ROBOTS  # noqa: E402
from gaitnet_sim.rl.export import bundle_from_run, latest_checkpoint  # noqa: E402
from gaitnet_sim.rl.model import GaitNetActor  # noqa: E402

N, K = 3, 8
NETWORK = {"class_name": "CandidateScorer", "shared_sizes": [16, 16], "candidate_sizes": [8, 8], "trunk_sizes": [16, 16]}


def write_run(tmp_path, actor: GaitNetActor, actor_groups=("state",)):
    env = {
        # tuples as yaml.dump writes them, like Isaac Lab's dump_yaml
        "gaitnet": {"robot": "go1", "grid_resolution": 0.02, "grid_size": (21, 21), "grid_border": 3,
                    "step_threshold": 0.03, "edge_margin": 1, "min_stance_after_step": 2},
        "observations": {
            "state": {"robot_state": {"params": {"features": list(DEFAULT_FEATURES)}}},
            "candidates": {"candidates": {"params": {"sampler": "uniform_jitter", "sampler_kwargs": {"per_leg": K}}}},
        },
        "actions": {"footstep": {"controller": {"class_type": "gaitnet_sim.controllers.pooled_mpc:PooledMpcController"}}},
        # other Python types Isaac Lab's dump tags, e.g. SceneEntityCfg's body_ids
        "rewards": {"foot_slip": {"params": {"asset_cfg": {"body_ids": slice(None)}}}},
    }
    agent = {"obs_groups": {"actor": list(actor_groups), "critic": ["state"]}, "actor": {"network": dict(NETWORK)}}
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text(yaml.dump(env))
    (tmp_path / "params" / "agent.yaml").write_text(yaml.dump(agent))
    for iteration in (5, 10):
        torch.save({"actor_state_dict": actor.state_dict(), "iter": iteration, "infos": None}, tmp_path / f"model_{iteration}.pt")


def make_actor() -> tuple[GaitNetActor, TensorDict]:
    state_dim = feature_dim(DEFAULT_FEATURES, ROBOTS["go1"].num_legs)
    candidates = Candidates(xyz=torch.randn(N, 4, K, 3) * 0.1, valid=torch.rand(N, 4, K) < 0.8, log_q=torch.zeros(N, 4, K))
    obs = TensorDict({"state": torch.randn(N, state_dim), "candidates": candidates.pack()}, batch_size=[N])
    actor = GaitNetActor(obs, {"actor": ["state"]}, "actor", action_layout.DIM, network=dict(NETWORK), duration_std=0.07)
    return actor, obs


def test_bundle_scores_like_the_actor(tmp_path):
    torch.manual_seed(0)
    actor, obs = make_actor()
    write_run(tmp_path, actor)

    bundle = bundle_from_run(tmp_path)
    assert bundle.extra["checkpoint"] == "model_10.pt" and bundle.extra["iteration"] == 10
    assert bundle.grid.resolution == 0.02 and tuple(bundle.grid.size) == (21, 21)
    assert bundle.rules.edge_margin == 1 and bundle.train_sampler == {"name": "uniform_jitter", "per_leg": K}
    assert bundle.duration_std == pytest.approx(0.07)

    candidates = Candidates.unpack(obs["candidates"])
    with torch.no_grad():
        expected = actor.network(obs["state"], candidates)
        loaded = load_bundle(save_bundle(tmp_path / "bundle.pt", bundle))
        got = loaded.actor(obs["state"], candidates)
    assert torch.equal(expected.step_logits, got.step_logits)
    assert torch.equal(expected.noop_logit, got.noop_logit)
    assert torch.equal(expected.duration, got.duration)


def test_rejects_actor_reading_other_groups(tmp_path):
    actor, _ = make_actor()
    write_run(tmp_path, actor, actor_groups=("state", "privileged"))
    with pytest.raises(BundleError):
        bundle_from_run(tmp_path)


def test_latest_checkpoint():
    assert latest_checkpoint(["model_9.pt", "model_10.pt", "events.out", "model_2.pt"]) == "model_10.pt"
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(["params"])
