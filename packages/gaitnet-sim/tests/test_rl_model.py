"""GaitNetActor through RSL-RL's PPO (construct, act, update) on a fake environment.

Needs rsl_rl but not Isaac Lab.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.algorithms import PPO  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from gaitnet_core import action_layout  # noqa: E402
from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction  # noqa: E402
from gaitnet_core.candidates import Candidates  # noqa: E402
from gaitnet_core.grid import FootholdGrid  # noqa: E402

N, L, K, S = 8, 4, 16, 41
GRID = FootholdGrid(resolution=0.015, size=(9, 9), border=2)
SCORER = {"class_name": "CandidateScorer", "shared_sizes": [32, 32], "candidate_sizes": [16, 16], "trunk_sizes": [32, 32]}
SPATIAL = {"class_name": "DenseSpatialCNN", "grid": GRID.to_dict(), "channels": [4, 4], "state_sizes": [16], "noop_sizes": [8]}


def fake_obs(groups: tuple[str, ...] = ("state", "candidates", "terrain", "privileged", "base_command")) -> TensorDict:
    valid = torch.rand(N, L, K) < 0.7
    xyz = torch.randn(N, L, K, 3) * 0.1
    candidates = Candidates(xyz=xyz, valid=valid, log_q=torch.zeros(N, L, K))
    all_groups = {
        "state": torch.randn(N, S),
        "candidates": candidates.pack(),
        "terrain": -0.26 + 0.05 * torch.randn(N, L, *GRID.patch_size),
        "privileged": torch.randn(N, 10),
        "base_command": torch.rand(N, 3) * 0.2,
    }
    return TensorDict({name: all_groups[name] for name in groups}, batch_size=[N])


class FakeEnv:
    num_envs = N
    num_actions = action_layout.DIM
    cfg: dict = {}

    def get_observations(self) -> TensorDict:
        return fake_obs()


def make_ppo(schedule: str, network: dict = SCORER, critic_groups=("state",), **actor) -> PPO:
    cfg = {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": schedule,
            "desired_kl": 0.01,
            "learning_rate": 3e-4,
        },
        "actor": {
            "class_name": "gaitnet_sim.rl.model:GaitNetActor",
            "network": dict(network),
            "distribution_cfg": None,
            **actor,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "relu"},
        "obs_groups": {"actor": ["state"], "critic": list(critic_groups)},
        "num_steps_per_env": 6,
        "multi_gpu": None,
    }
    return PPO.construct_algorithm(fake_obs(), FakeEnv(), cfg, "cpu")


def rollout_and_update(ppo: PPO) -> dict:
    obs = fake_obs()
    for _ in range(6):
        with torch.inference_mode():
            actions = ppo.act(obs)
            assert actions.shape == (N, action_layout.DIM)
            obs = fake_obs()
            ppo.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    with torch.inference_mode():
        ppo.compute_returns(obs)
    return ppo.update()


@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_ppo_round(schedule):
    torch.manual_seed(0)
    losses = rollout_and_update(make_ppo(schedule))
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses


def test_ppo_round_spatial_cnn_crop_encoder_and_privileged_critic():
    torch.manual_seed(0)
    for network in (SPATIAL, {**SCORER, "candidate_features": "xyz_crop", "grid": GRID.to_dict()}):
        losses = rollout_and_update(make_ppo("adaptive", network=network, critic_groups=("state", "privileged")))
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), (network, losses)


def test_observers_run_only_when_acting():
    torch.manual_seed(0)
    # slows (to a stop) on any tick where some leg may step
    observers = {"step_confidence_slowdown": {"patience": 1, "margin": -1e9, "scale": 0.0}}
    actor = make_ppo("fixed", observers=observers).actor
    (observer,) = actor.observers
    obs = fake_obs()
    can_step = Candidates.unpack(obs["candidates"]).valid.flatten(1).any(dim=1)

    with torch.inference_mode():
        nudge = EnvAction.decode(actor(obs, stochastic_output=True)).nudge
    expected = torch.where(can_step.unsqueeze(-1), -obs["base_command"], 0.0)
    assert torch.allclose(nudge, expected)
    waited = observer._waiting.clone()

    # PPO's update pass: gradients on, the observer neither runs nor advances
    nudge = EnvAction.decode(actor(obs, stochastic_output=True)).nudge
    assert (nudge == 0).all() and torch.equal(observer._waiting, waited)

    # episode ends clear the ended envs' memory
    with torch.inference_mode():
        actor.reset(torch.tensor([1, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert observer._waiting[0] == 0 and observer._waiting[-1] == 0
    assert torch.equal(observer._waiting[1:-1], waited[1:-1])


def test_actor_checks_its_observation_groups():
    from gaitnet_sim.rl.model import GaitNetActor

    def actor(obs, **kwargs):
        return GaitNetActor(obs, {"actor": ["state"]}, "actor", action_layout.DIM, **kwargs)

    with pytest.raises(ValueError, match="terrain"):
        actor(fake_obs(("state", "candidates")), network=SPATIAL)
    with pytest.raises(ValueError, match="patches"):
        actor(fake_obs(), network={**SPATIAL, "grid": FootholdGrid(size=(11, 11), border=2).to_dict()})
    with pytest.raises(ValueError, match="base_command"):
        actor(fake_obs(("state", "candidates")), network=SCORER, observers={"step_confidence_slowdown": {}})
    # a network that doesn't read terrain doesn't need the group
    actor(fake_obs(("state", "candidates")), network=SCORER)


def test_actions_resolve_to_candidates():
    torch.manual_seed(0)
    ppo = make_ppo("fixed")
    obs = fake_obs()
    candidates = Candidates.unpack(obs["candidates"])
    for stochastic in (True, False):
        with torch.inference_mode():
            action = EnvAction.decode(ppo.actor(obs, stochastic_output=stochastic))
        is_step = action.leg != NO_STEP_LEG
        # a step's target is the chosen candidate, which must be valid
        _, leg, target = candidates.gather(action.choice_index)
        assert torch.equal(action.leg[is_step], leg[is_step])
        assert torch.allclose(action.target[is_step], target[is_step])
        flat_valid = candidates.valid.flatten(1)[is_step]
        assert flat_valid.gather(1, action.choice_index[is_step].unsqueeze(1)).all()
        assert (action.nudge == 0).all()


def test_log_prob_matches_sampling_distribution():
    torch.manual_seed(0)
    ppo = make_ppo("fixed")
    obs = fake_obs()
    with torch.inference_mode():
        outputs = ppo.actor(obs, stochastic_output=True)
        params = ppo.actor.output_distribution_params
        log_prob = ppo.actor.get_output_log_prob(outputs)
        assert torch.isfinite(log_prob).all()
        # KL of a distribution with itself is zero
        kl = ppo.actor.get_kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_rejects_distribution_cfg():
    obs = fake_obs()
    from gaitnet_sim.rl.model import GaitNetActor

    with pytest.raises(ValueError):
        GaitNetActor(
            obs,
            {"actor": ["state"]},
            "actor",
            action_layout.DIM,
            network={"class_name": "CandidateScorer"},
            distribution_cfg={"class_name": "GaussianDistribution"},
        )
