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

N, L, K, S = 8, 4, 16, 41


def fake_obs() -> TensorDict:
    valid = torch.rand(N, L, K) < 0.7
    xyz = torch.randn(N, L, K, 3) * 0.1
    candidates = Candidates(xyz=xyz, valid=valid, log_q=torch.zeros(N, L, K))
    return TensorDict({"state": torch.randn(N, S), "candidates": candidates.pack()}, batch_size=[N])


class FakeEnv:
    num_envs = N
    num_actions = action_layout.DIM
    cfg: dict = {}

    def get_observations(self) -> TensorDict:
        return fake_obs()


def make_ppo(schedule: str) -> PPO:
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
            "network": {
                "class_name": "CandidateScorer",
                "shared_sizes": [32, 32],
                "candidate_sizes": [16, 16],
                "trunk_sizes": [32, 32],
            },
            "distribution_cfg": None,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "relu"},
        "obs_groups": {"actor": ["state"], "critic": ["state"]},
        "num_steps_per_env": 6,
        "multi_gpu": None,
    }
    return PPO.construct_algorithm(fake_obs(), FakeEnv(), cfg, "cpu")


@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_ppo_round(schedule):
    torch.manual_seed(0)
    ppo = make_ppo(schedule)
    obs = fake_obs()
    for _ in range(6):
        with torch.inference_mode():
            actions = ppo.act(obs)
            assert actions.shape == (N, action_layout.DIM)
            obs = fake_obs()
            ppo.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    with torch.inference_mode():
        ppo.compute_returns(obs)
    losses = ppo.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses


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
