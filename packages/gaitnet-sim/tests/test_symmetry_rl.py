"""Symmetry-augmented PPO (`gaitnet_sim.rl.symmetry`) on a fake environment laid out like the
real one: every observation group mirrors term by term, and PPO's update runs with
augmentation, with the mirror loss, and with the metric alone.

Needs rsl_rl and Isaac Lab's managers (for the observation terms), not the simulator.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rsl_rl")
pytest.importorskip("isaaclab")

from tensordict import TensorDict  # noqa: E402

from gaitnet_core import action_layout  # noqa: E402
from gaitnet_core.candidates import Candidates  # noqa: E402
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim  # noqa: E402
from gaitnet_core.grid import FootholdGrid  # noqa: E402
from gaitnet_core.symmetry import mirror_leg_maps, mirror_state_vector  # noqa: E402
from gaitnet_sim.env import observations  # noqa: E402
from gaitnet_sim.rl import symmetry  # noqa: E402
from gaitnet_sim.rl.symmetry import SymmetricPPO, augment, mirror_actions, mirror_observations  # noqa: E402

N, L, K = 8, 4, 16
GRID = FootholdGrid(resolution=0.015, size=(9, 9), border=2)
S = feature_dim(DEFAULT_FEATURES, L)
SCORER = {"class_name": "CandidateScorer", "shared_sizes": [32, 32], "candidate_sizes": [16, 16], "trunk_sizes": [32, 32]}
SPATIAL = {"class_name": "DenseSpatialCNN", "grid": GRID.to_dict(), "channels": [4, 4], "state_sizes": [16], "noop_sizes": [8]}

# group -> [(term name, func, params, shape)], as ObservationsCfg lays them out
GROUPS = {
    "state": [("robot_state", observations.robot_state, {"features": list(DEFAULT_FEATURES)}, (S,))],
    "candidates": [("candidates", observations.footstep_candidates, {}, (L, K, 5))],
    "terrain": [("heights", observations.terrain_heights, {}, (L, *GRID.patch_size))],
    "privileged": [
        ("terrain_heights", observations.terrain_height_summary, {"cells": 5}, (L * 25,)),
        ("foothold_validity", observations.foothold_validity_summary, {"cells": 5}, (L * 25,)),
        ("base_clearance", observations.base_terrain_clearance, {}, (1,)),
        ("contact_forces", observations.foot_contact_forces, {}, (L,)),
    ],
    "base_command": [("base_command", observations.base_command, {}, (3,))],
    "footholds": [("foothold_fraction", observations.foothold_fraction, {}, (L,))],
}


def fake_obs(n: int = N) -> TensorDict:
    candidates = Candidates(
        xyz=torch.randn(n, L, K, 3) * 0.1, valid=torch.rand(n, L, K) < 0.7, log_q=torch.zeros(n, L, K)
    )
    data = {
        "state": torch.randn(n, S),
        "candidates": candidates.pack(),
        "terrain": -0.26 + 0.05 * torch.randn(n, L, *GRID.patch_size),
        "privileged": torch.randn(n, sum(shape[0] for *_, shape in GROUPS["privileged"])),
        "base_command": torch.rand(n, 3) * 0.2,
        "footholds": torch.rand(n, L),
    }
    return TensorDict(data, batch_size=[n])


class FakeEnv:
    """The parts of an Isaac Lab env (through RSL-RL's wrapper) that the mirror reads."""

    num_envs = N
    num_actions = action_layout.DIM

    def __init__(self, groups: dict = GROUPS):
        self.unwrapped = self
        self.observation_manager = SimpleNamespace(
            active_terms={group: [name for name, *_ in terms] for group, terms in groups.items()},
            group_obs_term_dim={group: [shape for *_, shape in terms] for group, terms in groups.items()},
        )
        self.cfg = SimpleNamespace(
            observations=SimpleNamespace(
                **{
                    group: SimpleNamespace(**{name: SimpleNamespace(func=f, params=p) for name, f, p, _ in terms})
                    for group, terms in groups.items()
                }
            )
        )

    def get_observations(self) -> TensorDict:
        return fake_obs()


def make_ppo(augmentation: bool, mirror_loss: bool, network: dict = SCORER) -> SymmetricPPO:
    env = FakeEnv()
    cfg = {
        "algorithm": {
            "class_name": "gaitnet_sim.rl.symmetry:SymmetricPPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": "fixed",
            "learning_rate": 3e-4,
            "symmetry_cfg": {
                "use_data_augmentation": augmentation,
                "use_mirror_loss": mirror_loss,
                "data_augmentation_func": augment,
                "mirror_loss_coeff": 0.5,
            },
        },
        "actor": {"class_name": "gaitnet_sim.rl.model:GaitNetActor", "network": dict(network), "distribution_cfg": None},
        "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "relu"},
        "obs_groups": {"actor": ["state"], "critic": ["state", "privileged"]},
        "num_steps_per_env": 6,
        "multi_gpu": None,
    }
    return SymmetricPPO.construct_algorithm(fake_obs(), env, cfg, "cpu")


def rollout_and_update(ppo) -> dict:
    obs = fake_obs()
    for _ in range(6):
        with torch.inference_mode():
            ppo.act(obs)
            obs = fake_obs()
            ppo.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    with torch.inference_mode():
        ppo.compute_returns(obs)
    return ppo.update()


def test_observation_mirror_is_an_involution_term_by_term():
    torch.manual_seed(0)
    env, obs = FakeEnv(), fake_obs()
    mirrored = mirror_observations(env, obs)
    twice = mirror_observations(env, mirrored)
    for group in obs.keys():
        assert torch.allclose(twice[group], obs[group]), group

    assert torch.equal(mirrored["state"], mirror_state_vector(obs["state"], DEFAULT_FEATURES))
    assert torch.equal(mirrored["terrain"], mirror_leg_maps(obs["terrain"]))
    # the privileged group splits into its terms: summaries swap legs and flip, clearance stays
    heights = obs["privileged"][:, : L * 25].reshape(N, L, 5, 5)
    assert torch.equal(mirrored["privileged"][:, : L * 25].reshape(N, L, 5, 5), mirror_leg_maps(heights))
    assert torch.equal(mirrored["privileged"][:, 2 * L * 25], obs["privileged"][:, 2 * L * 25])
    assert torch.equal(mirrored["privileged"][:, -L:], obs["privileged"][:, -L:][:, [1, 0, 3, 2]])
    assert torch.equal(mirrored["footholds"], obs["footholds"][:, [1, 0, 3, 2]])


def test_augment_appends_the_mirror_image():
    torch.manual_seed(0)
    env, obs = FakeEnv(), fake_obs()
    actions = torch.randn(N, action_layout.DIM)
    actions[:, 0] = torch.randint(0, L * K + 1, (N,)).float()
    obs_aug, actions_aug = augment(env, obs, actions)
    assert obs_aug.batch_size[0] == 2 * N and actions_aug.shape == (2 * N, action_layout.DIM)
    assert torch.equal(obs_aug[:N]["state"], obs["state"]) and torch.equal(actions_aug[:N], actions)
    assert torch.equal(actions_aug[N:], mirror_actions(env, actions))
    assert augment(env, None, actions)[0] is None and augment(env, obs, None)[1] is None


def test_unmirrored_terms_are_refused():
    groups = {**GROUPS, "extra": [("mystery", lambda env: None, {}, (3,))]}
    obs = fake_obs()
    obs["extra"] = torch.zeros(N, 3)
    with pytest.raises(KeyError, match="extra.mystery"):
        mirror_observations(FakeEnv(groups), obs)


@pytest.mark.parametrize(
    "augmentation, mirror_loss", [(True, False), (True, True), (False, True), (False, False)]
)
@pytest.mark.parametrize("network", [SCORER, SPATIAL], ids=["scorer", "spatial"])
def test_ppo_round(augmentation, mirror_loss, network):
    torch.manual_seed(0)
    ppo = make_ppo(augmentation, mirror_loss, network)
    assert isinstance(ppo.symmetry, symmetry.MirrorSymmetry)
    losses = rollout_and_update(ppo)
    assert "symmetry" in losses and losses["symmetry"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    assert all(torch.isfinite(p).all() for p in ppo.actor.parameters())


def test_mirror_loss_is_the_same_with_and_without_augmentation():
    """Both routes compare the same two distributions; only where they come from differs."""
    torch.manual_seed(0)
    ppo = make_ppo(True, True)
    obs = fake_obs()
    batch = SimpleNamespace(observations=obs)
    aug_obs, _ = augment(ppo.symmetry.env, obs, None)
    ppo.actor(aug_obs, stochastic_output=True)
    with_augmentation = ppo.symmetry.compute_loss(ppo.actor, batch, N)

    ppo.symmetry.use_data_augmentation = False
    ppo.actor(obs, stochastic_output=True)
    without = ppo.symmetry.compute_loss(ppo.actor, batch, N)
    assert torch.allclose(with_augmentation, without, rtol=1e-5)

    # a gradient on the network, none on the duration std
    without.backward()
    assert ppo.actor.duration_log_std.grad is None or ppo.actor.duration_log_std.grad == 0
    grads = [p.grad for p in ppo.actor.network.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any((g != 0).any() for g in grads)


def test_without_symmetry_it_is_ppo():
    ppo = make_ppo(True, False)
    cfg_less = SymmetricPPO(ppo.actor, ppo.critic, ppo.storage, device="cpu")
    assert cfg_less.symmetry is None
