"""The foothold map: its masks and scores agree with the planner's own, and the runtime hands
each plan to its on_plan callbacks before commanding."""

from __future__ import annotations

import pytest
import torch

from conftest import make_observation
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim
from gaitnet_core.foothold_map import foothold_map
from gaitnet_core.mock_robot import ReplayRobot
from gaitnet_core.networks import CandidateScorer
from gaitnet_core.planner import FootstepPlanner
from gaitnet_core.robot_spec import GO1
from gaitnet_core.runtime import PlannerRuntime
from gaitnet_core.samplers import Dense, UniformJitter
from gaitnet_core.selection import corrected_step_logits


def _planner(sampler):
    torch.manual_seed(0)
    net = CandidateScorer(feature_dim(DEFAULT_FEATURES, 4), shared_sizes=[32], candidate_sizes=[16], trunk_sizes=[32], use_bf16=False)
    return FootstepPlanner(net, GO1, make_observation(1).terrain.grid, DEFAULT_FEATURES, sampler)


def _rough_observation():
    obs = make_observation(3)
    obs.terrain.heights[:, :, 5:12, 8:20] = float("-inf")  # a hole
    obs.terrain.heights[:, :, 20:, :] += 0.05  # a step
    obs.terrain.heights[1, 2] -= 0.3  # robot 1's RL out of reach everywhere
    obs.state.gait_timing[2, 3, 1] = 0.1  # robot 2's RR swinging
    return obs


def test_masks_and_scores_match_a_dense_plan():
    planner = _planner(Dense())
    obs = _rough_observation()
    plan = planner.plan(obs)
    ids = [2, 1]
    fmap = foothold_map(planner, plan, obs, ids)

    cells = planner.rules.valid_cells(obs, GO1)[ids]
    assert torch.equal(fmap.terrain_ok, cells)
    assert torch.equal(fmap.allowed, planner.rules.valid(obs, GO1)[ids])
    assert not fmap.eligible[0, 3] and fmap.eligible[1].all()
    assert not fmap.terrain_ok[1, 2].any()

    allowed = fmap.allowed.flatten(2)
    step_logits = plan.scores.step_logits[ids]
    assert torch.allclose(fmap.raw.flatten(2)[allowed], step_logits[allowed], atol=1e-5)
    # the corrected map is the planner's correction wherever the leg may step
    corrected = corrected_step_logits(plan.scores, plan.candidates)[ids]
    assert torch.allclose(fmap.logits("corrected").flatten(2)[allowed], corrected[allowed], atol=1e-5)
    assert torch.allclose(fmap.leg_logit, torch.logsumexp(corrected, dim=-1))

    probabilities = fmap.step_probabilities()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2))
    assert probabilities[1, 2] == 0  # no foothold for that leg
    assert torch.isfinite(fmap.raw).all()  # unknown heights are still scored


def test_best_cell_is_the_deterministic_choice():
    planner = _planner(Dense())
    obs = _rough_observation()
    plan = planner.plan(obs)
    fmap = foothold_map(planner, plan, obs, range(3))
    for r in range(3):
        leg = int(fmap.leg[r])
        if leg < 0:
            continue
        i, j = fmap.best_cells()[r, leg]
        assert torch.allclose(fmap.grid.cell_centers()[i, j], fmap.target[r, :2])


def test_map_is_dense_whatever_the_planner_sampled():
    planner = _planner(UniformJitter(8))
    obs = _rough_observation()
    fmap = foothold_map(planner, planner.plan(obs), obs, [0])
    assert fmap.raw.shape == (1, 4, *planner.grid.size)
    with pytest.raises(ValueError):
        fmap.logits("sideways")


def test_runtime_calls_on_plan_before_commanding():
    observations = [make_observation(1) for _ in range(2)]
    robot = ReplayRobot(observations)
    seen = []

    def watch(plan, observation):
        seen.append((len(robot.commands), observation))

    runtime = PlannerRuntime(robot, _planner(Dense()), rate_hz=None, on_plan=[watch])
    runtime.run(max_ticks=2)
    assert [n for n, _ in seen] == [0, 1]
    assert all(torch.equal(obs.terrain.heights, observations[i].terrain.heights) for i, (_, obs) in enumerate(seen))
