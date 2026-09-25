"""The left/right mirror: an involution, consistent between the structured and flat forms,
and matched by the grid, the foothold rules, the samplers and the selection math, so a
mirrored sample is one the environment could really have produced."""

import pytest
import torch

from conftest import GROUND
from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.candidates import Candidates
from gaitnet_core.features import DEFAULT_FEATURES, FEATURES, state_vector
from gaitnet_core.planner import FootholdRules
from gaitnet_core.robot_spec import GO1
from gaitnet_core.samplers import Dense
from gaitnet_core.selection import Scores, action_logits
from gaitnet_core.state import Observation, RobotState, TerrainPatch
from gaitnet_core.symmetry import (
    LEG_MIRROR,
    action_index_mirror,
    mirror_action_index,
    mirror_candidates,
    mirror_env_action,
    mirror_leg_maps,
    mirror_legs,
    mirror_observation,
    mirror_state,
    mirror_state_vector,
)
from gaitnet_core.terrain import inner_heights

N, L = 6, 4


def random_state(n: int = N) -> RobotState:
    timing = torch.rand(n, L, 3)
    swinging = torch.rand(n, L) < 0.3
    timing[..., 0] = torch.where(swinging, timing[..., 0], 0.0)
    timing[..., 1] = torch.where(swinging, timing[..., 1] * 0.3, 0.0)
    timing[..., 2] = torch.where(swinging, 0.0, timing[..., 2])
    return RobotState(
        foot_pos=torch.randn(n, L, 3),
        foot_vel=torch.randn(n, L, 3),
        base_lin_vel=torch.randn(n, 3),
        base_ang_vel=torch.randn(n, 3),
        projected_gravity=torch.randn(n, 3),
        contact=torch.rand(n, L) < 0.5,
        gait_timing=timing,
        command=torch.randn(n, 3),
        base_command=torch.randn(n, 3),
    )


def rough_terrain(grid, n: int = N) -> TerrainPatch:
    """Blocks of random height around the ground, some unknown cells: edges everywhere."""
    x, y = grid.patch_size
    blocks = GROUND + 0.04 * torch.randint(-2, 3, (n, L, x // 4 + 1, y // 4 + 1)).float()
    heights = blocks.repeat_interleave(4, -2).repeat_interleave(4, -1)[..., :x, :y].clone()
    heights[torch.rand(heights.shape) < 0.02] = float("-inf")
    return TerrainPatch(heights, grid)


def assert_states_equal(a: RobotState, b: RobotState):
    for name in RobotState.__dataclass_fields__:
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_leg_mirror_swaps_sides():
    assert LEG_MIRROR == (1, 0, 3, 2)


def test_mirroring_twice_is_the_identity(grid):
    torch.manual_seed(0)
    observation = Observation(random_state(), rough_terrain(grid))
    twice = mirror_observation(mirror_observation(observation))
    assert_states_equal(twice.state, observation.state)
    assert torch.equal(twice.terrain.heights, observation.terrain.heights)

    vector = state_vector(observation.state, DEFAULT_FEATURES)
    assert torch.equal(mirror_state_vector(mirror_state_vector(vector, DEFAULT_FEATURES), DEFAULT_FEATURES), vector)


@pytest.mark.parametrize("names", [(name,) for name in FEATURES] + [DEFAULT_FEATURES], ids=lambda n: "+".join(n))
def test_state_vector_mirror_matches_the_state_mirror(names):
    """Each feature's declared mirror agrees with mirroring the RobotState it's computed from."""
    torch.manual_seed(0)
    state = random_state()
    expected = state_vector(mirror_state(state), names)
    assert torch.equal(mirror_state_vector(state_vector(state, names), names), expected)


def test_state_vector_mirror_checks_the_width():
    vector = state_vector(random_state(), DEFAULT_FEATURES)
    with pytest.raises(ValueError, match="state vector"):
        mirror_state_vector(vector[:, :-1], DEFAULT_FEATURES)


def test_grid_cells_mirror_onto_the_other_side(grid):
    """The terrain map flip is the geometric mirror: FR's cell (i, j) and FL's cell
    (i, size_y - 1 - j) are mirror images, and likewise at the back."""
    centers = grid.cell_centers()  # (L, X, Y, 2)
    mirrored = mirror_leg_maps(centers.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
    assert torch.allclose(mirrored[..., 0], centers[..., 0])
    assert torch.allclose(mirrored[..., 1], -centers[..., 1])


def test_foothold_rules_are_mirror_equivariant(grid):
    """The reach band, edges, unknown cells and leg eligibility all mirror with the
    observation, so a mirrored candidate set only holds footholds the mirrored robot may take."""
    torch.manual_seed(0)
    observation = Observation(random_state(), rough_terrain(grid))
    rules = FootholdRules()
    valid = rules.valid(observation, GO1)
    assert valid.any() and not valid.all()
    assert torch.equal(rules.valid(mirror_observation(observation), GO1), mirror_leg_maps(valid))


def test_mirrored_candidates_are_the_mirrored_terrains_candidates(grid):
    """Dense candidates of the mirror image are the mirror of the dense candidates, slot for
    slot once each leg's y index is reversed."""
    torch.manual_seed(0)
    observation = Observation(random_state(), rough_terrain(grid))
    rules = FootholdRules()

    def dense(obs):
        valid = rules.valid(obs, GO1)
        return Dense().sample(valid, grid, heights=inner_heights(obs.terrain.heights, grid))

    mirrored = mirror_candidates(dense(observation))
    direct = dense(mirror_observation(observation))
    size_x, size_y = grid.size
    reverse_y = torch.arange(size_x * size_y).view(size_x, size_y).flip(-1).flatten()
    assert torch.equal(mirrored.valid, direct.valid[..., reverse_y])
    assert torch.allclose(mirrored.xyz, direct.xyz[..., reverse_y, :], atol=1e-6)


def test_action_index_mirror():
    k = 5
    index = torch.arange(L * k + 1)
    mirrored = mirror_action_index(index, L, k)
    assert mirrored[-1] == L * k  # the no-op stays
    assert torch.equal(mirror_action_index(mirrored, L, k), index)
    # FL slot 3 is FR slot 3
    assert mirrored[3] == k + 3 and mirrored[k + 3] == 3
    assert torch.equal(action_index_mirror(L, k), mirrored)


def test_selection_logits_mirror_with_the_scores():
    """With scores that mirror, the corrected logits (-log q, -log N_valid, masks) permute
    exactly as the action index does."""
    torch.manual_seed(0)
    k = 7
    candidates = Candidates(xyz=torch.randn(N, L, k, 3), valid=torch.rand(N, L, k) < 0.6, log_q=torch.randn(N, L, k))
    scores = Scores(step_logits=torch.randn(N, L, k), noop_logit=torch.randn(N), duration=torch.rand(N, L, k))
    mirrored_scores = Scores(
        step_logits=mirror_legs(scores.step_logits), noop_logit=scores.noop_logit, duration=mirror_legs(scores.duration)
    )
    logits = action_logits(scores, candidates)
    mirrored = action_logits(mirrored_scores, mirror_candidates(candidates))
    assert torch.equal(mirrored, logits[:, action_index_mirror(L, k)])


def test_env_action_mirror():
    torch.manual_seed(0)
    k = 5
    candidates = Candidates(xyz=torch.randn(N, L, k, 3), valid=torch.ones(N, L, k, dtype=torch.bool), log_q=torch.zeros(N, L, k))
    index = torch.randint(0, L * k + 1, (N,))
    index[0] = L * k
    is_step, leg, target = candidates.gather(index)
    action = EnvAction(
        choice_index=index,
        duration=torch.rand(N),
        leg=torch.where(is_step, leg, NO_STEP_LEG),
        target=target,
        nudge=torch.randn(N, 3),
    )
    mirrored = EnvAction.decode(mirror_env_action(action, L, k).encode())
    # it resolves to the mirrored candidate, and the no-op stays a no-op
    m_is_step, m_leg, m_target = mirror_candidates(candidates).gather(mirrored.choice_index)
    assert torch.equal(m_is_step, is_step)
    assert torch.equal(mirrored.leg, torch.where(is_step, m_leg, NO_STEP_LEG))
    assert torch.allclose(mirrored.target, m_target)
    assert torch.equal(mirrored.nudge[:, 1:], -action.nudge[:, 1:]) and torch.equal(mirrored.duration, action.duration)
    twice = mirror_env_action(mirrored, L, k)
    assert torch.equal(twice.encode(), action.encode())
