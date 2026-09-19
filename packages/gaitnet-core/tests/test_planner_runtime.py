import subprocess
import sys

import torch
import torch.nn as nn

from conftest import make_observation
from gaitnet_core.action_layout import DIM, EnvAction
from gaitnet_core.features import DEFAULT_FEATURES, feature_dim, state_vector
from gaitnet_core.interfaces import Nudge
from gaitnet_core.mock_robot import ReplayRobot
from gaitnet_core.networks import CandidateScorer
from gaitnet_core.planner import FootstepPlanner
from gaitnet_core.refine import refine_plan
from gaitnet_core.robot_spec import GO1
from gaitnet_core.runtime import PlannerRuntime
from gaitnet_core.samplers import Dense, UniformJitter
from gaitnet_core.selection import Scores


def _planner(sampler=None):
    torch.manual_seed(0)
    net = CandidateScorer(feature_dim(DEFAULT_FEATURES, 4), shared_sizes=[32], candidate_sizes=[16], trunk_sizes=[32], use_bf16=False)
    return FootstepPlanner(net, GO1, make_observation(1).terrain.grid, DEFAULT_FEATURES, sampler or UniformJitter(16))


def test_plan_and_env_action_round_trip():
    obs = make_observation(4)
    plan = _planner().plan(obs, deterministic=False, generator=torch.Generator().manual_seed(0))
    action = plan.env_action(Nudge(torch.ones(4, 3)))
    encoded = action.encode()
    assert encoded.shape == (4, DIM)
    decoded = EnvAction.decode(encoded)
    assert torch.equal(decoded.choice_index, plan.selection.index)
    command = decoded.footstep_command()
    assert torch.equal(command.active, plan.is_step)
    assert torch.allclose(command.target[command.active], plan.target[plan.is_step])
    assert torch.equal(decoded.nudge_command().command_delta, torch.ones(4, 3))


def test_swinging_legs_get_no_candidates():
    obs = make_observation(2)
    obs.state.gait_timing[0, :2, 1] = 0.1  # two legs swinging: robot 0 may not step at all
    obs.state.gait_timing[1, 3, 1] = 0.1  # RR swinging
    plan = _planner().plan(obs)
    assert plan.candidates.num_valid()[0].sum() == 0 and not plan.is_step[0]
    assert plan.candidates.num_valid()[1, 3] == 0 and plan.candidates.num_valid()[1, :3].min() > 0


class _Slowdown:
    def __init__(self):
        self.calls = 0

    def reset(self, robot_ids=None):
        self.calls = 0

    def observe(self, plan, base_command):
        self.calls += 1
        return Nudge(-0.5 * base_command)


def test_runtime_with_replay_robot_and_observer():
    observations = [make_observation(1) for _ in range(3)]
    for obs in observations:
        obs.state.base_command[:] = torch.tensor([0.2, 0.0, 0.0])
        # what the robot tracks after last tick's nudge; observers work from the base command
        obs.state.command[:] = torch.tensor([0.1, 0.0, 0.0])
    robot = ReplayRobot(observations)
    observer = _Slowdown()
    runtime = PlannerRuntime(robot, _planner(Dense()), observers=[observer], rate_hz=1000)
    assert runtime.run(max_ticks=3) == 3
    assert observer.calls == 3 and len(robot.commands) == 3
    for footsteps, nudge in robot.commands:
        assert torch.allclose(nudge.command_delta, torch.tensor([[-0.1, 0.0, 0.0]]))
        assert footsteps[0].target.shape == (1, 3)


class _Bowl(nn.Module):
    """Scores points by closeness to an off-lattice optimum, the same for every leg."""

    config: dict = {}

    def __init__(self, optimum):
        super().__init__()
        self.optimum = torch.tensor(optimum)

    def forward(self, state, candidates, terrain=None):
        d = ((candidates.xyz[..., :2] - self.optimum) ** 2).sum(-1)
        n = candidates.num_robots
        return Scores(step_logits=5 - 200 * d, noop_logit=torch.zeros(n), duration=0.2 + 0 * d)


def test_refine_moves_off_lattice_toward_optimum(grid):
    obs = make_observation(2)
    optimum = (0.0123, -0.0071)
    planner = FootstepPlanner(_Bowl(optimum), GO1, grid, DEFAULT_FEATURES, Dense())
    plan = planner.plan(obs)
    assert plan.is_step.all()
    valid = planner.rules.valid(obs, GO1)
    refined = refine_plan(planner.network, state_vector(obs.state, DEFAULT_FEATURES), plan, valid, grid, steps=8)
    target = torch.tensor(optimum)
    before = (plan.target[:, :2] - target).norm(dim=-1)
    after = (refined.target[:, :2] - target).norm(dim=-1)
    assert (after < before).all() and (after < 0.3 * grid.resolution).all()
    # refined footholds stay on valid cells
    cell, in_bounds = grid.xy_to_cell(refined.target[:, :2])
    assert in_bounds.all() and valid[torch.arange(2), refined.leg, cell[:, 0], cell[:, 1]].all()


def test_core_imports_no_simulator_or_rl_library():
    code = (
        "import sys, gaitnet_core, gaitnet_core.bundle, gaitnet_core.runtime, gaitnet_core.refine, "
        "gaitnet_core.mock_robot, gaitnet_core.observers\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('isaaclab', 'isaacsim', 'omni', 'carb', 'rsl_rl', 'gaitnet', 'gaitnet_mpc'))\n"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_out_of_range_index_is_a_no_op(grid):
    from gaitnet_core.samplers import UniformLattice

    valid = torch.ones(3, 4, *grid.size, dtype=torch.bool)
    cands = UniformLattice(8).sample(valid, grid)
    is_step, _, xyz = cands.gather(torch.tensor([-1, cands.noop_index, cands.noop_index + 5]))
    assert not is_step.any() and (xyz == 0).all()
