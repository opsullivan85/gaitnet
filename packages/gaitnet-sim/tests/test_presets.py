"""Experiment presets switch the env's observation groups and the agent's matching parts
together. Resolved the way Isaac Lab's entry points do it, without the simulator."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("isaaclab")

from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

from gaitnet_core.grid import FootholdGrid  # noqa: E402
from gaitnet_sim.tasks import register  # noqa: E402


def resolve(*overrides: str):
    register()
    return resolve_task_config("GaitNet-Pillars", "rsl_rl_cfg_entry_point", overrides=list(overrides))


def test_defaults_leave_the_optional_groups_off():
    env, agent = resolve()
    observations = env.observations
    assert observations.terrain is None and observations.privileged is None and observations.base_command is None
    assert observations.footholds is None
    assert agent.actor.network["class_name"] == "CandidateScorer" and agent.actor.network["candidate_features"] == "xyz"
    assert agent.actor.observers == {} and agent.obs_groups["critic"] == ["state"]


def test_presets_compose():
    env, agent = resolve("presets=spatial,privileged,slowdown")
    observations = env.observations
    assert observations.terrain is not None and observations.privileged is not None
    assert observations.base_command is not None
    assert agent.actor.network["class_name"] == "DenseSpatialCNN"
    assert FootholdGrid.from_dict(agent.actor.network["grid"]) == env.gaitnet.foothold_grid()
    assert agent.obs_groups == {"actor": ["state"], "critic": ["state", "privileged"]}
    assert list(agent.actor.observers) == ["step_confidence_slowdown"]

    env, agent = resolve("presets=slowdown_redirect")
    assert env.observations.base_command is not None
    assert list(agent.actor.observers) == ["step_confidence_slowdown", "blocked_leg_redirect"]
    assert env.observations.footholds is not None

    env, agent = resolve("presets=crop")
    assert env.observations.terrain is not None and env.observations.privileged is None
    assert agent.actor.network["candidate_features"] == "xyz_crop"


def test_swing_duration_ablation_fixes_the_duration():
    _, agent = resolve()
    assert "fixed_duration" not in agent.actor.network

    _, agent = resolve("presets=swing_duration_ablation")
    assert agent.actor.network["fixed_duration"] == 0.25
    assert agent.actor.network["class_name"] == "CandidateScorer"

    _, agent = resolve("presets=swing_duration_ablation", "agent.actor.network.fixed_duration=0.3")
    assert agent.actor.network["fixed_duration"] == 0.3


def test_gpu_mpc_preset_swaps_the_controller():
    """The low-level controller is a preset rather than an override, because Isaac Lab
    reads a whole-cfg override as choosing a preset by name. Its own fields still take
    overrides, which is how the solver budget is dialled."""
    from gaitnet_sim.controllers import BatchedMpc, PooledMpcController

    env, _ = resolve()
    assert env.actions.footstep.controller.class_type is PooledMpcController

    env, _ = resolve("presets=gpu_mpc")
    controller = env.actions.footstep.controller
    assert controller.class_type is BatchedMpc
    # the two controllers have to agree on the things the env depends on
    default = resolve()[0].actions.footstep.controller
    assert controller.iterations_between_mpc == default.iterations_between_mpc
    assert controller.foot_radius == default.foot_radius

    env, _ = resolve("presets=gpu_mpc", "env.actions.footstep.controller.solver_iterations=120")
    assert env.actions.footstep.controller.solver_iterations == 120


def test_symmetry_preset_selects_mirror_augmentation():
    from rsl_rl.utils import resolve_callable

    from gaitnet_sim.rl.symmetry import SymmetricPPO, augment

    _, agent = resolve()
    assert agent.algorithm.class_name == "PPO" and agent.algorithm.symmetry_cfg is None

    _, agent = resolve("presets=symmetry,privileged")
    symmetry = agent.algorithm.symmetry_cfg
    assert symmetry.use_data_augmentation and not symmetry.use_mirror_loss
    assert agent.obs_groups["critic"] == ["state", "privileged"]
    # what the runner hands RSL-RL
    algorithm = agent.to_dict()["algorithm"]
    assert resolve_callable(algorithm["class_name"]) is SymmetricPPO
    assert resolve_callable(algorithm["symmetry_cfg"]["data_augmentation_func"]) is augment

    _, agent = resolve(
        "presets=symmetry",
        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
        "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.1",
    )
    assert agent.algorithm.symmetry_cfg.use_mirror_loss and agent.algorithm.symmetry_cfg.mirror_loss_coeff == 0.1

    _, agent = resolve("presets=symmetry", "agent.algorithm.symmetry_cfg.use_data_augmentation=False")
    assert not agent.algorithm.symmetry_cfg.use_data_augmentation


def test_every_observation_term_has_a_mirror():
    """Symmetry augmentation mirrors every group the env has, so a new term needs a mirror."""
    from isaaclab.managers import ObservationTermCfg

    from gaitnet_sim.rl.symmetry import TERM_MIRRORS

    env, _ = resolve("presets=spatial,privileged,slowdown_redirect")
    groups = {name: group for name, group in vars(env.observations).items() if group is not None}
    assert set(groups) == {"state", "candidates", "terrain", "privileged", "base_command", "footholds"}
    for name, group in groups.items():
        terms = {term_name: term for term_name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}
        assert terms, name
        for term_name, term in terms.items():
            assert term.func in TERM_MIRRORS, f"{name}.{term_name}"


def test_readme_override_recipes():
    """The recipes in packages/gaitnet-sim/README.md resolve as documented."""
    # Isaac Lab 3 applies env./agent. overrides itself, as Python literals
    env, agent = resolve(
        "env.observations.state.robot_state.params.features=['foot_pos','base_lin_vel','command','gait_timing']",
        "env.observations.candidates.candidates.params.sampler=uniform_lattice",
        "env.observations.candidates.candidates.params.sampler_kwargs.per_leg=32",
        "env.gaitnet.min_stance_after_step=3",
        "env.rewards.xy_tracking.params.command=base",
        "agent.actor.network.trunk_sizes=[256,256]",
        "agent.algorithm.entropy_coef=0.01",
    )
    assert env.observations.state.robot_state.params["features"] == ["foot_pos", "base_lin_vel", "command", "gait_timing"]
    candidates = env.observations.candidates.candidates.params
    assert candidates["sampler"] == "uniform_lattice" and candidates["sampler_kwargs"]["per_leg"] == 32
    assert env.gaitnet.min_stance_after_step == 3
    assert env.rewards.xy_tracking.params["command"] == "base"
    assert agent.actor.network["trunk_sizes"] == [256, 256] and agent.algorithm.entropy_coef == 0.01

    env, agent = resolve(
        "presets=spatial,slowdown",
        "agent.actor.network.channels=[8,8,8]",
        "agent.actor.observers.step_confidence_slowdown.patience=5",
        "agent.actor.observers.step_confidence_slowdown.scale=0.7",
    )
    assert agent.actor.network["class_name"] == "DenseSpatialCNN" and agent.actor.network["channels"] == [8, 8, 8]
    assert agent.actor.observers == {"step_confidence_slowdown": {"patience": 5, "margin": 0.0, "scale": 0.7}}


def test_cfg_modules_stay_free_of_usd():
    # Isaac Lab's entry points resolve cfgs before Kit starts; an early pxr breaks Kit's USD
    code = (
        "import sys, gaitnet_sim.env.env_cfg, gaitnet_sim.rl.agent_cfg\n"
        "sys.exit(1 if 'pxr' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
