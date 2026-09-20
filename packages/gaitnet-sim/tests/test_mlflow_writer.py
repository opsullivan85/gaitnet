"""Run names and override params derived from the train command line."""

from __future__ import annotations

import pytest

pytest.importorskip("rsl_rl")

from gaitnet_sim.rl.mlflow_writer import _default_run_name, _overrides  # noqa: E402

ARGV = [
    "--task", "GaitNet-Holes", "--num_envs", "512", "--headless",
    "presets=gpu_mpc",
    "env.observations.candidates.candidates.params.sampler=uniform_lattice",
    "agent.actor.network.candidate_features=xy",
    "env.actions.footstep.observation_noise=None",
    "env.events.add_base_mass=None",
    "env.events.push_robot=None",
]  # fmt: skip


def test_overrides_are_kept_in_full():
    overrides = _overrides(ARGV)
    assert overrides["presets"] == "gpu_mpc"
    assert overrides["env.observations.candidates.candidates.params.sampler"] == "uniform_lattice"
    assert len(overrides) == 6


def test_name_is_short_and_says_how_many_overrides_it_left_out():
    name = _default_run_name("logs/x/2026-09-20_15-13-09", ARGV)
    assert name == (
        "Holes 512env gpu_mpc sampler=uniform_lattice candidate_features=xy"
        " observation_noise=None +2 2026-09-20_15-13-09"
    )
