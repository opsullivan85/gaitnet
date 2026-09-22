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


def test_resume_continues_the_run_and_tags_changed_params(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from gaitnet_sim.rl.mlflow_writer import MlflowLogWriter

    # the image has mlflow-skinny, which has no database backend
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    train_cfg = {"seed": 1, "num_steps_per_env": 24}

    monkeypatch.setattr("sys.argv", ["train", "--task", "GaitNet-Holes", "presets=gpu_mpc"])
    first = MlflowLogWriter(str(tmp_path / "2026-09-20_10-00-00"), tracking_uri=uri)
    first.store_config({}, train_cfg)
    first.add_scalar("loss", 1.0, global_step=0)
    first.stop()
    run_id = first.run.info.run_id

    monkeypatch.setattr("sys.argv", ["train", "--task", "GaitNet-Holes", "presets=gpu_mpc", "a.b=2"])
    resumed = MlflowLogWriter(str(tmp_path / "2026-09-21_10-00-00"), tracking_uri=uri, run_id=run_id)
    resumed.store_config({}, {**train_cfg, "num_steps_per_env": 48})
    resumed.add_scalar("loss", 0.5, global_step=1)
    resumed.stop()

    run = mlflow.get_run(run_id)
    assert resumed.run.info.run_id == run_id
    assert run.data.params["agent.num_steps_per_env"] == "24"
    assert run.data.params["override.a.b"] == "2"
    assert "48" in run.data.tags["resume.2026-09-21_10-00-00.changed_params"]
    assert "a.b=2" in run.data.tags["resume.2026-09-21_10-00-00.command"]
    history = mlflow.MlflowClient(uri).get_metric_history(run_id, "loss")
    assert [m.step for m in history] == [0, 1]
