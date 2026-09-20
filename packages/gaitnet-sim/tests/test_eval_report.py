"""The evaluation summary: metric definitions and the plot, on hand-made rows."""

from __future__ import annotations

import pytest

from gaitnet_sim.eval import report

STEP_DT = 0.5


def _row(difficulty, velocity, distance, steps, truncated, terminated_by=""):
    return {
        "difficulty": difficulty,
        "velocity": velocity,
        "trial": 0,
        "env": 0,
        "distance": distance,
        "steps": steps,
        "truncated": truncated,
        "terminated_by": terminated_by,
    }


ROWS = [
    # d=0: both survive; commanded 0.1 m/s * 10 steps * 0.5 s = 0.5 m, walked 0.5 and 0.25
    _row(0.0, 0.1, 0.5, 10, 1),
    _row(0.0, 0.1, 0.25, 10, 1),
    # d=0.2: one survives, one falls after 4 steps having walked 0.1 m (0.2 m commanded)
    _row(0.2, 0.1, 0.5, 10, 1),
    _row(0.2, 0.1, 0.1, 4, 0, "bad_height"),
]


def test_cells():
    per_cell = report.cells(ROWS, STEP_DT)
    assert per_cell[(0.0, 0.1)] == {"survival": 1.0, "distance_ratio": pytest.approx(0.75)}
    assert per_cell[(0.2, 0.1)] == {"survival": 0.5, "distance_ratio": pytest.approx(0.75)}


def test_summary_averages_cells_not_robots():
    rows = ROWS + [_row(0.2, 0.1, 0.5, 10, 1)] * 2  # d=0.2 now holds 4 robots, 3 survivors
    metrics = report.summarize(rows, STEP_DT)
    assert metrics["survival/d0.2_v0.1"] == 0.75
    assert metrics["survival_mean"] == pytest.approx((1.0 + 0.75) / 2)


def test_terminations_are_fractions_of_all_robots():
    metrics = report.summarize(ROWS, STEP_DT)
    assert metrics["terminated/truncated"] == 0.75
    assert metrics["terminated/bad_height"] == 0.25


def test_plot_has_a_line_per_velocity():
    rows = ROWS + [_row(0.0, 0.2, 1.0, 10, 1), _row(0.2, 0.2, 0.2, 5, 0, "bad_height")]
    fig = report.plot(rows, STEP_DT, title="t")
    assert len(fig.axes) == 2
    assert all(len(ax.get_lines()) == 2 for ax in fig.axes)


def test_eval_run_is_nested_in_the_training_runs_experiment(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # the skinny client has no SQL store
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    experiment = mlflow.create_experiment("training", artifact_location=str(tmp_path / "artifacts"))
    with mlflow.start_run(experiment_id=experiment) as training:
        pass
    csv = tmp_path / "eval.csv"
    csv.write_text("difficulty\n")

    run_id = report.log_to_mlflow(
        training.info.run_id, ROWS, STEP_DT, params={"task": "t"}, tags={"checkpoint": "c"}, csv_path=csv, run_name="eval"
    )

    run = mlflow.MlflowClient().get_run(run_id)
    assert run.info.experiment_id == experiment
    assert run.data.tags["mlflow.parentRunId"] == training.info.run_id
    assert run.data.metrics["survival_mean"] == pytest.approx(0.75)
