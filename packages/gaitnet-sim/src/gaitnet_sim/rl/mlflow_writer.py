"""RSL-RL log writer for MLflow.

Selected with `logger = {"class_name": "gaitnet_sim.rl.mlflow_writer:MlflowLogWriter", ...}` in
the runner cfg. Scalars also go to a local TensorBoard log in the run directory, so a run is
readable without the server. Checkpoints (`checkpoints/`) and the run directory's `params/`
(the env and agent cfgs Isaac Lab dumps) are uploaded as artifacts, which is all
`gaitnet_sim.rl.export` needs to build a policy bundle.

The tracking server comes from `MLFLOW_TRACKING_URI` (set by docker/compose.yaml) unless
`tracking_uri` is given; with neither, MLflow writes to ./mlruns.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from dataclasses import asdict, is_dataclass

from torch.utils.tensorboard import SummaryWriter

from rsl_rl.utils.log_writer import LogWriter


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _to_dict(cfg: dict | object) -> dict:
    """A JSON-safe dict of a config; values JSON can't hold become their str()."""
    if isinstance(cfg, dict):
        out = cfg
    elif hasattr(cfg, "to_dict"):
        out = cfg.to_dict()  # type: ignore[union-attr]
    elif is_dataclass(cfg):
        out = asdict(cfg)  # type: ignore[arg-type]
    else:
        out = {"repr": repr(cfg)}
    return json.loads(json.dumps(out, default=str))


def _flatten(prefix: str, value, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        # MLflow caps parameter values at a few thousand characters
        out[prefix] = str(value)[:500]


class MlflowLogWriter(SummaryWriter, LogWriter):
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "gaitnet",
        tracking_uri: str | None = None,
        run_name: str | None = None,
    ):
        import mlflow

        super().__init__(log_dir, flush_secs=10)
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        tags = {"log_dir": os.path.abspath(log_dir)}
        commit = _git_commit()
        if commit:
            tags["git_commit"] = commit
        self.run = mlflow.start_run(run_name=run_name or os.path.basename(os.path.normpath(log_dir)), tags=tags)
        # one request per iteration rather than per scalar
        self._step: int | None = None
        self._metrics: dict[str, float] = {}
        self._params_logged = False

    def _flush(self) -> None:
        if self._metrics:
            self._mlflow.log_metrics(self._metrics, step=self._step or 0)
            self._metrics = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False) -> None:
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        if global_step != self._step:
            self._flush()
            self._step = global_step
        self._metrics[tag] = float(scalar_value)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        env = _to_dict(env_cfg)
        self._mlflow.log_dict(env, "config/env.json")
        self._mlflow.log_dict(_to_dict(train_cfg), "config/agent.json")
        params: dict[str, str] = {}
        for key in ("algorithm", "actor", "critic", "obs_groups", "num_steps_per_env", "seed"):
            if key in train_cfg:
                _flatten(f"agent.{key}", train_cfg[key], params)
        if "gaitnet" in env:
            _flatten("env.gaitnet", env["gaitnet"], params)
        if "scene" in env:
            params["env.num_envs"] = str(env["scene"].get("num_envs"))
        self._mlflow.log_params(params)

    def _log_params_dir(self) -> None:
        # Isaac Lab writes params/ after the runner (and this writer) exists, so it goes up
        # with the first checkpoint rather than in store_config
        params_dir = os.path.join(self.log_dir, "params")
        if not self._params_logged and os.path.isdir(params_dir):
            self._mlflow.log_artifacts(params_dir, artifact_path="params")
            self._params_logged = True

    def save_model(self, model_path: str, it: int) -> None:
        self._log_params_dir()
        self._mlflow.log_artifact(model_path, artifact_path="checkpoints")

    def save_file(self, path: str) -> None:
        self._mlflow.log_artifact(path, artifact_path="files")

    def save_video(self, video: pathlib.Path, it: int) -> None:
        self._mlflow.log_artifact(str(video), artifact_path="videos")

    def stop(self) -> None:
        self._flush()
        self._log_params_dir()
        self._mlflow.end_run()
        self.close()
