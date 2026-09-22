"""Train a GaitNet task with RSL-RL, through Isaac Lab's training entry point.

    docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.train \\
        --task GaitNet-Holes --num_envs 1024

All of Isaac Lab's train arguments work (`--max_iterations`, `--seed`, `--checkpoint`, ...),
and so do presets and Hydra overrides of the env and agent cfgs, e.g.
`presets=spatial,privileged agent.algorithm.entropy_coef=0.01`; see
packages/gaitnet-sim/README.md. Runs are written to
`logs/rsl_rl/<experiment_name>/<timestamp>` and tracked in MLflow.

Resume with `--checkpoint latest` (or a path to a `model_*.pt`) and the same task, presets
and overrides. The terrain curriculum picks up from the checkpoint's levels when the
terrain rows are unchanged, and from random low rows otherwise. Add `--continue_mlflow` to
log to the checkpoint's MLflow run instead of starting a new one.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _diff_this_repo_only() -> None:
    """Have RSL-RL store this checkout's git diff instead of its own install's.

    It logs the diff of rsl_rl and Isaac Lab, neither of which is a git checkout in the image
    (hence "Could not find git repository ... Skipping"), and never our code.
    """
    from rsl_rl.utils.logger import Logger

    repo = str(Path(__file__).resolve())
    store = Logger._store_code_state

    def patched(self) -> list[str]:
        # the runner appends Isaac Lab's train script after the Logger exists, so replace at use
        self.git_status_repos = [repo]
        return store(self)

    Logger._store_code_state = patched


def _checkpoint_hooks(continue_mlflow: bool) -> None:
    """Save the terrain curriculum's row counts with every checkpoint, and restore them on
    resume (`--checkpoint`), so a resumed run starts on the terrain it had reached rather
    than on Isaac Lab's random low rows. See `gaitnet_sim.env.curriculum`.

    With `continue_mlflow`, a resumed run also logs to the checkpoint's MLflow run rather
    than a new one, found from the id `MlflowLogWriter` left in the checkpoint's run
    directory. The writer is only built when `learn()` starts, after the load, so the id
    reaches it through the logger cfg.
    """
    from rsl_rl.runners import OnPolicyRunner

    from gaitnet_sim.env.curriculum import restore_terrain_levels, terrain_levels_state
    from gaitnet_sim.rl.export import RUN_ID_FILE

    save, load = OnPolicyRunner.save, OnPolicyRunner.load

    def patched_save(self, path: str, infos: dict | None = None) -> None:
        state = terrain_levels_state(self.env.unwrapped)
        if state is not None:
            infos = {**(infos or {}), "terrain_levels": state}
        save(self, path, infos)

    def patched_load(self, path: str, *args, **kwargs) -> dict:
        infos = load(self, path, *args, **kwargs)
        env = self.env.unwrapped
        why_not = restore_terrain_levels(env, (infos or {}).get("terrain_levels"))
        if why_not is None:
            # every robot is still where the fresh env put it
            self.env.reset()
            print(f"[INFO] Restored terrain levels from {path}.")
        elif terrain_levels_state(env) is not None:
            print(f"[WARN] Not restoring terrain levels, {why_not}: keeping random placement.")

        if continue_mlflow:
            run_id_file = Path(path).parent / RUN_ID_FILE
            logger_cfg = self.logger.cfg.get("logger")
            if not isinstance(logger_cfg, dict) or not logger_cfg["class_name"].endswith("MlflowLogWriter"):
                raise SystemExit("--continue_mlflow needs the MLflow log writer")
            if not run_id_file.is_file():
                raise SystemExit(f"--continue_mlflow: no {RUN_ID_FILE} next to {path}")
            logger_cfg["run_id"] = run_id_file.read_text().strip()
            print(f"[INFO] Continuing MLflow run {logger_cfg['run_id']}.")
        return infos

    OnPolicyRunner.save, OnPolicyRunner.load = patched_save, patched_load


def main(argv: list[str] | None = None) -> None:
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

    argv = list(sys.argv[1:] if argv is None else argv)
    continue_mlflow = "--continue_mlflow" in argv
    if continue_mlflow:
        argv.remove("--continue_mlflow")
        if not any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in argv):
            raise SystemExit("--continue_mlflow resumes a run, it needs --checkpoint")

    _diff_this_repo_only()
    _checkpoint_hooks(continue_mlflow)

    run(["--external_callback", "gaitnet_sim.tasks.register", *argv])


if __name__ == "__main__":
    main()
