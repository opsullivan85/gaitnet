"""Train a GaitNet task with RSL-RL, through Isaac Lab's training entry point.

    docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.train \\
        --task GaitNet-Holes --num_envs 1024

All of Isaac Lab's train arguments work (`--max_iterations`, `--seed`, `--checkpoint`, ...),
and so do presets and Hydra overrides of the env and agent cfgs, e.g.
`presets=spatial,privileged agent.algorithm.entropy_coef=0.01`; see
packages/gaitnet-sim/README.md. Runs are written to
`logs/rsl_rl/<experiment_name>/<timestamp>` and tracked in MLflow.
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


def main(argv: list[str] | None = None) -> None:
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

    _diff_this_repo_only()

    run(["--external_callback", "gaitnet_sim.tasks.register", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
