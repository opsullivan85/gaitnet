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


def main(argv: list[str] | None = None) -> None:
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

    run(["--external_callback", "gaitnet_sim.tasks.register", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
