# Containers

The simulator runs in Docker: Isaac Sim 6.1 with Isaac Lab 3.0, plus this repository's
packages. Everything is driven from the repository root with
`docker compose -f docker/compose.yaml ...`.

## Images

- `gaitnet/isaaclab:3.0.0-11508e5` (`docker/build_isaaclab.sh`): Isaac Lab's own image,
  built from its `docker/Dockerfile.base` at commit `11508e5` of the `release/3.0.0`
  branch. That commit pins Isaac Sim 6.1.0 (`nvcr.io/nvidia/isaac-sim:6.1.0`) and
  `rsl-rl-lib==5.5.1`. Only the `rsl-rl` and `test` extras are installed. Build it once
  per Isaac Lab pin; it takes a while. The script adds two uv settings to Isaac Lab's
  Dockerfile (4 parallel downloads rather than 50, a 300 s timeout), because the large
  CUDA wheels time out on a ~20 Mbit/s link otherwise.
- `gaitnet/sim:dev` (service `sim`): the above plus `gaitnet-mpc` (compiled, regular
  install) and `gaitnet-core` / `gaitnet-sim` (editable). Rebuild when a package's
  dependencies or anything in `gaitnet-mpc` changes (its Python too, since it isn't
  editable); edits to core and sim don't need a rebuild.

```bash
docker/build_isaaclab.sh
docker compose -f docker/compose.yaml build sim
```

To move to a newer Isaac Lab, change `ISAACLAB_COMMIT` in `build_isaaclab.sh` and
`ISAACLAB_IMAGE` in `Dockerfile.sim`.

## Running

The container's entry point is Isaac Lab's Python (`isaaclab.sh -p`) and its working
directory is the checkout, bind-mounted at `/workspace/gaitnet`:

```bash
# scripted walk, the sim smoke test (flat ground; --task GaitNet-Pillars --difficulty 0.3
# for pillars)
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.walk --num_envs 4

# train (tasks GaitNet-Holes and GaitNet-Pillars; Isaac Lab's train entry point, so
# --max_iterations, --seed, --checkpoint, ... and Hydra overrides such as
# agent.algorithm.entropy_coef=0.01 all work)
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.train --task GaitNet-Holes --num_envs 1024

# a policy bundle from a run, by directory or MLflow run id
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.export_bundle \
    --run logs/rsl_rl/gaitnet_holes/<timestamp> --out data/bundles/policy.pt

# evaluate a bundle across terrain difficulties and velocities (writes data/evaluations/*.csv)
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.eval_sweep --bundle data/bundles/policy.pt \
    --task GaitNet-Pillars

# tests for the MPC need the compiled extension
docker compose -f docker/compose.yaml run --rm sim -m pytest packages/gaitnet-mpc/tests

# tests (core and the sim package's simulator-free ones)
docker compose -f docker/compose.yaml run --rm sim -m pytest packages/gaitnet-core/tests packages/gaitnet-sim/tests

# a shell
docker compose -f docker/compose.yaml run --rm --entrypoint bash sim
```

Runs are written to `logs/rsl_rl/<experiment>/<timestamp>` in the checkout and tracked in
MLflow. Files the container writes to the checkout belong to uid 1000, the image's
`isaaclab` user.

## MLflow

Service `mlflow` (`ghcr.io/mlflow/mlflow:v3.16.1`) starts with any `sim` run and keeps
running; the UI is at http://localhost:5000. Its database and artifacts (checkpoints, each
run's `params/`, exported bundles) live in the `gaitnet_mlflow-data` volume. The sim image's
MLflow client replaces the Isaac Lab venv's protobuf 7.36.0rc1 with 6.33.6, see
`Dockerfile.sim`.

```bash
docker compose -f docker/compose.yaml up -d mlflow   # start it on its own
docker compose -f docker/compose.yaml stop mlflow
```

Kit, shader, asset and warp caches live in named volumes (`gaitnet_kit-cache`, ...), so only
the first run pays for shader compilation and asset downloads.

## Host requirements

An NVIDIA driver new enough for Isaac Sim 6.1 and the NVIDIA Container Toolkit. Tested with
driver 580.82 and toolkit 1.12 on an RTX 5070 Ti.
