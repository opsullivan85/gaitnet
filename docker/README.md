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
  dependencies or the MPC's C++ change; Python edits don't need a rebuild.

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
# scripted walk, the sim smoke test
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.walk --num_envs 4

# a shell
docker compose -f docker/compose.yaml run --rm --entrypoint bash sim
```

Files the container writes to the checkout belong to uid 1000, the image's `isaaclab` user.

Kit, shader, asset and warp caches live in named volumes (`gaitnet_kit-cache`, ...), so only
the first run pays for shader compilation and asset downloads.

## Host requirements

An NVIDIA driver new enough for Isaac Sim 6.1 and the NVIDIA Container Toolkit. Tested with
driver 580.82 and toolkit 1.12 on an RTX 5070 Ti.
