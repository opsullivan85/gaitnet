# AGENTS.md

Greedy, CNN-based non-gaited footstep planning for dynamic quadruped locomotion. A policy
scores candidate footholds each tick and picks at most one step; a convex MPC turns that
footstep into torques. The same planner code runs in Isaac Lab and on a real robot.

## Layout

A uv workspace. Everything lives under `packages/`:

| Package | Python | What it is |
| --- | --- | --- |
| `gaitnet-core` | 3.11+ | The planner itself: sim/real contract, candidate samplers, networks, selection, bundles, deployment runtime. No simulator, no ROS. |
| `gaitnet-mpc` | 3.11+ | CPU convex MPC controller (vendored rl-mpc-locomotion) plus its process pool. Has a compiled extension. |
| `gaitnet-sim` | 3.12+ | Isaac Lab 3 environments, RSL-RL training, export, evaluation. |
| `gaitnet-ros1` | 3.11+ | The planner on a real robot, over rosbridge websockets. No ROS install needed. |

`gaitnet-core` is the hub: sim and ros1 both implement its `RobotInterface`
([interfaces.py](packages/gaitnet-core/src/gaitnet_core/interfaces.py)) and both drive its
`PlannerRuntime`. Put logic that both sides need in core, not in one of the adapters.

Read these before changing anything substantial — they are the real documentation:

- [packages/gaitnet-sim/README.md](packages/gaitnet-sim/README.md) — tasks, presets, cfg overrides.
- [packages/gaitnet-ros1/README.md](packages/gaitnet-ros1/README.md) — the robot ↔ planner message contract, frames, units.
- [docker/README.md](docker/README.md) — images, every run command, debugging, MLflow.

## Running things

There is no local interpreter for the sim; **everything Isaac-related runs in Docker**,
driven from the repo root. Full command list is in [docker/README.md](docker/README.md).

```bash
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.walk --num_envs 4
docker compose -f docker/compose.yaml run --rm sim -m pytest packages/gaitnet-core/tests packages/gaitnet-sim/tests packages/gaitnet-ros1/tests
```

`gaitnet-core` and `gaitnet-ros1` are pure Python and can also be tested on the host with
a CPU torch install — that is what CI does
([.github/workflows/core.yml](.github/workflows/core.yml)):

```bash
pytest -q packages/gaitnet-core/tests packages/gaitnet-ros1/tests
ruff check --select E9,F packages/gaitnet-core packages/gaitnet-ros1/src packages/gaitnet-ros1/tests
```

`gaitnet-mpc` tests need the compiled extension, so they run in the sim image.

CI only covers core and ros1. Nothing checks the sim package automatically, so run its
tests yourself when you touch it.

## Rules of the road

- **Ask before launching Isaac Sim.** Sim runs take minutes, hold a GPU, and a killed run
  can orphan MPC pool workers. Don't start one to "check" something; ask first, and clean
  up after.
- **Don't edit vendored code.** `packages/gaitnet-mpc/src/gaitnet_mpc/mpc/`,
  `packages/gaitnet-mpc/cpp/` and `packages/gaitnet-mpc/extern/` are third party
  ([THIRD_PARTY.md](packages/gaitnet-mpc/THIRD_PARTY.md)); `extern/eigen3` is a submodule.
  Changes there belong in `controller.py` or `pool.py` where possible.
- **Keep the contract in sync.** The foothold grid, robot spec and foothold rules appear in
  three places: `gaitnet_core.bundle` (saved with a policy),
  `gaitnet_sim.env.contract` (what the env agrees to) and the ray-cast scanners in
  `gaitnet_sim.env.scene`. Bundle loading validates the manifest against the code, so a
  change to features, grid or network arguments invalidates existing bundles — bump
  `FORMAT_VERSION` in [bundle.py](packages/gaitnet-core/src/gaitnet_core/bundle.py) when
  the format itself changes.
- **Leg order is FL, FR, RL, RR, everywhere.** Per-leg vectors are flattened leg-major.
  Units are SI. Frames are base / yaw / hip-yaw, defined in the
  [ros1 README](packages/gaitnet-ros1/README.md#conventions).
- **Rebuild the sim image** when `gaitnet-mpc` changes at all (it is installed
  non-editable) or when any package's dependencies change. Edits to core and sim are
  picked up without a rebuild.
- **Measure performance changes.** Rollouts are dominated by the CPU MPC, not the GPU
  policy. If a change is meant to make something faster, benchmark before and after
  against a frozen baseline rather than reasoning about it.

## Style

- Ruff, line length 100, target py311 (config in [pyproject.toml](pyproject.toml)).
- `from __future__ import annotations` at the top of every module; modern type hints
  (`X | None`, builtin generics).
- Every module opens with a docstring saying what the thing is and where it sits relative
  to the rest — not a summary of its functions. Match that.
- Dataclass fields are documented with a docstring under the field giving shape, dtype,
  frame and units: `"""(N, 3) foothold in the leg's hip yaw frame (m), z is the terrain
  height there."""`. Keep this up for anything batched.
- Everything is batched over robots, first dimension N; the real robot is N = 1.
- Comments explain why, and are sparse. The code is expected to read clearly on its own.

## Not in git

`data/`, `training/`, `logs/` and `.vscode/` are ignored — policy bundles, evaluation CSVs
and run outputs are local only. Training runs land in `logs/rsl_rl/<experiment>/<timestamp>`
and in MLflow (http://localhost:5000). Files the container writes are owned by uid 1000.
