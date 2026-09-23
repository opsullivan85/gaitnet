# gaitnet-sim

GaitNet's Isaac Lab 3 environments, RSL-RL training glue, policy export and evaluation.
How to build and run the container is in [docker/README.md](../../docker/README.md); this
page covers what to change for an experiment.

## Tasks

| Task | Terrain | Difficulty |
| --- | --- | --- |
| `GaitNet-Holes` | flat ground with random holes | fraction of holes |
| `GaitNet-Pillars` | square pillars at random heights over a void | gap width and height spread |

Both train on a curriculum over difficulty rows and share everything but the terrain.

## Presets

Variants are Isaac Lab presets. `presets=<name>[,<name>...]` switches every cfg field
that has an alternative of that name, in the env and the agent cfg together, so a
variant's network and the observation group it reads can't get out of step. Presets
compose.

| Preset | Env | Agent | Update cost* |
| --- | --- | --- | --- |
| *(none)* | groups `state`, `candidates` | `CandidateScorer`, candidate features `xyz` | 0.44 s |
| `spatial` | + group `terrain` | `DenseSpatialCNN` (D1): a CNN over each leg's height patch, candidates sample its score map | 1.86 s |
| `crop` | + group `terrain` | `CandidateScorer` with `xyz_crop`: each candidate also sees the 5 x 5 cells of terrain around it | 0.56 s |
| `privileged` | + group `privileged` (coarse terrain and foothold validity per leg, base clearance, contact forces) | the critic reads `state` + `privileged` | ~0 |
| `slowdown` | + group `base_command` | the actor runs the `step_confidence_slowdown` observer while acting | ~0 |
| `redirect` | + group `base_command`, `footholds` | the actor runs the `blocked_leg_redirect` observer while acting | ~0 |
| `slowdown_redirect` | + group `base_command`, `footholds` | both observers, nudges summed | ~0 |
| `swing_duration_ablation` | — | `CandidateScorer` with `fixed_duration=0.25` s: no duration head, the policy is the footstep choice alone | ~0 |
| `gpu_mpc` | the low-level controller runs batched on the GPU instead of in a CPU process pool | — | ~0 |

\* Forward and backward of the actor on one PPO minibatch (64000 rows, 4 legs x 64
candidates) on an RTX 5070 Ti; PPO runs 32 per iteration. Without `gpu_mpc`, rollouts are
dominated by the CPU MPC either way.

`spatial` and `crop` both choose the actor network; if both are given, the first wins.

Each preset is written up in full — what it switches, what it costs, what to watch for — in
[ARCHITECTURE.md](../../ARCHITECTURE.md#3-presets), which also puts the two scoring networks
side by side. Keep the two in step when you add or change one.

```bash
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.train \
    --task GaitNet-Pillars --num_envs 1024 presets=spatial,privileged
```

The `terrain` group costs ~4 GB of rollout storage at 1024 envs x 250 steps, which is why
it is off unless a preset reads it.

### Low-level controller

The footstep action term owns a controller that turns the planner's footsteps into joint
torques. Two implement it, running the same convex MPC for the same robot:

| Cfg | Where it runs | Use it for |
| --- | --- | --- |
| `PooledMpcControllerCfg` *(default)* | `gaitnet_mpc`, one robot per CPU worker | the reference: every bundle and baseline in this repo was produced against it |
| `BatchedMpcControllerCfg` (`presets=gpu_mpc`) | `gaitnet_core.control`, the whole batch on the GPU | anything past a few hundred envs |

The CPU pool costs about 6.6 ms per physics step at 100 envs and 223 ms at 4096, roughly
linear once past the core count. The batched one costs 5.0 ms and 36.8 ms, so it is worth
1.3x at 100 envs and 6.1x at 4096, and it is what makes the large counts affordable at
all. It tracks the CPU controller's torques to well under a percent; the measurements,
the accuracy against solver budget, and the handful of deliberate differences are in
[gaitnet_core/control/README.md](../gaitnet-core/src/gaitnet_core/control/README.md).

```bash
docker compose -f docker/compose.yaml run --rm sim -m gaitnet_sim.scripts.train \
    --task GaitNet-Pillars --num_envs 4096 presets=gpu_mpc,privileged

# more solver iterations per MPC solve: closer to the CPU controller, slower
presets=gpu_mpc env.actions.footstep.controller.solver_iterations=100
```

Switching controllers is a sim2real-relevant change, not a pure speed-up: the two agree
closely but not exactly, so compare a policy trained under one against the other before
trusting a result that crosses them.

### Feedback observers in training

With `slowdown` (or `redirect`, `slowdown_redirect`), the actor scores the candidates as usual and hands the plan to the
observer, whose nudge (a delta on the velocity command) goes into the action vector. The
environment applies it, the policy observes the nudged command, and the tracking rewards
follow it (`env.rewards.xy_tracking.params.command=base` tracks the operator's command
instead). The nudge is part of the environment's dynamics: log-probabilities ignore it.
Observers only run while acting (gradients off), so PPO's update passes don't advance
their memory, and they are reset for envs whose episode ended. Exported bundles carry the
observers, and `eval_sweep` runs them unless given `--no_observers`.

## Overrides

Anything in the env or agent cfg can be overridden on the command line. Isaac Lab 3
applies `env.*` and `agent.*` overrides itself and parses values as Python literals, so
lists of names need quoted strings (and the whole argument quoted for the shell).

```bash
# the state vector (names from gaitnet_core.features.FEATURES)
"env.observations.state.robot_state.params.features=['foot_pos','base_lin_vel','command','gait_timing']"

# the training sampler (gaitnet_core.samplers.SAMPLERS) and candidates per leg
env.observations.candidates.candidates.params.sampler=uniform_lattice
env.observations.candidates.candidates.params.sampler_kwargs.per_leg=32

# foothold rules
env.gaitnet.min_stance_after_step=3 env.gaitnet.edge_margin=1

# rewards
env.rewards.step_taken.weight=-0.2 env.rewards.xy_tracking.params.command=base

# network sizes (keys of the selected network's constructor)
agent.actor.network.trunk_sizes=[256,256]
presets=spatial agent.actor.network.channels=[8,8,8]

# observer parameters
presets=slowdown agent.actor.observers.step_confidence_slowdown.patience=5

# PPO
agent.algorithm.entropy_coef=0.01 agent.algorithm.learning_rate=1e-4
```

A value that is itself a preset (`agent.actor.network`, `agent.actor.observers`,
`agent.obs_groups.critic`, the optional observation groups) can't be replaced whole with
an override, since Isaac Lab reads that as choosing a preset by name; override its keys,
or add a preset.

The foothold grid (`env.gaitnet.grid_*`) is also baked into the scanners' ray patterns
(built with the scene cfg) and into networks that read terrain
(`agent.actor.network.grid`), so changing it takes more than an override. `RobotIO`
refuses scanners whose ray count doesn't fit the grid, and export refuses a network built
for another grid, but a changed resolution alone would go unnoticed by the scanners.

`packages/gaitnet-sim/tests/test_presets.py` checks that these recipes resolve as
described.

## Adding a variant

- A state feature: an entry in `gaitnet_core.features.FEATURES`.
- A candidate encoding: an entry in `gaitnet_core.networks.CANDIDATE_FEATURES`.
- A network: a module in `gaitnet_core.networks.NETWORKS` taking `(state, candidates,
  terrain)` and returning `Scores`; set `uses_terrain` if it reads terrain.
- An observer: a class in `gaitnet_core.observers.OBSERVERS`.
- A sampler: an entry in `gaitnet_core.samplers.SAMPLERS`.

Then give it a preset: a field with the variant's name on the relevant `preset(...)` in
`env/env_cfg.py` and `rl/agent_cfg.py`, and a line in the table above.

## Sim2real hardening

Training runs are hardened by default:

- **Observation noise** (`env.actions.footstep.observation_noise`, see `env/noise.py`):
  uniform noise on the planner's whole view of the robot, drawn once per planning step. It
  covers foot positions and velocities, base velocities, gravity, and each leg's terrain
  patch (a patch offset plus small per-cell noise). The state vector, the candidate
  footholds and the terrain group therefore all see the same corrupted world, as on
  hardware. Contact, gait timing and commands stay exact. Terminations, rewards and
  privileged observations read the truth.
- **Dynamics**: foot friction in 0.5 to 1.25 static and 0.4 to 1.0 dynamic, trunk mass −1 to
  +2 kg (both per robot at startup), and velocity pushes of up to 0.2 m/s every 8 to 12 s.
  The MPC keeps its nominal model throughout.

`env.play_mode()` turns all of this off: nominal friction of 1.0, no added mass, no pushes,
no noise. `eval_sweep` and `walk` use it unless given `--randomize`. To train without it,
`env.actions.footstep.observation_noise=None env.events.push_robot=None ...` (or add a
preset). Deploying a bundle on a robot is in [gaitnet-ros1](../gaitnet-ros1/README.md).

## Evaluation

`gaitnet_sim.scripts.eval_sweep` runs a bundle through the deployment runtime
(`PlannerRuntime` + `IsaacRobot`) across difficulties and velocities. Options for the
experimental pieces: `--sampler` / `--per_leg` (the default is dense), `--refine` (gradient
refinement of each footstep on the network's score, `--refine_steps`), `--stochastic`,
`--no_observers`, and `--randomize` (training's randomization and noise instead of nominal).
