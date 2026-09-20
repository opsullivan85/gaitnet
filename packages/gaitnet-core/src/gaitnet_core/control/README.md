# The batched low-level controller

`gaitnet_core.control` is the planner's low-level controller written to run every robot
at once on the GPU: a convex single-rigid-body MPC for the stance forces, a Bezier arc
for the swinging foot, and a Cartesian PD through the leg Jacobian for the torques. It
implements `gaitnet_core.interfaces.LowLevelController`, so it is a drop-in replacement
for `gaitnet_sim.controllers.PooledMpcController`, which runs the same controller one
robot per CPU worker.

It exists because rollout collection was bound by that CPU solve, and the bound got
worse with every environment added. Select it with `presets=gpu_mpc`; see
[packages/gaitnet-sim/README.md](../../../../gaitnet-sim/README.md).

```
compute_torques(joint state, base pose, base twist, velocity command) -> (N, 12) torques
  kinematics.py   where each foot is and how fast, and the Jacobian
  controller.py   orientation and a body height read off the stance feet
  controller.py   the planner's hip-frame targets -> touchdown points
  srbd.py         the state -> a dense QP in the contact forces      | every 5th
  admm.py         that QP, for the whole batch                       | step
  swing.py        the arc a swinging foot follows
  controller.py   Cartesian PD + the MPC's forces -> joint torques
```

## What it is a copy of

The behaviour is meant to match `gaitnet-mpc` exactly, not to improve on it. Policy
bundles, the reward shaping and every evaluation baseline in the repo were produced
against that controller, and the real robot runs its own controller in any case (see the
[ros1 README](../../../../gaitnet-ros1/README.md)), so a "better" stance controller here
buys nothing and silently invalidates the comparisons.

| Here | Copied from |
| --- | --- |
| `srbd.py` | `gaitnet-mpc/cpp/mpc_osqp.cc`, `ConvexMpc::ComputeContactForces` |
| `admm.py` | the algorithm that file calls into, OSQP, rewritten batched |
| `controller.py` | `ConvexMPCLocomotion.run` and `SpecifiedFootstepLocomotion` |
| `gait.py` | `Gait.CalculatedGait` |
| `swing.py` | `FootSwingTrajectory` |
| `kinematics.py` | `LegController.computeLegJacobianAndPosition` |
| `rotations.py` | `math_utils.orientation_tools` |
| `model.py` | `Quadruped(RobotType.GO1)` and `Parameters` |

Upstream attribution and licences are in
[gaitnet-mpc/THIRD_PARTY.md](../../../../gaitnet-mpc/THIRD_PARTY.md). The two papers the
formulation and the solver come from are cited at the top of `srbd.py` and `admm.py`.

## How close it is

`../../../tests/test_control_parity.py` drives both controllers through the same forty
states, covering three swings and a body that rolls, pitches and yaws while moving. With
the QP solved to convergence the two agree to **0.02 N m** of joint torque, against peaks
of 32 N m; what is left is the CPU path's own precision, since `orientation_tools`
computes in float16 and every rotation matrix over there is good to about three decimals.

At the shipped settings the solver stops early, and that costs accuracy in a transient
right after a foot changes state, where the warm start helps least:

| ADMM iterations | worst step | typical step | mean control step, 4096 envs |
| --- | --- | --- | --- |
| 60 | 2.50 N m (7.7%) | 0.13 N m | 29.5 ms |
| **80 (default)** | **0.20 N m (0.6%)** | **0.025 N m** | **36.7 ms** |
| 100 | 0.13 N m (0.4%) | 0.015 N m | 44.4 ms |
| 400 | 0.02 N m (0.1%) | 0.013 N m | — |

`solver_iterations` and `solver_rho_update_interval` were tuned together and should be
changed together: each step-size update re-inverts, so an interval that does not divide
the budget well wastes most of it. 80 iterations with an interval of 20 was the best
value found; 80/30 is three times worse for less time saved than the accuracy costs.

## What it costs

Measured on one RTX 5070 Ti and a Ryzen 7 7800X3D (16 threads), controller only, outside
Isaac Sim, float32, horizon 10, one MPC solve every five control steps. "Mean step" is
what a physics step costs on average, which is the number that scales a rollout.

| envs | CPU pool | batched GPU | speed-up | GPU peak |
| --- | --- | --- | --- | --- |
| 100 | 6.6 ms | 5.0 ms | 1.3x | 50 MiB |
| 400 | 23.6 ms | 5.9 ms | 4.0x | 169 MiB |
| 1000 | 57.3 ms | 11.2 ms | 5.1x | 411 MiB |
| 2048 | 117.6 ms | 20.2 ms | 5.8x | 833 MiB |
| 4096 | 222.9 ms | 36.8 ms | 6.1x | 1663 MiB |

The shape matters more than any single row. The CPU pool is close to linear in the
environment count once past the core count, so it never gets cheaper per robot; the GPU
controller amortises, and its non-MPC steps cost **1.45 ms whatever the batch size**.
Those steps are bound by kernel launches, not arithmetic - 277 launches with the GPU idle
about 80% of the time - which is why they do not shrink at 100 environments and why the
speed-up there is small. `torch.compile` was tried and bought nothing (it breaks the
graph on the update counter and the in-place state). Fusing the rotation and kinematics
chains by hand is the obvious next thing if small-batch runs matter.

Peak memory grows with the **square** of the horizon: the intermediate in
`srbd._forced_response` carries every (step, step) pair of input-to-state blocks. Halving
the horizon roughly quarters it.

## Where it deliberately differs

Four places, all covered by the tests:

1. **The QP solver.** A fixed budget of batched ADMM iterations with a warm start,
   rather than qpOASES solving an active-set problem to convergence. The iterate is
   returned whatever its residual, which `QpSolution` reports. Nothing branches on a
   robot's data, which is what lets one call cover the batch.
2. **Precision.** float32 throughout by default, against float16 rotations on the CPU
   side. `double_precision` exists for checking, and roughly doubles the solve cost.
3. **The MPC update phase** is shared by the batch instead of being counted per robot,
   so one solve covers everyone. Resetting a robot no longer shifts which control steps
   its MPC refreshes on. The gait clock stays per robot, because the planner observes it.
4. **`rotation_to_rpy`** reads the angles off the matrix instead of going through a
   quaternion. Same map, fewer operations.

Three upstream results that look like mistakes are reproduced rather than fixed:

- the last block row of the free-response matrix is left zero, so the final horizon step
  predicts no free response (`srbd.build`);
- the forced-response blocks carry one power of the dynamics too many (`_forced_response`);
- the body-height estimate applies a rotation the wrong way round (`_update_height`).

Together they leave the controller asking for about **22% more normal force than the
robot weighs** when it is standing still at the height it is tracking. That is not a
porting error - the CPU controller does the same thing to six figures, and
`test_a_standing_robot_pushes_down_evenly` pins the number. Correcting them would change
how the robot walks, so do it as its own change with its own before-and-after, not as
part of a port.
