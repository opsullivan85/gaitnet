# gaitnet-ros1

GaitNet on a real robot. The footstep planner runs off-board on an x86 machine and talks to
the robot through a [rosbridge](http://wiki.ros.org/rosbridge_suite) websocket server, so the
planner machine needs no ROS install. The robot runs its own controller, which executes the
planner's footsteps.

```
robot (ROS 1)                                              planner machine
 controller, state estimator, elevation map                 gaitnet_ros1.run
   publishes  /gaitnet/observation   ── rosbridge ──►       Ros1Robot → PlannerRuntime
   executes   /gaitnet/command       ◄── (websocket) ──     (bundle: policy + contract)
```

This page is the contract between the two sides: what the robot publishes, what the
planner sends back, and in which frames and units. The message definitions are in
[ros/gaitnet_msgs/msg](ros/gaitnet_msgs/msg).

## Topics

| Topic | Type | Direction | Rate |
| --- | --- | --- | --- |
| `/gaitnet/observation` | `gaitnet_msgs/Observation` | robot → planner | the planning rate, 25 Hz for current policies |
| `/gaitnet/command` | `gaitnet_msgs/PlannerCommand` | planner → robot | one per observation |

The robot sets the pace. The planner answers each observation as soon as it arrives. If it
falls behind, it plans on the newest one and skips the rest. Policies are trained at 25 Hz,
so publish at 25 Hz.

## Conventions

- **Leg order**: FL, FR, RL, RR everywhere, including arrays, terrain patches and
  `FootstepCommand.leg` (0 to 3).
- **Units**: SI: m, s, m/s, rad/s.
- **Frames**:
  - *base*: the trunk, x forward, y left, z up.
  - *yaw*: origin at the base, rotated by the base's yaw only, so its z is true vertical.
    Roll and pitch don't tilt it.
  - *hip yaw*: the yaw frame moved to a leg's hip (abduction) joint. Terrain patches and
    footstep targets are in this frame, one per leg.
- Per-leg vectors are flattened leg-major: FL x, FL y, FL z, FR x, and so on.

## Observation

`header.stamp` is when the state was measured. The planner echoes it in its answer.

### `state` (`gaitnet_msgs/RobotState`)

| Field | Meaning |
| --- | --- |
| `foot_pos[12]` | each foot relative to the base origin, in the yaw frame (m), from leg kinematics |
| `foot_vel[12]` | each foot's velocity relative to the base, in the base frame (m/s): J(q) q̇, so a planted foot reads −(v_base + ω × r_foot) |
| `base_lin_vel[3]` | base linear velocity, base frame (m/s), from the state estimator |
| `base_ang_vel[3]` | base angular velocity, base frame (rad/s), from the IMU |
| `projected_gravity[3]` | unit gravity direction in the base frame; (0, 0, −1) when level |
| `contact[4]` | measured foot contact |
| `gait_timing` | the controller's *schedule*, see below |
| `command[3]` | the (vx, vy, yaw rate) the controller is tracking, base frame, **including** the planner's nudge |
| `base_command[3]` | the operator's command, **before** the nudge |

### Gait timing

Per leg, from the controller's footstep schedule. These are **scheduled, not measured**
values:

- `swing_phase`: progress through the current swing, 0 to 1, and 0 in stance.
- `swing_remaining`: seconds until the scheduled touchdown, and 0 in stance.
- `time_since_touchdown`: seconds since the scheduled touchdown, and 0 in swing.

A foot that touches down early still reads as swinging until its scheduled touchdown. The
planner decides which legs may take a new step from `swing_remaining`, so it must be
exact. A leg reported in stance while the controller still swings it could be given a
second, overlapping footstep.

### `terrain` (`gaitnet_msgs/TerrainPatch`)

These are terrain heights on a grid of cells near each hip, one patch per leg, each in its
hip's yaw frame. The patch is centred at (`center_x`, `center_y`) from a left hip and at
(`center_x`, −`center_y`) from a right hip, so positive `center_y` is outboard on every leg.
With s = +1 for FL and RL and −1 for FR and RR, for cell (i, j):

- position: x = `center_x` + (i − (size_x − 1) / 2) · resolution,
  y = s · `center_y` + (j − (size_y − 1) / 2) · resolution
- value: `heights[(leg · size_x + i) · size_y + j]` is the terrain's height there relative to
  the hip, in m and negative below it (about −0.26 on flat ground at the nominal stance)
- a cell with no data (unseen, or no return) is `TerrainPatch.UNKNOWN` (−1000). NaN and inf
  can't cross rosbridge's JSON.

The size, resolution and centre are the policy's, and the planner refuses anything else.
The defaults are a 25 × 25 grid of 1.5 cm cells plus a 3-cell border of context, so
31 × 31 at 0.015 m, centred at (0, 0.08) m; bundles trained before the centre existed are
centred on the hip, (0, 0), which is also what a robot that leaves the centre fields out is
taken to send. Sample the elevation map at the cell centres. Don't pre-filter the patch for
steppability: the planner applies its own reach and edge rules.

## PlannerCommand

| Field | Meaning |
| --- | --- |
| `header.stamp` | when the planner sent it |
| `observation_stamp` | the `header.stamp` of the observation it answers |
| `footsteps[]` | swings to start **now**, often none |
| `nudge.command_delta[3]` | added to the operator's command; **replaces** the previous nudge |

Each `FootstepCommand` fields:

- `leg`: which leg to swing.
- `target[3]`: the foothold, relative to that leg's hip, in the hip yaw frame **of the
  observation it answers**. Its z is the terrain **surface** there, where the sole touches.
  Add your foot's radius to get the foot centre (the simulator adds 0.02 m for the Go1).
  It names a spot on the ground, not an offset to keep: fix it in your odometry frame
  using the hip position and heading at `observation_stamp`, and swing to that point
  wherever the body has moved by touchdown. The simulated controllers, which the policies
  are trained against, do exactly this; re-applying the offset to the hip later lands the
  foot off by however far the body travelled, turned or tilted in between.
- `duration`: seconds from lift-off to touchdown, between 0.1 and 0.3 for current policies.

What the planner guarantees:
- A footstep's leg is in scheduled stance.
- At least two legs stay in scheduled stance after it lifts off.
- The target lies within the leg's terrain patch, on a cell that passed the planner's reach
  and edge rules.

The controller decides everything else: swing trajectory (the simulated controller clears
the higher end of the swing by a third of the body height), stance forces, and body height.

The nudge comes from feedback observers, such as slowing down when the policy keeps
declining to step. Apply it as `command = base_command + command_delta` and report both in
the next observation.

## Timing and failures

- **Latency**: the robot can measure it as now minus `observation_stamp` when a command
  arrives. The robot should ignore commands answering an observation older than it
  tolerates, for example 100 ms. Keep that much history of the base pose in the odometry
  frame, so a target can be pinned from the pose at `observation_stamp` rather than the
  pose when the command arrives: at 0.2 m/s, 50 ms of latency is 1 cm.
- **Stale observations**: the planner stops, with an error, when no new observation
  arrives within `--timeout` (0.5 s by default). It never re-plans on old data.
- **Missing commands**: the robot must stay safe when commands stop arriving at any time,
  whether from a planner crash, a network drop or a stopped operator. What it does then
  (for example, finish the current swings and stand) is the robot side's choice.

## Robot side: building the messages

Copy or link `ros/gaitnet_msgs` into the robot's catkin workspace and build it. The
messages depend only on `std_msgs`. Then run a rosbridge server next to the controller:

```bash
roslaunch rosbridge_server rosbridge_websocket.launch   # port 9090
```

## Planner side

From the repository root, with bundles in `data/bundles` (`gaitnet_sim.scripts.export_bundle`):

```bash
docker compose -f docker/compose.yaml build deploy
docker compose -f docker/compose.yaml run --rm deploy --bundle /bundles/policy.pt --host <robot>
```

Or without Docker: `pip install packages/gaitnet-core packages/gaitnet-ros1` (plus torch),
then `python -m gaitnet_ros1.run --bundle policy.pt --host <robot>`.

Options: `--refine` refines each footstep by gradient ascent on the policy's score,
`--no_observers` drops the bundle's feedback observers, and `--sampler` / `--per_leg`
change the candidates (dense by default). The run log reports planning time per tick and
skipped observations.

## Testing without a robot

`docker/ros_roundtrip.sh data/bundles/<bundle>.pt` runs the planner container against a
fake robot. The fake robot is a ROS Noetic container with rosbridge that publishes flat
ground at 25 Hz and follows a toy gait schedule. It reports how many observations were
answered, the end-to-end latency, and whether any footstep broke the contract.
