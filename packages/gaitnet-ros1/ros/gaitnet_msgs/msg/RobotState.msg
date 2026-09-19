# The planner's view of the robot's body. See packages/gaitnet-ros1/README.md.
#
# Legs in the order FL, FR, RL, RR. Per-leg vectors are leg-major: FL x, FL y, FL z, FR x, ...
# Frames:
#   base: the trunk, x forward, y left, z up
#   yaw:  origin at the base, rotated by the base's yaw only, so z is true vertical

float32[12] foot_pos          # m, each foot relative to the base origin, yaw frame
float32[12] foot_vel          # m/s, each foot relative to the base, base frame (J(q) qdot)
float32[3] base_lin_vel       # m/s, base frame
float32[3] base_ang_vel       # rad/s, base frame
float32[3] projected_gravity  # unit gravity direction in the base frame, (0, 0, -1) when level
bool[4] contact               # measured foot contact
GaitTiming gait_timing
float32[3] command            # (vx m/s, vy m/s, yaw rate rad/s) the controller is tracking,
                              # base frame, including the planner's nudge
float32[3] base_command       # the operator's command, before the nudge
