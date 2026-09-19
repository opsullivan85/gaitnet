# The low-level controller's *scheduled* gait, per leg in the order FL, FR, RL, RR.
#
# Scheduled, not measured: a foot that touches down early still reads as swinging until
# its scheduled touchdown. The planner relies on this to decide which legs may step.

float32[4] swing_phase           # progress through the current swing, 0..1; 0 in stance
float32[4] swing_remaining       # s until the scheduled touchdown; 0 in stance
float32[4] time_since_touchdown  # s since the scheduled touchdown; 0 in swing
