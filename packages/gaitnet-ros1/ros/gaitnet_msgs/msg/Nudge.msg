# Feedback from the planner to the controller, outside the learned policy.

float32[3] command_delta    # added to the operator's (vx, vy, yaw rate) command, base frame
