# The planner's answer to one Observation.

Header header                 # stamp: when the planner sent it
time observation_stamp        # header.stamp of the Observation this answers, for latency
FootstepCommand[] footsteps   # swings to start now; empty for none
Nudge nudge                   # replaces the previous nudge
