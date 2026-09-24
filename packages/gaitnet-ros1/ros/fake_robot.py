#!/usr/bin/env python3
"""A stand-in robot for testing the planner over rosbridge (ROS Noetic, Python 3.8).

Publishes gaitnet_msgs/Observation on flat ground at a fixed rate, keeps a toy gait schedule
from the footsteps it is sent, and measures the planner's end-to-end latency: from an
observation's stamp to the PlannerCommand answering it arriving back here.

    python3 fake_robot.py --answers 500 --timeout 120

Prints one JSON summary line. Exits 0 if the planner answered at least --min_answered of the
observations published after its first answer and sent no invalid footstep.
"""

import argparse
import json
import math
import statistics
import sys

import rospy
from gaitnet_msgs.msg import Observation, PlannerCommand, TerrainPatch

# feet at the nominal stance, relative to the base in its yaw frame (m), FL, FR, RL, RR
NOMINAL_FEET = [(0.19, 0.13, -0.26), (0.19, -0.13, -0.26), (-0.19, 0.13, -0.26), (-0.19, -0.13, -0.26)]
GROUND = -0.26  # m below each hip
COMMAND = (0.1, 0.0, 0.0)


class FakeRobot:
    def __init__(self, args):
        self.args = args
        now = rospy.get_time()
        self.swing_start = [now] * 4
        self.swing_end = [now] * 4
        self.published = 0
        self.published_at_first_answer = None
        self.published_at_last_answer = 0
        self.last_answer = None
        self.latencies = []
        self.footsteps = [0] * 4
        self.invalid = []
        self.heights = self._flat_patch()
        self.publisher = rospy.Publisher("/gaitnet/observation", Observation, queue_size=1)
        rospy.Subscriber("/gaitnet/command", PlannerCommand, self.on_command, queue_size=10)

    def _flat_patch(self):
        """Flat ground, with the outermost ring of each patch unknown."""
        n = self.args.patch
        heights = []
        for _leg in range(4):
            for i in range(n):
                for j in range(n):
                    edge = i in (0, n - 1) or j in (0, n - 1)
                    heights.append(TerrainPatch.UNKNOWN if edge else GROUND)
        return heights

    def on_patch(self, leg, target, half):
        """Whether `target` is within `half` of the leg's patch centre (right legs mirror y)."""
        side = 1.0 if leg in (0, 2) else -1.0
        return abs(target[0] - self.args.center_x) <= half and abs(target[1] - side * self.args.center_y) <= half

    def swinging(self, leg, now):
        return now < self.swing_end[leg]

    def on_command(self, msg):
        now = rospy.get_time()
        if self.published_at_first_answer is None:
            self.published_at_first_answer = self.published
        self.last_answer = now
        self.published_at_last_answer = self.published
        self.latencies.append(now - msg.observation_stamp.to_sec())
        half = (self.args.patch - 1) / 2 * self.args.resolution
        for step in msg.footsteps:
            leg, target = step.leg, list(step.target)
            problem = None
            if not 0 <= leg < 4:
                problem = "leg %d" % leg
            elif not all(math.isfinite(v) for v in target) or not self.on_patch(leg, target, half):
                problem = "target %s" % target
            elif not 0.05 < step.duration < 1.0:
                problem = "duration %.3f" % step.duration
            elif self.swinging(leg, now):
                problem = "leg %d is still swinging" % leg
            if problem:
                self.invalid.append(problem)
                continue
            self.swing_start[leg] = now
            self.swing_end[leg] = now + step.duration
            self.footsteps[leg] += 1

    def observation(self, now):
        msg = Observation()
        msg.header.stamp = rospy.Time.from_sec(now)
        state = msg.state
        state.foot_pos = [v for foot in NOMINAL_FEET for v in foot]
        state.foot_vel = [0.0] * 12
        state.base_lin_vel = list(COMMAND)
        state.base_ang_vel = [0.0] * 3
        state.projected_gravity = [0.0, 0.0, -1.0]
        state.contact = [not self.swinging(leg, now) for leg in range(4)]
        timing = state.gait_timing
        timing.swing_phase, timing.swing_remaining, timing.time_since_touchdown = [], [], []
        for leg in range(4):
            if self.swinging(leg, now):
                duration = self.swing_end[leg] - self.swing_start[leg]
                timing.swing_phase.append((now - self.swing_start[leg]) / duration)
                timing.swing_remaining.append(self.swing_end[leg] - now)
                timing.time_since_touchdown.append(0.0)
            else:
                timing.swing_phase.append(0.0)
                timing.swing_remaining.append(0.0)
                timing.time_since_touchdown.append(now - self.swing_end[leg])
        state.command = list(COMMAND)
        state.base_command = list(COMMAND)
        msg.terrain.resolution = self.args.resolution
        msg.terrain.size_x = msg.terrain.size_y = self.args.patch
        msg.terrain.center_x = self.args.center_x
        msg.terrain.center_y = self.args.center_y
        msg.terrain.heights = self.heights
        return msg

    def run(self):
        rate = rospy.Rate(self.args.rate)
        start = rospy.get_time()
        while not rospy.is_shutdown() and len(self.latencies) < self.args.answers:
            now = rospy.get_time()
            # the planner stopped (or never started)
            if now - start > self.args.timeout or (self.last_answer is not None and now - self.last_answer > 2.0):
                break
            self.publisher.publish(self.observation(rospy.get_time()))
            self.published += 1
            rate.sleep()

    def summary(self):
        latencies = sorted(self.latencies)
        # observations from the one the first answer came for to the one the last came for
        first = self.published_at_first_answer or self.published
        expected = self.published_at_last_answer - first + 1
        answered = len(latencies) / max(1, expected)
        result = {
            "observations_since_first_answer": expected,
            "answered": len(latencies),
            "answered_fraction": round(answered, 3),
            "footsteps_per_leg": self.footsteps,
            "invalid_footsteps": self.invalid[:5],
        }
        if latencies:
            result["latency_ms"] = {
                "median": round(statistics.median(latencies) * 1e3, 1),
                "p95": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] * 1e3, 1),
                "max": round(latencies[-1] * 1e3, 1),
            }
        ok = answered >= self.args.min_answered and not self.invalid and len(latencies) > 0
        return result, ok


def main():
    parser = argparse.ArgumentParser(description="fake robot for the GaitNet planner")
    parser.add_argument("--rate", type=float, default=25.0, help="Observations per second.")
    parser.add_argument("--answers", type=int, default=500, help="Stop after this many answers.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Stop after this long (s).")
    parser.add_argument("--min_answered", type=float, default=0.9)
    parser.add_argument("--patch", type=int, default=31, help="Terrain patch cells per side, border included.")
    parser.add_argument("--resolution", type=float, default=0.015)
    parser.add_argument("--center_x", type=float, default=0.0, help="Patch centre ahead of each hip (m).")
    parser.add_argument(
        "--center_y", type=float, default=0.08, help="Patch centre outboard of each hip (m); 0 for older bundles."
    )
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("gaitnet_fake_robot")
    robot = FakeRobot(args)
    robot.run()
    result, ok = robot.summary()
    print(json.dumps(result), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
