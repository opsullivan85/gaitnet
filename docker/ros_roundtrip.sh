#!/usr/bin/env bash
# The deployed planner against a fake robot over rosbridge, end to end: builds the ROS and
# deploy images, starts rosbridge, then runs the fake robot and the planner (500 ticks, 20 s
# at the robot's 25 Hz). The fake robot prints a JSON summary: how many observations the
# planner answered, the end-to-end latency, footsteps per leg. Exits non-zero if either side
# fails.
#
#   docker/ros_roundtrip.sh data/bundles/policy.pt [planner arguments, e.g. --refine]
set -euo pipefail
cd "$(dirname "$0")/.."

bundle=${1:?usage: docker/ros_roundtrip.sh <bundle in data/bundles> [planner arguments]}
shift
compose=(docker compose -f docker/compose.yaml)
in_ros=("${compose[@]}" exec -T ros bash -c)

"${compose[@]}" build ros deploy
"${compose[@]}" up -d ros
trap '"${compose[@]}" stop ros >/dev/null 2>&1; "${compose[@]}" rm -f ros >/dev/null 2>&1' EXIT

for _ in $(seq 60); do
    if "${in_ros[@]}" "source /catkin_ws/devel/setup.bash && rosnode list 2>/dev/null | grep -q rosbridge"; then
        break
    fi
    sleep 1
done

"${in_ros[@]}" "source /catkin_ws/devel/setup.bash && python3 /catkin_ws/fake_robot.py --answers 500 --timeout 120" &
robot=$!
"${compose[@]}" run --rm -T deploy --bundle "/bundles/$(basename "$bundle")" --host ros --max_ticks 500 "$@"
wait "$robot"
