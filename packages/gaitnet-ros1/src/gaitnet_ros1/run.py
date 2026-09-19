"""Run a policy bundle on a robot over ROS 1.

    python -m gaitnet_ros1.run --bundle policy.pt --host <rosbridge host>
    python -m gaitnet_ros1.run --bundle policy.pt --host robot.local --refine --max_ticks 500

Connects to the robot's rosbridge server, then answers each gaitnet_msgs/Observation with a
gaitnet_msgs/PlannerCommand as soon as it arrives (the robot sets the rate), through the same
`PlannerRuntime` as the simulated evaluation: dense candidates, deterministic selection and
the bundle's feedback observers unless told otherwise. Stops with an error when the robot
stops publishing for `--timeout` s.
"""

from __future__ import annotations

import argparse
import logging
import statistics

import torch

from gaitnet_core.bundle import load_bundle
from gaitnet_core.refine import Refiner
from gaitnet_core.runtime import PlannerRuntime
from gaitnet_core.samplers import make_sampler
from gaitnet_ros1.robot import Ros1Robot, StaleObservation
from gaitnet_ros1.transport import RosbridgeTransport

logger = logging.getLogger("gaitnet_ros1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", required=True, help="Policy bundle file (gaitnet_sim.scripts.export_bundle).")
    parser.add_argument("--host", default="localhost", help="rosbridge websocket server.")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--observation_topic", default="/gaitnet/observation")
    parser.add_argument("--command_topic", default="/gaitnet/command")
    parser.add_argument("--timeout", type=float, default=0.5, help="Longest wait for an observation (s).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampler", default="dense", help="Candidate sampler (gaitnet_core.samplers.SAMPLERS).")
    parser.add_argument("--per_leg", type=int, default=None, help="Candidates per leg, for the sampling samplers.")
    parser.add_argument("--refine", action="store_true", help="Refine each footstep by gradient ascent on the score.")
    parser.add_argument("--stochastic", action="store_true", help="Sample footsteps instead of the deterministic choice.")
    parser.add_argument("--no_observers", action="store_true", help="Leave out the bundle's feedback observers.")
    parser.add_argument("--max_ticks", type=int, default=None, help="Stop after this many planning ticks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    torch.set_grad_enabled(False)

    bundle = load_bundle(args.bundle, map_location=args.device)
    sampler = make_sampler(args.sampler, **({"per_leg": args.per_leg} if args.per_leg is not None else {}))
    planner = bundle.planner(sampler)
    observers = [] if args.no_observers else bundle.make_observers()
    logger.info(
        f"bundle {args.bundle}: {type(bundle.actor).__name__}, trained {bundle.extra.get('mlflow_run_id', '?')}"
        f" at {bundle.extra.get('git_commit', '?')}; observers {bundle.observers if observers else 'none'};"
        f" sampler {args.sampler}; refine {args.refine}; stochastic {args.stochastic}"
    )

    robot = Ros1Robot(
        RosbridgeTransport(args.host, args.port),
        bundle.robot,
        bundle.grid,
        observation_topic=args.observation_topic,
        command_topic=args.command_topic,
        timeout=args.timeout,
        device=args.device,
    )
    runtime = PlannerRuntime(
        robot,
        planner,
        observers=observers,
        rate_hz=None,
        deterministic=not args.stochastic,
        postprocess=Refiner(planner) if args.refine else None,
    )
    logger.info(f"connected to {args.host}:{args.port}, planning on {args.observation_topic}")
    code = 0
    try:
        runtime.run(max_ticks=args.max_ticks)
    except StaleObservation as error:
        logger.error(f"stopping: {error}")
        code = 1
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()

    durations = sorted(runtime.plan_durations)
    if durations:
        p95 = durations[min(len(durations) - 1, int(0.95 * len(durations)))]
        logger.info(
            f"{len(durations)} ticks planned: median {statistics.median(durations) * 1e3:.1f} ms,"
            f" 95% {p95 * 1e3:.1f} ms, max {durations[-1] * 1e3:.1f} ms; {robot.skipped} observations skipped"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
