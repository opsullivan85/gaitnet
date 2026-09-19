"""Evaluate a policy bundle across terrain difficulties and commanded forward velocities.

    python -m gaitnet_sim.scripts.eval_sweep --bundle bundle.pt
    python -m gaitnet_sim.scripts.eval_sweep --bundle bundle.pt --task GaitNet-Pillars --refine
    python -m gaitnet_sim.scripts.eval_sweep --bundle bundle.pt --difficulties 0 0.2 --velocities 0.1 \\
        --envs_per_difficulty 8

One scene holds every difficulty (one terrain row each), and velocities are swept in place,
so there is one simulator boot, terrain cook and controller pool for the whole sweep. The
policy runs through the same `PlannerRuntime` as on hardware, with dense candidates,
deterministic selection and the bundle's feedback observers, on nominal dynamics with exact
observations, unless told otherwise; `--refine` adds gradient refinement of each footstep
and `--randomize` keeps training's randomization. Writes one CSV row per robot and trial:
difficulty, velocity, trial, env, distance (m walked along +x before the robot's first
episode ended), steps, truncated, terminated_by.

Trailing key=value arguments are Hydra overrides of the env cfg.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--bundle", required=True, help="Policy bundle file (gaitnet_sim.scripts.export_bundle).")
parser.add_argument("--task", default="GaitNet-Holes", help="Task whose terrain type to sweep.")
parser.add_argument(
    "--difficulties", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
)
parser.add_argument("--velocities", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
parser.add_argument("--envs_per_difficulty", type=int, default=50)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--terrain_length", type=float, default=None, help="Sub-terrain length (m); sized to the episode by default.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Episode length (s); the env's by default.")
parser.add_argument("--sampler", default="dense", help="Candidate sampler (gaitnet_core.samplers.SAMPLERS).")
parser.add_argument("--per_leg", type=int, default=None, help="Candidates per leg, for the sampling samplers.")
parser.add_argument("--stochastic", action="store_true", help="Sample footsteps instead of the deterministic choice.")
parser.add_argument("--refine", action="store_true", help="Refine each footstep by gradient ascent on the network's score.")
parser.add_argument("--refine_steps", type=int, default=4, help="Ascent steps per footstep, with --refine.")
parser.add_argument("--no_observers", action="store_true", help="Leave out the feedback observers the bundle was trained with.")
parser.add_argument(
    "--randomize", action="store_true", help="Keep training's friction, mass and push randomization and observation noise."
)
parser.add_argument("--out", default=None, help="CSV path; data/evaluations/<bundle>_<time>.csv by default.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import csv  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from gaitnet_core.bundle import load_bundle  # noqa: E402
from gaitnet_core.refine import Refiner  # noqa: E402
from gaitnet_core.runtime import PlannerRuntime  # noqa: E402
from gaitnet_core.samplers import make_sampler  # noqa: E402
from gaitnet_sim.eval.env_cfg import make_eval_env_cfg  # noqa: E402
from gaitnet_sim.eval.evaluator import Evaluator  # noqa: E402
from gaitnet_sim.isaac_robot import IsaacRobot  # noqa: E402
from gaitnet_sim.tasks import register  # noqa: E402
from gaitnet_sim.terrains import envs_for_difficulty  # noqa: E402

# Isaac Lab configures the root logger (so basicConfig would do nothing); log on our own
logger = logging.getLogger("eval_sweep")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [eval] %(message)s"))
logger.addHandler(_handler)
logging.getLogger("gaitnet_sim").addHandler(_handler)
logging.getLogger("gaitnet_sim").setLevel(logging.INFO)

COLUMNS = ["difficulty", "velocity", "trial", "env", "distance", "steps", "truncated", "terminated_by"]


def summarize(rows: list[dict], difficulty: float, velocity: float) -> str:
    distances = [row["distance"] for row in rows]
    reasons = Counter(row["terminated_by"] for row in rows)
    ended = ", ".join(f"{name}={count}" for name, count in reasons.most_common())
    return (
        f"d={difficulty:g} v={velocity:g}: distance mean {sum(distances) / len(distances):.2f} m,"
        f" max {max(distances):.2f} m; ended by {ended}"
    )


def main() -> int:
    register()
    device = args_cli.device or "cuda:0"
    bundle = load_bundle(args_cli.bundle, map_location=device)
    logger.info(f"bundle {args_cli.bundle}: {bundle.extra}")

    env_cfg = parse_env_cfg(args_cli.task, device=device, overrides=hydra_overrides)
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    make_eval_env_cfg(
        env_cfg,
        bundle,
        difficulties=args_cli.difficulties,
        velocities=args_cli.velocities,
        envs_per_difficulty=args_cli.envs_per_difficulty,
        terrain_length=args_cli.terrain_length,
        randomize=args_cli.randomize,
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = IsaacRobot(env)
    sampler_kwargs = {"per_leg": args_cli.per_leg} if args_cli.per_leg is not None else {}
    planner = bundle.planner(make_sampler(args_cli.sampler, **sampler_kwargs))
    observers = [] if args_cli.no_observers else bundle.make_observers()
    logger.info(f"observers: {bundle.observers if observers else 'none'}; refine: {args_cli.refine}")
    runtime = PlannerRuntime(
        robot,
        planner,
        observers=observers,
        deterministic=not args_cli.stochastic,
        postprocess=Refiner(planner, steps=args_cli.refine_steps) if args_cli.refine else None,
    )
    evaluator = Evaluator(env)
    command_term = env.command_manager.get_term("base_velocity")

    stem = f"{Path(args_cli.bundle).stem}_{args_cli.task}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args_cli.out or f"data/evaluations/{stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for velocity in args_cli.velocities:
            command_term.set_command((velocity, 0.0, 0.0))
            for trial in range(args_cli.trials):
                env.reset()
                runtime.reset()
                evaluator.start()
                start, ticks = time.monotonic(), 0
                with torch.inference_mode():
                    while simulation_app.is_running():
                        runtime.step()
                        ticks += 1
                        if evaluator.record(robot.last_step):
                            break
                if not simulation_app.is_running():
                    logger.error("the simulator stopped mid-trial")
                    return 1
                elapsed = time.monotonic() - start
                logger.info(f"v={velocity:g} trial {trial}: {ticks} ticks in {elapsed:.1f} s ({ticks / elapsed:.1f} ticks/s)")
                rows = evaluator.rows()
                for index, difficulty in enumerate(args_cli.difficulties):
                    group = rows[envs_for_difficulty(index, args_cli.envs_per_difficulty)]
                    logger.info(summarize(group, difficulty, velocity))
                    for row in group:
                        writer.writerow({"difficulty": difficulty, "velocity": velocity, "trial": trial, **row})
                f.flush()

    robot.term.controller.close()
    env.close()
    logger.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
