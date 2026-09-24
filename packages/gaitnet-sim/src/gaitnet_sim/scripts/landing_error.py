"""Measure how far feet land from the footholds a policy commands.

    python -m gaitnet_sim.scripts.landing_error --bundle bundle.pt --num_envs 16
    python -m gaitnet_sim.scripts.landing_error --bundle bundle.pt --difficulty 0.5 presets=gpu_mpc
    python -m gaitnet_sim.scripts.landing_error --bundle bundle.pt \\
        env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]

The policy runs through `PlannerRuntime` as in `play`, on training's terrain at one difficulty
without the curriculum, under training's random velocity commands, on nominal dynamics with
exact observations unless `--randomize` is given. `gaitnet_sim.eval.landing.LandingProbe`
wraps the controller and follows every footstep at the control rate; the contact sensor is
switched to update every physics step for it (the planner reads it on planning ticks, where
the two agree). One row per footstep goes to `--out`, and a summary is printed.

Trailing key=value arguments are Hydra overrides of the env cfg.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--bundle", required=True, help="Policy bundle file (gaitnet_sim.scripts.export_bundle).")
parser.add_argument("--task", default="GaitNet-Holes")
parser.add_argument("--difficulty", type=float, default=0.0, help="Terrain difficulty of every sub-terrain.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=750, help="Planning ticks (25 Hz).")
parser.add_argument("--sampler", default="dense", help="Candidate sampler (gaitnet_core.samplers.SAMPLERS).")
parser.add_argument("--stochastic", action="store_true", help="Sample footsteps instead of the deterministic choice.")
parser.add_argument("--no_observers", action="store_true", help="Leave out the feedback observers the bundle was trained with.")
parser.add_argument(
    "--randomize", action="store_true", help="Keep training's friction, mass and push randomization and observation noise."
)
parser.add_argument("--out", default=None, help="CSV path; default logs/landing/<bundle>_<task>_<difficulty>.csv.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from gaitnet_core.bundle import load_bundle  # noqa: E402
from gaitnet_core.runtime import PlannerRuntime  # noqa: E402
from gaitnet_core.samplers import make_sampler  # noqa: E402
from gaitnet_sim.eval.env_cfg import apply_bundle_contract  # noqa: E402
from gaitnet_sim.eval.landing import LandingProbe, summarize, write_csv  # noqa: E402
from gaitnet_sim.isaac_robot import IsaacRobot  # noqa: E402
from gaitnet_sim.tasks import register  # noqa: E402

# Isaac Lab configures the root logger (so basicConfig would do nothing); log on our own
logger = logging.getLogger("landing_error")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [landing_error] %(message)s"))
logger.addHandler(_handler)


def main() -> int:
    register()
    device = args_cli.device or "cuda:0"
    bundle = load_bundle(args_cli.bundle, map_location=device)
    logger.info(f"bundle {args_cli.bundle}: {bundle.extra}")

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    apply_bundle_contract(env_cfg, bundle)
    generator = env_cfg.scene.terrain.terrain_generator
    generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    generator.curriculum = False
    env_cfg.curriculum.terrain_levels = None
    if not args_cli.randomize:
        env_cfg.play_mode()
    env_cfg.scene.contact_forces.update_period = 0.0
    # the planner samples its own candidates from the terrain scan
    env_cfg.observations.candidates = None
    env_cfg.observations.terrain = None
    controller_cfg = env_cfg.actions.footstep.controller
    logger.info(f"controller: {type(controller_cfg).__name__}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = IsaacRobot(env)
    observers = [] if args_cli.no_observers else bundle.make_observers()
    planner = bundle.planner(make_sampler(args_cli.sampler))
    runtime = PlannerRuntime(robot, planner, observers=observers, deterministic=not args_cli.stochastic)

    with torch.inference_mode():
        # made here so its buffers are inference tensors like everything that updates them
        probe = LandingProbe(
            robot.term.controller,
            robot.term.io,
            dt=env.physics_dt,
            foot_radius=controller_cfg.foot_radius,
            device=env.device,
        )
        robot.term.controller = probe
        env.reset()
        runtime.reset()
        ticks = 0
        while simulation_app.is_running() and ticks < args_cli.num_steps:
            runtime.step()
            ticks += 1
            _, _, terminated, truncated, _ = robot.last_step
            ended = (terminated | truncated).nonzero().flatten()
            if len(ended):
                runtime.reset(ended)
            if ticks % 125 == 0:
                logger.info(f"{ticks}/{args_cli.num_steps} ticks, {len(probe.rows)} footsteps")

    rows = probe.rows
    probe.close()
    env.close()

    out = args_cli.out or f"logs/landing/{Path(args_cli.bundle).stem}_{args_cli.task}_{args_cli.difficulty:g}.csv"
    if rows:
        write_csv(rows, out)
        logger.info(f"wrote {len(rows)} footsteps to {out}")
    for line in summarize(rows):
        logger.info(line)
    return 0 if rows else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
