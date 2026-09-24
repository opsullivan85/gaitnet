"""Run a policy bundle on its task's training terrain until stopped: for watching it (over a
livestream, docker/README.md), not for measuring it, which is `eval_sweep`'s job.

    python -m gaitnet_sim.scripts.play --bundle bundle.pt --num_envs 16
    python -m gaitnet_sim.scripts.play --bundle bundle.pt --task GaitNet-Pillars \\
        env.scene.terrain.max_init_terrain_level=9

The scene is training's: the terrain grid with its curriculum moving robots between rows,
random velocity commands, and robots that fall reset on their own. The policy runs through
the same `PlannerRuntime` as `eval_sweep` and hardware, with the bundle's contract, dense
candidates, deterministic selection and the bundle's feedback observers, on nominal dynamics
with exact observations unless `--randomize` is given. Nothing is written or logged to
MLflow; a line of episode outcomes and the mean terrain level is printed every
`--report_s` seconds.

`--footholds` draws what the planner sees for the robots it names (`gaitnet_sim.viz`): per-leg
heatmaps of the scores and masks, written to `--footholds_dir/robot<id>.png` (replaced as it
runs; open it in a viewer that reloads). They show raw logits, what the deterministic policy
compares within a leg, or with `--footholds_logits corrected` the ones it samples from, on the
no-op's scale; by default whichever the run's policy uses.

    python -m gaitnet_sim.scripts.play --bundle bundle.pt --footholds 0 3

Trailing key=value arguments are Hydra overrides of the env cfg.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--bundle", required=True, help="Policy bundle file (gaitnet_sim.scripts.export_bundle).")
parser.add_argument("--task", default="GaitNet-Holes")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=None, help="Stop after this many ticks; runs until closed by default.")
parser.add_argument("--sampler", default="dense", help="Candidate sampler (gaitnet_core.samplers.SAMPLERS).")
parser.add_argument("--stochastic", action="store_true", help="Sample footsteps instead of the deterministic choice.")
parser.add_argument("--no_observers", action="store_true", help="Leave out the feedback observers the bundle was trained with.")
parser.add_argument(
    "--randomize", action="store_true", help="Keep training's friction, mass and push randomization and observation noise."
)
parser.add_argument("--report_s", type=float, default=10.0, help="Seconds between progress lines.")
parser.add_argument("--footholds", type=int, nargs="+", default=[], help="Robots whose foothold maps to draw.")
parser.add_argument(
    "--footholds_logits", choices=["raw", "corrected"], default=None, help="Default: raw, or corrected with --stochastic."
)
parser.add_argument("--footholds_every", type=int, default=1, help="Planning ticks between drawings.")
parser.add_argument("--footholds_dir", default="logs/footholds", help="Where --footholds writes its images.")
parser.add_argument("--footholds_frames", action="store_true", help="Keep every image drawn, in --footholds_dir/frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from gaitnet_core.bundle import load_bundle  # noqa: E402
from gaitnet_core.runtime import PlannerRuntime  # noqa: E402
from gaitnet_core.samplers import make_sampler  # noqa: E402
from gaitnet_sim.eval.env_cfg import apply_bundle_contract  # noqa: E402
from gaitnet_sim.isaac_robot import IsaacRobot  # noqa: E402
from gaitnet_sim.tasks import register  # noqa: E402

# Isaac Lab configures the root logger (so basicConfig would do nothing); log on our own
logger = logging.getLogger("play")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [play] %(message)s"))
logger.addHandler(_handler)


def foothold_plots(num_envs: int, planner) -> list:
    """The `--footholds` plotter, as `PlannerRuntime` on_plan callbacks."""
    robots = args_cli.footholds
    if not robots:
        return []
    if not all(0 <= robot < num_envs for robot in robots):
        raise ValueError(f"--footholds {robots} out of range for {num_envs} envs")
    from gaitnet_sim.viz.foothold_plot import FootholdPlot

    kind = args_cli.footholds_logits or ("corrected" if args_cli.stochastic else "raw")
    out_dir = args_cli.footholds_dir
    logger.info(f"foothold plots of robots {robots} ({kind} logits) in {out_dir}")
    return [
        FootholdPlot(
            planner, robots, out_dir, kind=kind, every=args_cli.footholds_every, keep_frames=args_cli.footholds_frames
        )
    ]


def main() -> int:
    register()
    device = args_cli.device or "cuda:0"
    bundle = load_bundle(args_cli.bundle, map_location=device)
    logger.info(f"bundle {args_cli.bundle}: {bundle.extra}")

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    apply_bundle_contract(env_cfg, bundle)
    if not args_cli.randomize:
        env_cfg.play_mode()
    # the planner samples its own candidates from the terrain scan
    env_cfg.observations.candidates = None
    env_cfg.observations.terrain = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = IsaacRobot(env)
    observers = [] if args_cli.no_observers else bundle.make_observers()
    logger.info(f"observers: {bundle.observers if observers else 'none'}")
    planner = bundle.planner(make_sampler(args_cli.sampler))
    runtime = PlannerRuntime(
        robot,
        planner,
        observers=observers,
        deterministic=not args_cli.stochastic,
        on_plan=foothold_plots(env.num_envs, planner),
    )
    terrain = env.scene.terrain

    ticks, ended_by = 0, Counter()
    last_report = time.monotonic()
    with torch.inference_mode():
        env.reset()
        runtime.reset()
        while simulation_app.is_running() and (args_cli.num_steps is None or ticks < args_cli.num_steps):
            runtime.step()
            ticks += 1
            _, _, terminated, truncated, _ = robot.last_step
            ended = (terminated | truncated).nonzero().flatten()
            if len(ended):
                # the env has already reset these robots; their observers start over too
                runtime.reset(ended)
                for name in env.termination_manager.active_terms:
                    ended_by[name] += int(env.termination_manager.get_term(name)[ended].sum())
            if time.monotonic() - last_report >= args_cli.report_s:
                levels = getattr(terrain, "terrain_levels", None)
                level = f", mean terrain level {levels.float().mean():.2f}" if levels is not None else ""
                outcomes = ", ".join(f"{name}={count}" for name, count in ended_by.most_common()) or "none"
                logger.info(f"{ticks} ticks; episodes ended by {outcomes}{level}")
                last_report = time.monotonic()

    robot.term.controller.close()
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
