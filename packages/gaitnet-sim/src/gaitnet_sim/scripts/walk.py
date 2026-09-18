"""Smoke test: a scripted trot, driven through the low-level controller, in the training env.

    python -m gaitnet_sim.scripts.walk --num_envs 4 --num_steps 1000

The policy's action is the no-op throughout; the diagonal trot is commanded straight to the
controller, which the footstep interface allows (several footsteps per tick). Fails if an
observation goes non-finite, if the controller's view of the base orientation disagrees with
the simulator's (the quaternion convention), if the terrain scan doesn't read the ground
below the hips, or if a leg never has a valid footstep candidate.

Trailing key=value arguments are Hydra overrides of the env cfg.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--spawn_yaw", type=float, default=0.7, help="Base yaw at reset (rad), for the orientation check.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import logging  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction  # noqa: E402
from gaitnet_core.interfaces import FootstepCommand  # noqa: E402
from gaitnet_core.terrain import inner_heights  # noqa: E402
from gaitnet_sim.env.actions import FootstepControlAction  # noqa: E402
from gaitnet_sim.tasks import register  # noqa: E402

# Isaac Lab configures the root logger (so basicConfig would do nothing); log on our own
logger = logging.getLogger("walk")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [walk] %(message)s"))
logger.addHandler(_handler)

# diagonal pairs, stepped half a cycle apart
TROT = ((0, 3), (1, 2))
# (x, y) in each hip's frame: slightly outward, fore and aft
STEP_XY = {0: (0.05, 0.1), 1: (0.05, -0.1), 2: (-0.05, 0.1), 3: (-0.05, -0.1)}
CYCLE_S = 0.4
SWING_S = 0.2
YAW_TOLERANCE = 0.05
GROUND_BELOW_HIP = (-0.35, -0.18)


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def noop_action(num_envs: int, device) -> torch.Tensor:
    zeros = torch.zeros(num_envs, device=device)
    return EnvAction(
        choice_index=zeros.long(),
        duration=zeros,
        leg=torch.full((num_envs,), NO_STEP_LEG, device=device),
        target=torch.zeros(num_envs, 3, device=device),
        nudge=torch.zeros(num_envs, 3, device=device),
    ).encode()


def step_pair(term: FootstepControlAction, legs: tuple[int, ...], lead: torch.Tensor) -> None:
    """Start a swing on each of `legs`, led in the direction of travel by `lead` (N, 2)."""
    n = lead.shape[0]
    for leg in legs:
        xy = torch.tensor(STEP_XY[leg], device=lead.device) + lead
        z = torch.full((n, 1), -term.spec.nominal_height, device=lead.device)
        term.controller.command_footsteps(
            FootstepCommand(
                active=torch.ones(n, dtype=torch.bool, device=lead.device),
                leg=torch.full((n,), leg, dtype=torch.long, device=lead.device),
                target=torch.cat([xy, z], dim=-1),
                duration=torch.full((n,), SWING_S, device=lead.device),
            )
        )


def check_orientation(term: FootstepControlAction, spawn_yaw: float) -> list[str]:
    sim_yaw = math_utils.euler_xyz_from_quat(term.io.robot.data.root_link_quat_w.torch)[2]
    controller_yaw = term.controller.estimated_rpy()[:, 2]
    logger.info(f"yaw: spawn {spawn_yaw:.3f}, sim {sim_yaw.tolist()}, controller {controller_yaw.tolist()}")
    errors = []
    if wrap(sim_yaw - spawn_yaw).abs().max() > 0.1:
        errors.append("the simulator's yaw doesn't match the reset event's")
    if wrap(controller_yaw - sim_yaw).abs().max() > YAW_TOLERANCE:
        errors.append("the controller's yaw doesn't match the simulator's: base pose convention")
    return errors


def check_terrain(term: FootstepControlAction) -> list[str]:
    heights = inner_heights(term.terrain().heights, term.grid)
    centre = heights[..., heights.shape[-2] // 2, heights.shape[-1] // 2]  # under each hip
    logger.info(f"ground below each hip (m), env 0: {centre[0].tolist()}")
    low, high = GROUND_BELOW_HIP
    if not ((centre > low) & (centre < high)).all():
        return [f"terrain under the hips reads {centre.min().item():.3f}..{centre.max().item():.3f} m, expected {GROUND_BELOW_HIP}"]
    return []


def main() -> int:
    register()
    env_cfg = parse_env_cfg("GaitNet-Holes", device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    # training's terrain and events, but flat and without the curriculum
    env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
    env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (args_cli.spawn_yaw, args_cli.spawn_yaw)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    term: FootstepControlAction = env.action_manager.get_term("footstep")  # type: ignore[assignment]
    noop = noop_action(env.num_envs, env.device)
    cycle_steps = max(2, round(CYCLE_S / env.step_dt))
    # each leg is re-planted once per cycle, so lead by a cycle's worth of travel
    lead_time = cycle_steps * env.step_dt

    errors: list[str] = []
    swing_ticks = 0
    # most valid candidates any robot had, per leg; zero while a leg can't step (swing, or
    # the minimum-contact rule), so only the maximum over the run says the pipeline works
    max_valid = torch.zeros(term.spec.num_legs, device=env.device)
    env.reset()
    start = time.monotonic()
    count = 0
    with torch.inference_mode():
        while simulation_app.is_running() and count < args_cli.num_steps and not errors:
            phase = count % cycle_steps
            if phase in (0, cycle_steps // 2):
                lead = term.base_command()[:, :2] * lead_time
                step_pair(term, TROT[0] if phase == 0 else TROT[1], lead)

            observations, *_ = env.step(noop)
            for group, value in observations.items():
                if not torch.isfinite(value).all():
                    errors.append(f"non-finite {group} observation at step {count}")

            swing_ticks += int((term.controller.gait_timing()[..., 0] > 0).any(dim=-1).sum())
            max_valid = torch.maximum(max_valid, observations["candidates"][..., 3].sum(dim=-1).amax(dim=0))
            if count == 1:
                errors += check_orientation(term, args_cli.spawn_yaw)
                errors += check_terrain(term)
            if count % 100 == 0:
                logger.info(f"step {count}/{args_cli.num_steps}")
            count += 1
    elapsed = time.monotonic() - start

    term.controller.close()
    env.close()

    logger.info(
        f"{count} steps x {args_cli.num_envs} envs in {elapsed:.1f} s ({count / elapsed:.1f} steps/s),"
        f" a leg in swing {swing_ticks / max(1, count * args_cli.num_envs):.0%} of robot-steps"
    )
    logger.info(f"most valid candidates per leg: {max_valid.tolist()}")
    if (max_valid == 0).any():
        errors.append("a leg never had a valid footstep candidate")
    if errors or count < args_cli.num_steps:
        for error in errors or [f"stopped after {count} steps"]:
            logger.error(error)
        return 1
    logger.info("walk smoke test passed")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
