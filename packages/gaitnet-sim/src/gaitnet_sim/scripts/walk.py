"""Smoke test: a scripted trot, driven through the low-level controller, in the training env.

    python -m gaitnet_sim.scripts.walk --num_envs 4 --num_steps 1000
    python -m gaitnet_sim.scripts.walk --task GaitNet-Pillars --difficulty 0.3

The policy's action is the no-op throughout; the diagonal trot is commanded straight to the
controller, which the footstep interface allows (several footsteps per tick). Each foot goes
to the valid foothold nearest its nominal spot, at the height the terrain scan reads there,
so on pillars the controller has to place feet at different heights. Dynamics are nominal and
observations exact (the env cfg's `play_mode`) unless `--randomize` is given.

Fails if an observation goes non-finite, if the controller's view of the base orientation
disagrees with the simulator's (the quaternion convention), if the terrain scan doesn't read
the ground below the hips, if a leg never has a valid footstep candidate, or if feet land
too far from the height they were sent to.

Trailing key=value arguments are Hydra overrides of the env cfg.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="GaitNet-Holes")
parser.add_argument("--difficulty", type=float, default=0.0, help="Terrain difficulty of every sub-terrain.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--spawn_yaw", type=float, default=0.7, help="Base yaw at reset (rad), for the orientation check.")
parser.add_argument(
    "--randomize", action="store_true", help="Keep training's friction, mass and push randomization and observation noise."
)
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
from gaitnet_core.terrain import inner_heights, valid_footholds  # noqa: E402
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
# nominal (x, y) in each hip's frame: slightly outward, fore and aft
STEP_XY = {0: (0.05, 0.1), 1: (0.05, -0.1), 2: (-0.05, 0.1), 3: (-0.05, -0.1)}
CYCLE_S = 0.4
SWING_S = 0.2
YAW_TOLERANCE = 0.05
GROUND_BELOW_HIP = (-0.35, -0.18)
LANDING_TOLERANCE = 0.02
"""Largest median distance (m) between where feet land and the height they were sent to."""


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


def nearest_foothold(term: FootstepControlAction, leg: int, desired_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The valid foothold (cell centre, at its terrain height) nearest `desired_xy` (N, 2).

    Returns:
        target: (N, 3) in the leg's hip yaw frame; `desired_xy` at the nominal height where
            the leg has no valid foothold
        found: (N,) bool
    """
    rules = term._env.cfg.gaitnet.foothold_rules()
    heights = term.terrain().heights
    valid = valid_footholds(heights, term.spec, term.grid, rules.step_threshold, rules.edge_margin)[:, leg]
    centres = term.grid.cell_centers(desired_xy.device).flatten(0, 1)  # (H W, 2)
    distance = (centres.unsqueeze(0) - desired_xy.unsqueeze(1)).square().sum(-1)
    distance = distance.masked_fill(~valid.flatten(1), float("inf"))
    best = distance.argmin(dim=1)
    found = torch.isfinite(distance.gather(1, best.unsqueeze(1)).squeeze(1))
    z = inner_heights(heights, term.grid)[:, leg].flatten(1).gather(1, best.unsqueeze(1)).squeeze(1)
    nominal = torch.cat([desired_xy, torch.full_like(z, -term.spec.nominal_height).unsqueeze(1)], dim=1)
    target = torch.cat([centres[best], z.unsqueeze(1)], dim=1)
    return torch.where(found.unsqueeze(1), target, nominal), found


def step_pair(
    term: FootstepControlAction, legs: tuple[int, ...], lead: torch.Tensor, heights: list[torch.Tensor]
) -> tuple[dict[int, torch.Tensor], int]:
    """Start a swing on each of `legs`, led in the direction of travel by `lead` (N, 2).
    Appends the commanded footholds' heights below the hip to `heights`.

    Returns:
        the commanded foothold height in the world frame per leg, (N,), and how many
        footsteps had no valid foothold
    """
    n = lead.shape[0]
    landing_z, missing = {}, 0
    hips_z = torch.stack([scanner.data.pos_w.torch[:, 2] for scanner in term.io.scanners], dim=1)
    for leg in legs:
        desired = torch.tensor(STEP_XY[leg], device=lead.device) + lead
        target, found = nearest_foothold(term, leg, desired)
        missing += int((~found).sum())
        heights.append(target[found, 2])
        term.controller.command_footsteps(
            FootstepCommand(
                active=torch.ones(n, dtype=torch.bool, device=lead.device),
                leg=torch.full((n,), leg, dtype=torch.long, device=lead.device),
                target=target,
                duration=torch.full((n,), SWING_S, device=lead.device),
            )
        )
        landing_z[leg] = torch.where(found, hips_z[:, leg] + target[:, 2], torch.full_like(target[:, 2], float("nan")))
    return landing_z, missing


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
    """The highest surface in each leg's grid is about a stance height below the hip, which
    catches a scan in the wrong frame on any terrain (the cell under a hip may be a gap)."""
    highest = inner_heights(term.terrain().heights, term.grid).amax(dim=(-2, -1))
    logger.info(f"highest ground in each leg's grid, below the hip (m), env 0: {highest[0].tolist()}")
    low, high = GROUND_BELOW_HIP
    if not ((highest > low) & (highest < high)).all():
        return [f"terrain below the hips reads {highest.min().item():.3f}..{highest.max().item():.3f} m, expected {GROUND_BELOW_HIP}"]
    return []


def main() -> int:
    register()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    # training's terrain type and events, at one difficulty and without the curriculum
    generator = env_cfg.scene.terrain.terrain_generator
    generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    generator.curriculum = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (args_cli.spawn_yaw, args_cli.spawn_yaw)
    if not args_cli.randomize:
        env_cfg.play_mode()

    env = ManagerBasedRLEnv(cfg=env_cfg)
    term: FootstepControlAction = env.action_manager.get_term("footstep")  # type: ignore[assignment]
    noop = noop_action(env.num_envs, env.device)
    cycle_steps = max(2, round(CYCLE_S / env.step_dt))
    # each leg is re-planted once per cycle, so lead by a cycle's worth of travel
    lead_time = cycle_steps * env.step_dt
    foot_radius = env_cfg.actions.footstep.controller.foot_radius
    num_legs = term.spec.num_legs

    errors: list[str] = []
    swing_ticks, missing, episodes_ended = 0, 0, 0
    ended_by: dict[str, int] = {}
    # most valid candidates any robot had, per leg; zero while a leg can't step (swing, or
    # the minimum-contact rule), so only the maximum over the run says the pipeline works
    max_valid = torch.zeros(num_legs, device=env.device)
    # world height each foot was last sent to, NaN when unknown or reset since
    landing_target = torch.full((env.num_envs, num_legs), float("nan"), device=env.device)
    landing_errors: list[torch.Tensor] = []
    foothold_heights: list[torch.Tensor] = []
    env.reset()
    was_swinging = torch.zeros(env.num_envs, num_legs, dtype=torch.bool, device=env.device)
    start = time.monotonic()
    count = 0
    with torch.inference_mode():
        while simulation_app.is_running() and count < args_cli.num_steps and not errors:
            phase = count % cycle_steps
            if phase in (0, cycle_steps // 2):
                lead = term.base_command()[:, :2] * lead_time
                commanded, no_foothold = step_pair(term, TROT[0] if phase == 0 else TROT[1], lead, foothold_heights)
                missing += no_foothold
                for leg, z in commanded.items():
                    landing_target[:, leg] = z

            observations, _, terminated, truncated, _ = env.step(noop)
            for group, value in observations.items():
                if not torch.isfinite(value).all():
                    errors.append(f"non-finite {group} observation at step {count}")

            # a foot whose scheduled swing just ended has landed: compare the bottom of the
            # foot with the height it was sent to
            swinging = term.controller.gait_timing()[..., 0] > 0
            landed = was_swinging & ~swinging
            ended = terminated | truncated
            landed[ended] = False
            if landed.any():
                feet_z = term.io.robot.data.body_link_pos_w.torch[:, term.io.foot_ids, 2] - foot_radius
                error = (feet_z - landing_target)[landed]
                landing_errors.append(error[torch.isfinite(error)])
            landing_target[ended] = float("nan")
            episodes_ended += int(ended.sum())
            for name in env.termination_manager.active_terms:
                ended_by[name] = ended_by.get(name, 0) + int(env.termination_manager.get_term(name).sum())
            was_swinging = swinging

            swing_ticks += int(swinging.any(dim=-1).sum())
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
        f"{args_cli.task} at difficulty {args_cli.difficulty}: {count} steps x {args_cli.num_envs} envs in"
        f" {elapsed:.1f} s ({count / elapsed:.1f} steps/s), a leg in swing"
        f" {swing_ticks / max(1, count * args_cli.num_envs):.0%} of robot-steps, {episodes_ended} episodes ended"
        f" ({', '.join(f'{name}={n}' for name, n in ended_by.items() if n) or 'none'})"
    )
    logger.info(f"most valid candidates per leg: {max_valid.tolist()}; footsteps without a valid foothold: {missing}")
    if foothold_heights:
        commanded = torch.cat(foothold_heights)
        spread = torch.quantile(commanded, torch.tensor([0.05, 0.5, 0.95], device=commanded.device))
        logger.info(
            f"commanded foothold heights below the hip (m): 5% {spread[0]:+.4f}, median {spread[1]:+.4f},"
            f" 95% {spread[2]:+.4f}"
        )
    if landing_errors:
        landing = torch.cat(landing_errors)
        quantiles = torch.quantile(landing, torch.tensor([0.05, 0.5, 0.95], device=landing.device))
        logger.info(
            f"landing height error over {landing.numel()} footsteps (m): 5% {quantiles[0]:+.4f},"
            f" median {quantiles[1]:+.4f}, 95% {quantiles[2]:+.4f}"
        )
        if quantiles[1].abs() > LANDING_TOLERANCE:
            errors.append(f"feet land {quantiles[1]:+.3f} m from the commanded height (median)")
    else:
        errors.append("no footstep landed")
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
