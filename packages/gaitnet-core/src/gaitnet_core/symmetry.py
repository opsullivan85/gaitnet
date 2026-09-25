"""The robot's left/right mirror symmetry, applied to what the planner observes and decides.

The robot and its task are symmetric, up to the robot model's small asymmetries, under a
reflection in the base's x-z plane: y becomes -y and every left leg trades places with the
right leg beside it. Symmetry-augmented training (`gaitnet_sim.rl.symmetry`) uses this
module to add the mirror image of every sample; anything that needs to know how a quantity
mirrors asks here, and mirroring twice is the identity.

How things mirror:
- vectors (positions, velocities, gravity) flip y; angular velocities, being pseudovectors,
  flip x and z; a velocity command or nudge (vx, vy, yaw rate) flips vy and the yaw rate
- per-leg tensors swap FL with FR and RL with RR
- per-leg maps over the foothold grid (terrain patches, masks) also flip their y axis: the
  right legs' grid is the left legs' mirrored (`gaitnet_core.grid`), so cell (i, j) of FR
  is the mirror image of cell (i, size_y - 1 - j) of FL, border included
- a candidate keeps its slot and moves to the mirrored leg, so the flat action index maps
  leg block to leg block and the no-op stays where it is
- the state vector mirrors feature by feature, as each `gaitnet_core.features.Feature`
  declares
"""

from __future__ import annotations

from functools import lru_cache

import torch

from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.candidates import Candidates
from gaitnet_core.features import FEATURES
from gaitnet_core.robot_spec import LEG_NAMES
from gaitnet_core.state import Observation, RobotState, TerrainPatch


def mirrored_legs(leg_names: tuple[str, ...] = LEG_NAMES) -> tuple[int, ...]:
    """Index of each leg's mirror image: (1, 0, 3, 2) for FL, FR, RL, RR."""
    other_side = {"L": "R", "R": "L"}
    return tuple(leg_names.index(name[:-1] + other_side[name[-1]]) for name in leg_names)


LEG_MIRROR: tuple[int, ...] = mirrored_legs()


def _signed(t: torch.Tensor, signs: tuple[float, ...]) -> torch.Tensor:
    return t * torch.tensor(signs, device=t.device, dtype=t.dtype)


def mirror_vector(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) a position, velocity or direction, y flipped."""
    return _signed(v, (1.0, -1.0, 1.0))


def mirror_angular(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) an angular velocity, x and z flipped."""
    return _signed(w, (-1.0, 1.0, -1.0))


def mirror_command(command: torch.Tensor) -> torch.Tensor:
    """(..., 3) a velocity command or nudge (vx, vy, yaw rate), vy and yaw rate flipped."""
    return _signed(command, (1.0, -1.0, -1.0))


def mirror_legs(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """`t` with its leg dimension `dim` reordered so each leg holds its mirror image's values."""
    return t.index_select(dim, torch.tensor(LEG_MIRROR, device=t.device))


def mirror_leg_maps(maps: torch.Tensor) -> torch.Tensor:
    """(..., L, X, Y) per-leg maps over the foothold grid (or the terrain patch around it),
    legs swapped and each map's y axis flipped."""
    return mirror_legs(maps, dim=-3).flip(-1)


def mirror_state(state: RobotState) -> RobotState:
    # every field by name, so a new one can't pass through unmirrored
    return RobotState(
        foot_pos=mirror_vector(mirror_legs(state.foot_pos)),
        foot_vel=mirror_vector(mirror_legs(state.foot_vel)),
        base_lin_vel=mirror_vector(state.base_lin_vel),
        base_ang_vel=mirror_angular(state.base_ang_vel),
        projected_gravity=mirror_vector(state.projected_gravity),
        contact=mirror_legs(state.contact),
        gait_timing=mirror_legs(state.gait_timing),
        command=mirror_command(state.command),
        base_command=mirror_command(state.base_command),
    )


def mirror_terrain(terrain: TerrainPatch) -> TerrainPatch:
    return TerrainPatch(heights=mirror_leg_maps(terrain.heights), grid=terrain.grid)


def mirror_observation(observation: Observation) -> Observation:
    return Observation(mirror_state(observation.state), mirror_terrain(observation.terrain))


def mirror_candidates(candidates: Candidates) -> Candidates:
    return Candidates(
        xyz=mirror_vector(mirror_legs(candidates.xyz)),
        valid=mirror_legs(candidates.valid),
        log_q=mirror_legs(candidates.log_q),
    )


def mirror_action_index(index: torch.Tensor, num_legs: int, per_leg: int) -> torch.Tensor:
    """Flat action indices (leg * per_leg + slot, or the no-op at num_legs * per_leg) of the
    mirrored choice."""
    is_step = (index >= 0) & (index < num_legs * per_leg)
    leg, slot = index // per_leg, index % per_leg
    mirrored = torch.tensor(LEG_MIRROR, device=index.device)[leg.clamp(0, num_legs - 1)] * per_leg + slot
    return torch.where(is_step, mirrored, index)


def action_index_mirror(num_legs: int, per_leg: int, device: torch.device | str | None = None) -> torch.Tensor:
    """(num_legs * per_leg + 1,) the permutation of the flat action space: entry i of a
    mirrored sample's per-action tensor (logits, duration means) is entry `perm[i]` of the
    original's."""
    return mirror_action_index(torch.arange(num_legs * per_leg + 1, device=device), num_legs, per_leg)


def mirror_env_action(action: EnvAction, num_legs: int, per_leg: int) -> EnvAction:
    is_step = action.leg != NO_STEP_LEG
    leg = torch.tensor(LEG_MIRROR, device=action.leg.device)[action.leg.clamp(0, num_legs - 1)]
    return EnvAction(
        choice_index=mirror_action_index(action.choice_index, num_legs, per_leg),
        duration=action.duration,
        leg=torch.where(is_step, leg, action.leg),
        target=mirror_vector(action.target),
        nudge=mirror_command(action.nudge),
    )


@lru_cache
def _state_vector_mirror(names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    num_legs = len(LEG_MIRROR)
    index: list[int] = []
    signs: list[float] = []
    offset = 0
    for name in names:
        feature = FEATURES[name]
        if feature.dim_per_leg and feature.dim_fixed:
            raise ValueError(f"feature '{name}' mixes per-leg and fixed dimensions, which can't be mirrored")
        per = feature.dim_per_leg or feature.dim_fixed
        if len(feature.mirror) != per:
            raise ValueError(f"feature '{name}' has {per} components but {len(feature.mirror)} mirror signs")
        if feature.dim_fixed:
            index += range(offset, offset + per)
            signs += feature.mirror
        else:
            for position in range(num_legs * per):
                if feature.leg_major:
                    leg, component = divmod(position, per)
                    source = LEG_MIRROR[leg] * per + component
                else:
                    component, leg = divmod(position, num_legs)
                    source = component * num_legs + LEG_MIRROR[leg]
                index.append(offset + source)
                signs.append(feature.mirror[component])
        offset += feature.dim(num_legs)
    return tuple(index), tuple(signs)


def mirror_state_vector(vector: torch.Tensor, names: tuple[str, ...] | list[str]) -> torch.Tensor:
    """(N, feature_dim) a `gaitnet_core.features.state_vector` of the named features, mirrored."""
    index, signs = _state_vector_mirror(tuple(names))
    if len(index) != vector.shape[-1]:
        raise ValueError(f"features {list(names)} make a {len(index)}-dim state vector, got {vector.shape[-1]}")
    return _signed(vector[..., torch.tensor(index, device=vector.device)], signs)
