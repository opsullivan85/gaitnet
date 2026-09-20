"""Batched leg kinematics: where each foot is, and the Jacobian that turns a foot force
into joint torques.

A three-link leg (abduction, thigh, calf) in closed form, the same expressions as
`LegController.computeLegJacobianAndPosition` in `gaitnet_mpc`, evaluated for N robots
and all four legs at once. Positions are relative to the leg's own hip, in a frame
aligned with the base; leg order is FL, FR, RL, RR and joint order within a leg is hip,
thigh, calf.
"""

from __future__ import annotations

import torch

from gaitnet_core.control.model import SrbdModel

SIDE_SIGN: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)
"""+1 for a left leg, -1 for a right one, in leg order. `utils.getSideSign` upstream."""


def leg_offsets(model: SrbdModel, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """(L, 3) the link offsets each leg's kinematics are built from.

    Columns are the abduction offset along y (signed by which side the leg is on) and
    the thigh and calf lengths, both negative because the links hang below the hip.
    """
    side = torch.tensor(SIDE_SIGN[: model.num_legs], device=device, dtype=dtype)
    abad = side * model.spec.abad_length
    thigh = torch.full_like(abad, -model.spec.thigh_length)
    calf = torch.full_like(abad, -model.spec.calf_length)
    return torch.stack([abad, thigh, calf], dim=-1)


def foot_position_and_jacobian(
    joint_pos: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Foot positions and Jacobians for every leg of every robot.

    Args:
        joint_pos: (N, L, 3) joint angles (rad), hip, thigh, calf within a leg.
        offsets: (L, 3) from `leg_offsets`.

    Returns:
        (N, L, 3) foot position relative to the leg's hip, in the base frame (m), and
        (N, L, 3, 3) the Jacobian d(position)/d(joint angles) (m/rad).
    """
    dy, dz1, dz2 = offsets.unbind(-1)
    sin, cos = torch.sin(joint_pos), torch.cos(joint_pos)
    s1, s2, s3 = sin.unbind(-1)
    c1, c2, c3 = cos.unbind(-1)
    # the thigh and calf angles only ever act through their sum
    c23 = c2 * c3 - s2 * s3
    s23 = s2 * c3 + c2 * s3

    position = torch.stack(
        [
            dz2 * s23 + dz1 * s2,
            dy * c1 - dz1 * c2 * s1 - dz2 * s1 * c23,
            dy * s1 + dz1 * c1 * c2 + dz2 * c1 * c23,
        ],
        dim=-1,
    )

    zero = torch.zeros_like(s1)
    abduction = (
        zero,
        -dy * s1 - dz2 * c1 * c23 - dz1 * c1 * c2,
        -dz2 * s1 * c23 + dy * c1 - dz1 * c2 * s1,
    )
    columns = (
        abduction,
        (dz2 * c23 + dz1 * c2, dz2 * s1 * s23 + dz1 * s1 * s2, -dz2 * c1 * s23 - dz1 * c1 * s2),
        (dz2 * c23, dz2 * s1 * s23, -dz2 * c1 * s23),
    )
    jacobian = torch.stack([torch.stack(column, dim=-1) for column in columns], dim=-1)
    return position, jacobian
