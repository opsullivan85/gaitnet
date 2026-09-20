"""Batched rotations, in the conventions the CPU controller uses.

These mirror `gaitnet_mpc.mpc.math_utils.orientation_tools` and the Eigen calls in
`gaitnet_mpc/cpp/mpc_osqp.cc`, so the GPU controller sees the same frames as the one it
stands in for. Two conventions appear there and both are needed:

- *active* rotations turn a vector inside one frame. `rpy_to_rotation_zyx` and
  `rpy_to_rotation_xyz` build them, and upstream uses both, from the same angles, a few
  lines apart (see `srbd.py`).
- *frame* rotations re-express a vector's components in another frame, and are the
  transposes of the active ones. `quat_to_rotation` returns one.

Everything is batched, first dimension N.
"""

from __future__ import annotations

import torch

PITCH_SINE_LIMIT = 0.99999
"""What the sine of the pitch is clamped to before `asin`.

Upstream (`orientation_tools.quat_to_rpy`) clamps only from above, which leaves a NaN
for a sine below -1; clamping both ways can only differ from it where it would have
produced that NaN.
"""


def _basic_rotation(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """(..., 3, 3) active rotation of `angle` (rad) about `axis`, 0 for x, 1 for y, 2 for z."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 0:
        rows = ((one, zero, zero), (zero, cos, -sin), (zero, sin, cos))
    elif axis == 1:
        rows = ((cos, zero, sin), (zero, one, zero), (-sin, zero, cos))
    else:
        rows = ((cos, -sin, zero), (sin, cos, zero), (zero, zero, one))
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def rotation_z(yaw: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) active rotation about z by an (N,) yaw (rad)."""
    return _basic_rotation(yaw, 2)


def quat_to_rotation(quat: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) frame rotation from the world into the base, for an (N, 4) xyzw quaternion.

    `v_base = quat_to_rotation(q) @ v_world`. Matches `orientation_tools.quat_to_rot`,
    whose docstring calls this "a coordinate transformation into the frame which has the
    orientation specified by the quaternion".
    """
    x, y, z, w = quat.unbind(-1)
    # world_R_base, transposed on return to get the frame rotation
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    active = torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
    return active.transpose(-1, -2)


def quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """(N, 3) roll, pitch, yaw (rad) of an (N, 4) xyzw quaternion, z-y'-x" intrinsic."""
    x, y, z, w = quat.unbind(-1)
    roll = torch.atan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z)
    sine = torch.clamp(-2 * (x * z - w * y), -PITCH_SINE_LIMIT, PITCH_SINE_LIMIT)
    yaw = torch.atan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z)
    return torch.stack([roll, torch.asin(sine), yaw], dim=-1)


def rotation_to_rpy(rotation: torch.Tensor) -> torch.Tensor:
    """(N, 3) roll, pitch, yaw (rad) of an (N, 3, 3) *frame* rotation.

    The same angles `orientation_tools.rot_to_rpy` produces. That goes matrix to
    quaternion to angles; these come straight off the matrix, which is the same map
    (the quaternion's sign cancels in every term of `quat_to_rpy`).
    """
    roll = torch.atan2(rotation[..., 1, 2], rotation[..., 2, 2])
    sine = torch.clamp(-rotation[..., 0, 2], -PITCH_SINE_LIMIT, PITCH_SINE_LIMIT)
    yaw = torch.atan2(rotation[..., 0, 1], rotation[..., 0, 0])
    return torch.stack([roll, torch.asin(sine), yaw], dim=-1)


def rpy_to_rotation_zyx(rpy: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) active Rz(yaw) Ry(pitch) Rx(roll) for an (N, 3) roll, pitch, yaw.

    `ConvertRpyToRot` in `mpc_osqp.cc`, which uses it to carry the inertia into the yaw
    frame. Its transpose is `orientation_tools.rpy_to_rot`, the frame rotation.
    """
    roll, pitch, yaw = rpy.unbind(-1)
    return _basic_rotation(yaw, 2) @ _basic_rotation(pitch, 1) @ _basic_rotation(roll, 0)


def rpy_to_rotation_xyz(rpy: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) active Rx(roll) Ry(pitch) Rz(yaw) for an (N, 3) roll, pitch, yaw.

    The `com_rotation` quaternion in `ConvexMpc::ComputeContactForces`, which uses it to
    carry the foot positions into the yaw frame. It composes the *same* angles in the
    opposite order to `rpy_to_rotation_zyx` and so is a different rotation whenever roll
    and pitch are both non-zero; upstream uses one for the inertia and the other for the
    foot positions in the same solve, and we keep that.
    """
    roll, pitch, yaw = rpy.unbind(-1)
    return _basic_rotation(roll, 0) @ _basic_rotation(pitch, 1) @ _basic_rotation(yaw, 2)


def skew(vector: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) skew-symmetric matrix of a (..., 3) vector, so `skew(a) @ b == a x b`."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    rows = ((zero, -z, y), (z, zero, -x), (-y, x, zero))
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
