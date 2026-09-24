"""Pinned footholds: a footstep's target is fixed in the world when it is commanded, and the
swing keeps aiming there while the body travels, turns and tilts."""

import math

import numpy as np

from gaitnet_mpc import MpcFootstepController

DT = 0.004
VELOCITY = np.array([0.3, -0.1, 0.02])
TARGET = np.array([0.12, 0.09, -0.28])


def rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Base to world, z-y'-x" intrinsic."""
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    (cr, sr), (cp, sp), (cy, sy) = [(math.cos(a / 2), math.sin(a / 2)) for a in (roll, pitch, yaw)]
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def travelling(step: int) -> tuple[np.ndarray, tuple[float, float, float]]:
    """(13,) body state moving at a constant world velocity, as the simulator integrates
    it, while turning, pitching and rolling; and its angles."""
    angles = (0.03 * math.sin(step / 7), 0.05 - 0.001 * step, 0.4 + 0.01 * step)
    body = np.zeros(13)
    body[:3] = np.array([1.0, 2.0, 0.27]) + VELOCITY * DT * step
    body[3:7] = quaternion(*angles)
    body[7:10] = VELOCITY
    return body, angles


def standing_joints() -> np.ndarray:
    joint = np.zeros((4, 3, 2))
    joint[:, 0, 0] = [0.05, -0.05, 0.05, -0.05]
    joint[:, 1, 0] = 0.8
    joint[:, 2, 0] = -1.5
    return joint


def aimed_in_world(controller: MpcFootstepController, body: np.ndarray, angles, leg: int) -> np.ndarray:
    """Where the swing is heading: the vendored controller's frame is the body's, with its
    origin at the estimated position under the base."""
    cmpc = controller.robot_runner.cMPC
    position = controller.robot_runner._stateEstimator.getResult().position
    from_base = (cmpc.foot_swing_trajectories[leg]._pf - position).flatten()
    return body[:3] + rotation(*angles) @ from_base


def test_a_pinned_foothold_stays_put_in_the_world():
    controller = MpcFootstepController(dt=DT, iterations_between_mpc=5)
    joints = standing_joints()
    controller.get_torques(joints, travelling(0)[0], np.zeros(3))

    leg = 0
    controller.initiate_footstep(leg, TARGET, 0.2)
    body, angles = travelling(1)
    hip = body[:3] + rotation(*angles) @ controller.robot_runner._quadruped.getHipLocation(leg).flatten()
    expected = hip + rotation(0.0, 0.0, angles[2]) @ TARGET

    for step in range(1, 41):
        body, angles = travelling(step)
        controller.get_torques(joints, body, np.zeros(3))
        # the vendored rotations are float16, so its frame is good to about three decimals
        np.testing.assert_allclose(aimed_in_world(controller, body, angles, leg), expected, atol=2e-3)


def test_reset_releases_the_pin():
    controller = MpcFootstepController(dt=DT)
    joints = standing_joints()
    controller.get_torques(joints, travelling(0)[0], np.zeros(3))
    controller.initiate_footstep(0, TARGET, 0.2)
    controller.get_torques(joints, travelling(1)[0], np.zeros(3))
    assert controller._pinned[0]
    controller.reset()
    assert not controller._pinned.any() and not controller._pin_pending.any()
    # the vendored travel prediction stays off across the runner's rebuild
    assert not controller.robot_runner.cMPC.compensate_body_travel
