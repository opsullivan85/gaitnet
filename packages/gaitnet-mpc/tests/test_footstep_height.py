"""Footsteps carry a height: the swing ends at the commanded point below the hip, and its apex
clears the higher end of the swing."""

import numpy as np
import pytest

from gaitnet_mpc import MpcFootstepController
from gaitnet_mpc.mpc.common.FootSwingTrajectory import FootSwingTrajectory

DT = 0.004


def standing_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint = np.zeros((4, 3, 2))
    joint[:, 0, 0] = [0.05, -0.05, 0.05, -0.05]
    joint[:, 1, 0] = 0.8
    joint[:, 2, 0] = -1.5
    body = np.zeros(13)
    body[2] = 0.27
    body[6] = 1.0  # xyzw identity
    return joint, body, np.zeros(3)


@pytest.mark.parametrize("step", [0.1, -0.08])
def test_swing_apex_clears_the_higher_end(step):
    swing = FootSwingTrajectory()
    swing.setInitialPosition(np.zeros((3, 1), dtype=np.float32))
    swing.setFinalPosition(np.array([[0.2], [0.0], [step]], dtype=np.float32))
    swing.setHeight(0.05)
    heights = []
    for phase in np.linspace(0.0, 1.0, 41):
        swing.computeSwingTrajectoryBezier(float(phase), 0.2)
        heights.append(float(swing.getPosition()[2, 0]))
    assert max(heights) == pytest.approx(max(0.0, step) + 0.05, abs=1e-6)
    assert heights[-1] == pytest.approx(step, abs=1e-6)


def test_swing_ends_at_the_commanded_point_below_the_hip():
    controller = MpcFootstepController(dt=DT, iterations_between_mpc=5)
    joint, body, command = standing_state()
    controller.get_torques(joint, body, command)

    leg, target = 0, np.array([0.05, 0.08, -0.18])
    controller.initiate_footstep(leg, target, 0.2)
    controller.get_torques(joint, body, command)

    cmpc = controller.robot_runner.cMPC
    position = controller.robot_runner._stateEstimator.getResult().position
    hip = controller.robot_runner._quadruped.getHipLocation(leg)
    final = cmpc.foot_swing_trajectories[leg]._pf - position - hip
    np.testing.assert_allclose(final.flatten(), target, atol=1e-5)


def test_default_footstep_height_is_the_nominal_body_height():
    controller = MpcFootstepController(dt=DT)
    runner = controller.robot_runner
    np.testing.assert_allclose(runner.cMPC.footstep_locations_hip[:, 2], -runner._quadruped._bodyHeight)


def test_rejects_planar_footsteps():
    controller = MpcFootstepController(dt=DT)
    with pytest.raises(ValueError):
        controller.initiate_footstep(0, np.array([0.05, 0.08]), 0.2)
