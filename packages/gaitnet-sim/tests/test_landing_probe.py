"""`LandingProbe`'s bookkeeping on a scripted footstep, with a fake scene and controller.
Without the simulator."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_core.interfaces import FootstepCommand  # noqa: E402
from gaitnet_sim.eval.landing import LandingProbe, summarize  # noqa: E402

DT = 0.01
RADIUS = 0.02
HIP = (0.1, 0.2, 0.3)


class FakeController:
    num_robots = 1

    def __init__(self):
        self.commands: list[FootstepCommand] = []

    def command_footsteps(self, footsteps):
        self.commands.append(footsteps)

    def compute_torques(self, *args):
        return torch.zeros(1, 12)

    def reset(self, robot_ids):
        pass

    def gait_timing(self):
        return torch.zeros(1, 4, 3)

    def close(self):
        pass


class FakeIO:
    """Bodies 0-3 are the hips, 4-7 the feet; the base sits at the origin facing +y."""

    def __init__(self):
        yaw = math.pi / 2
        self.pose = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
        self.bodies = torch.zeros(1, 8, 3)
        self.bodies[0, 0] = torch.tensor(HIP)
        self.forces = torch.zeros(1, 4, 3)
        self.robot = SimpleNamespace(
            find_bodies=lambda names, preserve_order: ([0, 1, 2, 3], names),
            data=SimpleNamespace(body_link_pos_w=SimpleNamespace(torch=self.bodies)),
        )
        self.contact_sensor = SimpleNamespace(
            data=SimpleNamespace(net_normal_forces_w=SimpleNamespace(torch=self.forces))
        )
        self.foot_ids = [4, 5, 6, 7]
        self.contact_ids = [0, 1, 2, 3]
        self.contact_threshold = 1.0

    def base_pose(self):
        return self.pose

    def base_vel(self):
        return torch.zeros(1, 6)

    def place_foot(self, xyz, contact: bool):
        self.bodies[0, 4] = torch.tensor(xyz) + torch.tensor([0.0, 0.0, RADIUS])
        self.forces[0, 0, 2] = 50.0 if contact else 0.0


def test_a_footstep_is_measured_against_the_commanded_foothold():
    io, controller = FakeIO(), FakeController()
    probe = LandingProbe(controller, io, dt=DT, foot_radius=RADIUS, device="cpu")
    io.place_foot((0.1, 0.2, 0.0), contact=True)
    probe.command_footsteps(
        FootstepCommand(
            active=torch.tensor([True]),
            leg=torch.tensor([0]),
            target=torch.tensor([[0.05, 0.0, -0.3]]),
            duration=torch.tensor([0.1]),
        )
    )
    assert len(controller.commands) == 1

    # lifted for steps 1-7, down 2 cm past the foothold (along the heading, +y) at step 8
    for step in range(30):
        if step == 1:
            io.place_foot((0.1, 0.22, 0.05), contact=False)
        if step == 8:
            io.place_foot((0.1, 0.27, 0.0), contact=True)
        probe.compute_torques(None, None, io.pose, torch.zeros(1, 6), torch.zeros(1, 3))

    assert len(probe.rows) == 1
    row = probe.rows[0]
    # the target is 5 cm ahead of the hip; ahead is +y in the world
    assert row["commanded_x"] == pytest.approx(0.1, abs=1e-6)
    assert row["commanded_y"] == pytest.approx(0.25, abs=1e-6)
    assert row["commanded_z"] == pytest.approx(0.0, abs=1e-6)
    assert row["touchdown_y"] == pytest.approx(0.27, abs=1e-6)
    assert row["touchdown_t"] == pytest.approx(-0.02, abs=1e-6)
    assert row["lifted"] == 1.0 and row["end_contact"] == 1.0 and row["interrupted"] == 0

    lines = summarize(probe.rows)
    touchdown = next(line for line in lines if line.startswith("touchdown - commanded"))
    # |xy| 2 cm, all of it along the heading, none lateral
    size, _, _, along, lateral = map(float, touchdown.split()[4:9])
    assert (size, along, lateral) == pytest.approx((0.02, 0.02, 0.0), abs=1e-4)


def test_a_reset_drops_the_footstep_in_flight():
    io, controller = FakeIO(), FakeController()
    probe = LandingProbe(controller, io, dt=DT, foot_radius=RADIUS, device="cpu")
    probe.command_footsteps(
        FootstepCommand(
            active=torch.tensor([True]),
            leg=torch.tensor([2]),
            target=torch.tensor([[0.0, 0.0, -0.3]]),
            duration=torch.tensor([0.1]),
        )
    )
    probe.compute_torques(None, None, io.pose, torch.zeros(1, 6), torch.zeros(1, 3))
    probe.reset(torch.tensor([0]))
    for _ in range(30):
        probe.compute_torques(None, None, io.pose, torch.zeros(1, 6), torch.zeros(1, 3))
    assert probe.rows == []
