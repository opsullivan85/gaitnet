"""The simulated Go1, driven by joint torques from our low-level controller."""

from __future__ import annotations

import copy

from isaaclab.actuators import DCMotorCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from gaitnet_core.robot_spec import LEG_NAMES

JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{joint}_joint" for leg in LEG_NAMES for joint in ("hip", "thigh", "calf")
)
"""Leg-major (FL hip, FL thigh, FL calf, FR hip, ...), the order of the core contract."""
FOOT_NAMES: tuple[str, ...] = tuple(f"{leg}_foot" for leg in LEG_NAMES)
HIP_NAMES: tuple[str, ...] = tuple(f"{leg}_hip" for leg in LEG_NAMES)
"""Hip links. Each link's origin is its abduction joint, the origin of the leg's hip frame."""
BASE_NAME = "trunk"

# joint-side limits of the Go1 motors, as in isaaclab_assets' GO1_ACTUATOR_CFG and go1.usd:
# the calf sits behind an extra 1.5:1 knee reduction
_EFFORT_LIMIT = {".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 35.55}
_VELOCITY_LIMIT = {".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 20.06}

# deep copied: the nested spawn and init_state cfgs are modified below, and a shallow
# `replace` would share them with isaaclab_assets' global config
GO1_TORQUE_CFG = copy.deepcopy(UNITREE_GO1_CFG)
GO1_TORQUE_CFG.actuators = {
    # zero gains: the DC motor model only applies the controller's torques, clipped to the
    # motor's torque-speed curve
    "base_legs": DCMotorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        saturation_effort=_EFFORT_LIMIT,
        actuator_effort_limit=_EFFORT_LIMIT,
        actuator_velocity_limit=_VELOCITY_LIMIT,
        # keep the solver's clamp from cutting in below the motor model's
        joint_effort_limit=_EFFORT_LIMIT,
        stiffness=0.0,
        damping=0.0,
    ),
}
# the USD's baked-in foot material has unknown (likely low) friction, which would undercut
# the terrain's friction once combined; "max" makes the higher of the two win. Isaac Lab 3's
# Go1 has instanced collision prims, which can't take a material binding until uninstanced.
GO1_TORQUE_CFG.spawn.make_uninstanceable = True
GO1_TORQUE_CFG.spawn.physics_material = PhysxRigidBodyMaterialCfg(
    static_friction=1.5,
    dynamic_friction=1.5,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)
# the default of 4 converges the friction cone poorly, showing up as visible foot slip
GO1_TORQUE_CFG.spawn.articulation_props.solver_position_iteration_count = 8
# spawn at the MPC's nominal stance height (Quadruped._bodyHeight = 0.26) rather than the
# USD's 0.4 m, so episodes don't open with a drop the controller has to catch
GO1_TORQUE_CFG.init_state.pos = (0.0, 0.0, 0.27)
