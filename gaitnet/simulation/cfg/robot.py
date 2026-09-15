import copy

import isaaclab.sim as sim_utils  # type: ignore
from isaaclab.actuators import DCMotorCfg  # type: ignore
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # type: ignore


# override the default ActuatorNetMLPCfg motors so we can usetorque control
ROBOT_CFG = copy.deepcopy(UNITREE_GO1_CFG)
ROBOT_CFG.actuators["base_legs"] = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.7,
    saturation_effort=23.7,
    velocity_limit=30.0,
    stiffness=0.0,
    damping=0.0,
)

# the USD asset's baked-in foot material has unknown (likely low) friction, which
# silently undercuts the high ground friction set in terrain.py once combined.
# "max" combine mode ensures the higher of the two materials always wins.
ROBOT_CFG.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.5,
    dynamic_friction=1.5,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)

# default of 4 converges the friction cone poorly, showing up as visible foot slip
ROBOT_CFG.spawn.articulation_props.solver_position_iteration_count = 8

# the base UNITREE_GO1_CFG spawns at the USD's default standing height (0.4 m),
# well above the MPC's nominal stance height (_bodyHeight = 0.26 in
# Quadruped.py). Combined with the near-straight reset leg pose, this caused
# every episode to open with an uncontrolled ~13 cm drop before the MPC could
# regain height. Spawn at the MPC's target height instead.
ROBOT_CFG.init_state.pos = (0.0, 0.0, 0.27)
