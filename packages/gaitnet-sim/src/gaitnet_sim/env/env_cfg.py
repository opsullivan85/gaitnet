"""GaitNet's manager-based environment configs.

A stock `ManagerBasedRLEnv` runs these; nothing is subclassed. Experiments change them
with Isaac Lab presets (`presets=spatial,privileged`, which switch on the observation
groups those variants read, together with the agent cfg's matching parts) or overrides,
e.g. `"env.observations.state.robot_state.params.features=['foot_pos','command']"` for the
state vector or `env.observations.candidates.candidates.params.sampler=dense` for the
sampler. See packages/gaitnet-sim/README.md.
"""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import preset

from gaitnet_core.features import DEFAULT_FEATURES
from gaitnet_sim.env import curriculum, observations, rewards, terminations
from gaitnet_sim.env.actions_cfg import FootstepControlActionCfg
from gaitnet_sim.env.contract import GaitNetCfg
from gaitnet_sim.env.scene import GaitNetSceneCfg
from gaitnet_sim.robot import BASE_NAME
from gaitnet_sim.terrains import pillars_terrain_cfg


@configclass
class ObservationsCfg:
    @configclass
    class StateCfg(ObsGroup):
        robot_state = ObsTerm(func=observations.robot_state, params={"features": list(DEFAULT_FEATURES)})

    @configclass
    class TerrainCfg(ObsGroup):
        heights = ObsTerm(func=observations.terrain_heights)

    @configclass
    class CandidatesCfg(ObsGroup):
        candidates = ObsTerm(
            func=observations.footstep_candidates,
            params={"sampler": "uniform_jitter", "sampler_kwargs": {"per_leg": 64}},
        )

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Sim-only inputs for a privileged critic: the terrain ahead of every leg, coarsely,
        the base's clearance and the contact forces."""

        terrain_heights = ObsTerm(func=observations.terrain_height_summary, params={"cells": 5})
        foothold_validity = ObsTerm(func=observations.foothold_validity_summary, params={"cells": 5})
        base_clearance = ObsTerm(func=observations.base_terrain_clearance)
        contact_forces = ObsTerm(func=observations.foot_contact_forces)

    @configclass
    class BaseCommandCfg(ObsGroup):
        base_command = ObsTerm(func=observations.base_command)

    state: StateCfg = StateCfg()
    candidates: CandidatesCfg = CandidatesCfg()

    # Optional groups, off unless a preset needs them: RSL-RL keeps every group in its
    # rollout buffer, and terrain patches alone cost ~4 GB at 1024 envs x 250 steps.
    terrain = preset(default=None, spatial=TerrainCfg(), crop=TerrainCfg())
    """For actors whose networks read terrain (the dense spatial CNN, the crop encoder)."""
    privileged = preset(default=None, privileged=PrivilegedCfg())
    """For the privileged critic."""
    base_command = preset(default=None, slowdown=BaseCommandCfg())
    """For feedback observers running in the actor during training."""


@configclass
class ActionsCfg:
    footstep = FootstepControlActionCfg()


_MAX_XY_VELOCITY = 0.2
_MAX_YAW_RATE = 0.4


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-_MAX_XY_VELOCITY, _MAX_XY_VELOCITY),
            lin_vel_y=(-_MAX_XY_VELOCITY, _MAX_XY_VELOCITY),
            ang_vel_z=(-_MAX_YAW_RATE, _MAX_YAW_RATE),
        ),
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.4)
    xy_tracking = RewTerm(func=rewards.track_lin_vel_xy_exp, weight=0.5, params={"std": 0.1})
    yaw_tracking = RewTerm(func=rewards.track_ang_vel_z_exp, weight=0.5, params={"std": 0.1})

    step_taken = RewTerm(func=rewards.step_taken, weight=-0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    foot_slip = RewTerm(func=rewards.foot_slip, weight=-6.0, params={"threshold": 1.0})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(20)})
    # relative to the terrain, so they mean the same on holes and on pillars
    bad_height = DoneTerm(func=terminations.base_below_terrain_clearance, params={"minimum_height": 0.15})
    foot_below_ground = DoneTerm(func=terminations.feet_below_walkable_terrain, params={"margin": 0.05})
    terrain_out_of_bounds = DoneTerm(
        func=terminations.out_of_terrain, params={"distance_buffer": 0.5}, time_out=True
    )


_JOINT_POS_SCALE = 1.2
"""Reset joint positions scale the defaults by about this. Centred on 1.2 so the reset
stance height (~0.26 m) matches the MPC's nominal height; nearer 0.5 left the legs almost
straight and every episode opened with the robot dropping ~13 cm."""


NOMINAL_FRICTION = 1.0
"""Effective foot-ground friction without randomization (the robot's material multiplies the
terrain's 1.0, see gaitnet_sim.robot)."""


@configclass
class EventsCfg:
    """Resets, plus the sim2real randomization: friction and trunk mass per robot at startup,
    and pushes. `GaitNetEnvCfg.play_mode` turns the randomization off."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            # dynamic friction at most static
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_NAME),
            # the Go1 is ~12 kg, its trunk ~5 kg; the MPC keeps its nominal model
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (_JOINT_POS_SCALE - 0.05, _JOINT_POS_SCALE + 0.05), "velocity_range": (0.0, 0.0)},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=curriculum.terrain_levels_survival)


@configclass
class GaitNetEnvCfg(ManagerBasedRLEnvCfg):
    """Everything but the terrain type, which the task variants below choose."""

    gaitnet: GaitNetCfg = GaitNetCfg()
    scene: GaitNetSceneCfg = GaitNetSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 250 Hz physics and torque control, 25 Hz footstep planning
        self.sim.dt = 0.004
        self.decimation = 10
        self.sim.render_interval = 5
        self.episode_length_s = 20.0
        # the MPC's torques are applied every physics step, which needs a backend that runs
        # the decimation loop in Python (PhysX); see ManagerBasedRLEnv.step
        self.sim.physics = PhysxCfg()
        self.sim.physics_material = self.scene.terrain.physics_material

        # sensors are read once per planning step
        for name in self.scene.__dataclass_fields__:
            sensor = getattr(self.scene, name)
            if isinstance(sensor, SensorBaseCfg):
                sensor.update_period = self.decimation * self.sim.dt

        # the generator lays difficulties out by row only when a curriculum moves robots
        # between rows
        generator = self.scene.terrain.terrain_generator
        if generator is not None:
            generator.curriculum = getattr(self.curriculum, "terrain_levels", None) is not None

    def play_mode(self) -> None:
        """Nominal dynamics and exact observations, for evaluation and smoke tests: nominal
        friction, no added mass, no pushes, no observation noise. (Isaac Lab's play entry
        points call this; our scripts call it themselves.)"""
        friction = self.events.physics_material.params
        friction["static_friction_range"] = (NOMINAL_FRICTION, NOMINAL_FRICTION)
        friction["dynamic_friction_range"] = (NOMINAL_FRICTION, NOMINAL_FRICTION)
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.actions.footstep.observation_noise = None


@configclass
class GaitNetHolesEnvCfg(GaitNetEnvCfg):
    """Randomly holed flat ground; difficulty is the fraction of holes."""


@configclass
class GaitNetPillarsEnvCfg(GaitNetEnvCfg):
    """Square pillars at random heights; difficulty widens the gaps and the height spread."""

    def __post_init__(self):
        self.scene.terrain = pillars_terrain_cfg()
        super().__post_init__()
