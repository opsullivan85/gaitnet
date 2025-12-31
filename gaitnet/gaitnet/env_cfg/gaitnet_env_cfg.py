from typing import TypeVar
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from gaitnet import sim2real
from gaitnet.gaitnet.env_cfg.curriculum import CurriculumCfg
from gaitnet.simulation.cfg.scene import SceneCfg
import numpy as np

from gaitnet import get_logger
from gaitnet.gaitnet.env_cfg.observations import ObservationsCfg
from gaitnet.gaitnet.env_cfg.actions import ActionsCfg
from gaitnet.gaitnet.env_cfg.events import EventsCfg
from gaitnet.gaitnet.env_cfg.terminations import TerminationsCfg
from gaitnet.gaitnet.env_cfg.rewards import RewardsCfg
from gaitnet.gaitnet.env_cfg.commands import CommandsCfg
from gaitnet.util import VectorPool

logger = get_logger()


@configclass
class GaitNetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(env_spacing=2.5)  # type: ignore

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: ActionsCfg = ActionsCfg()  # type: ignore
    events: EventsCfg = EventsCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    rewards: RewardsCfg = RewardsCfg()  # type: ignore
    commands: CommandsCfg = CommandsCfg()  # type: ignore
    curriculum: CurriculumCfg = CurriculumCfg()  # type: ignore

    robot_controllers: VectorPool[sim2real.Sim2RealInterface] = None  # type: ignore to be set later

    # TODO: do this the right way
    # ['trunk', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    hip_indices = np.asarray([1, 2, 3, 4], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    thigh_indices = np.asarray([5, 6, 7, 8], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    calf_indices = np.asarray([9, 10, 11, 12], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    foot_indices = np.asarray([13, 14, 15, 16], dtype=np.int32)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # env decimation -> 25 Hz footstep planning
        self.render_interval = (
            self.decimation / 2
        )  # render faster than footstep planning rate
        # simulation settings
        self.sim.dt = 0.004  # simulation timestep -> 250 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material

        self.episode_length_s = 20


def get_env_cfg(num_envs: int, device: str) -> GaitNetEnvCfg:
    cfg = GaitNetEnvCfg()

    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    # update sensor update periods
    # we tick all the sensors based on the smallest update period (physics update period)
    # update all properties of cfg.scene which subclass SensorBaseCfg
    for attr_name in cfg.scene.__dir__():
        attr = getattr(cfg.scene, attr_name)
        if not issubclass(type(attr), SensorBaseCfg):
            continue
        sensor: SensorBaseCfg = attr  # type: ignore
        sensor.update_period = cfg.decimation * cfg.sim.dt  # control rate
    return cfg


def update_controllers(
    cfg: GaitNetEnvCfg, num_envs: int
) -> None:
    """Update the controllers in the environment configuration.

    Args:
        envcfg (GaitNetEnvCfg): The environment configuration.
        controllers (VectorPool[sim2real.Sim2RealInterface]): The controllers to set.
    """
    controllers: VectorPool[sim2real.Sim2RealInterface] = VectorPool(
        instances=num_envs,
        cls=sim2real.SimInterface,
        dt=cfg.sim.dt,  # 250 Hz leg PD control
        iterations_between_mpc=5,  # 50 Hz MPC
        debug_logging=False,
    )
    cfg.robot_controllers = controllers  # type: ignore

# generic for manager_class
T = TypeVar("T", bound=ManagerBasedRLEnv)

def get_env(
    num_envs: int, device: str, manager_class: type[T] = ManagerBasedRLEnv
) -> T:
    """Get the environment configuration and the environment instance."""
    env_cfg = get_env_cfg(num_envs, device)
    env = manager_class(cfg=env_cfg)
    update_controllers(env_cfg, num_envs)
    return env
