
from isaaclab.envs import ManagerBasedRLEnvCfg  # type: ignore
from isaaclab.utils import configclass  # type: ignore

from gaitnet.sim2real.siminterface import SimInterface
from gaitnet.util import VectorPool
from gaitnet.simulation.cfg.manager_components import (
    ActionsCfg,
    EventsCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from gaitnet.simulation.cfg.scene import SceneCfg
import numpy as np

from gaitnet import get_logger
logger = get_logger()


@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(env_spacing=2.5)  # type: ignore
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: ActionsCfg = ActionsCfg()  # type: ignore
    events: EventsCfg = EventsCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    rewards: RewardsCfg = RewardsCfg()  # type: ignore

    # controller
    # we need this here so the events can access it to reset individual robots
    controllers: VectorPool[SimInterface] | None = None

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
        self.decimation = 1  # env decimation -> 250 Hz control
        self.render_interval = self.decimation  # render at control rate
        # simulation settings
        self.sim.dt = 0.004  # simulation timestep -> 250 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material

        self.episode_length_s = 5


def get_quadruped_env_cfg(num_envs: int, device: str) -> QuadrupedEnvCfg:
    """Get the quadruped environment configuration.

    Args:
        num_envs (int): Number of environments.
        device (str): Device to use.

    Returns:
        QuadrupedEnvCfg: The quadruped environment configuration.
    """
    cfg = QuadrupedEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    # update sensor update periods
    # we tick all the sensors based on the smallest update period (physics update period)
    for scanner in [
        cfg.scene.FR_foot_scanner,
        cfg.scene.FL_foot_scanner,
        cfg.scene.RL_foot_scanner,
        cfg.scene.RR_foot_scanner,
        cfg.scene.contact_forces,
    ]:
        scanner.update_period = cfg.decimation * cfg.sim.dt  # control rate
    return cfg
