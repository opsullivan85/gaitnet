"""Cfg of the footstep action term (`gaitnet_sim.env.actions.FootstepControlAction`)."""

from __future__ import annotations

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import preset

from gaitnet_sim import robot as go1
from gaitnet_sim.controllers import BatchedMpcControllerCfg, PooledMpcControllerCfg
from gaitnet_sim.env.noise import ObservationNoiseCfg
from gaitnet_sim.env.scene import SCANNER_NAMES


@configclass
class FootstepControlActionCfg(ActionTermCfg):
    class_type: str = "gaitnet_sim.env.actions:FootstepControlAction"
    asset_name: str = "robot"

    controller = preset(default=PooledMpcControllerCfg(), gpu_mpc=BatchedMpcControllerCfg())
    """Any controller cfg whose `class_type` implements `LowLevelController`.

    The CPU process pool is the default because it is the controller every bundle and
    baseline in the repo was produced against. `presets=gpu_mpc` swaps in the batched
    GPU controller, which runs the same MPC for every robot at once and is what makes
    large env counts affordable."""
    command_name: str = "base_velocity"
    """The command term holding the velocity command before the nudge."""
    apply_nudge: bool = True
    """Add the action's nudge to the command. The nudge is zero unless a feedback observer
    produced one."""
    observation_noise: ObservationNoiseCfg | None = ObservationNoiseCfg()
    """Noise on the planner's view of the robot (`planner_observation`), None for the truth."""

    joint_names: tuple[str, ...] = go1.JOINT_NAMES
    foot_names: tuple[str, ...] = go1.FOOT_NAMES
    contact_sensor_name: str = "contact_forces"
    scanner_names: tuple[str, ...] = SCANNER_NAMES
    """One foothold scanner per leg, in leg order."""
    contact_threshold: float = 1.0
    """Normal force above which a foot counts as in contact (N)."""
