from isaaclab.utils import configclass
from gaitnet.gaitnet.actions.footstep_action import FSCActionCfg
from gaitnet.gaitnet.actions.hierarchical_action import HierarchicalActionCfg
from gaitnet.gaitnet.actions.mpc_action import MPCActionCfg, MPCControlActionCfg


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    footstep_controller = FSCActionCfg(
        asset_name="robot",
    )

    # trick type hinting system into giving us better hints by faking type
    # note that the MPCControlActionCfg doesn't consume any actions, it just reads from commands
    mpc_controller: MPCControlActionCfg = HierarchicalActionCfg(  # type: ignore
        action_cfg=MPCControlActionCfg(
            command_name="base_velocity",
            asset_name="robot",
            joint_names=[".*"],  # all joints
        ),
        skip=1,  # run at full speed
        asset_name="robot",
    )

