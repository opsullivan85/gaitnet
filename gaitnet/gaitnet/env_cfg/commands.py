from isaaclab.envs import mdp
from isaaclab.utils import configclass

max_xy_vel = 0.2
max_raw_rate = 0.4

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-max_xy_vel, max_xy_vel), lin_vel_y=(-max_xy_vel, max_xy_vel), ang_vel_z=(-max_raw_rate, max_raw_rate)
        ),
    )