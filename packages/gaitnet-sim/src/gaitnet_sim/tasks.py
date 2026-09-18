"""Gym registrations. Isaac Lab's entry points load them through `--external_callback
gaitnet_sim.tasks.register`; scripts call `register()` directly."""

from __future__ import annotations

import gymnasium as gym

TASKS: dict[str, dict[str, str]] = {
    "GaitNet-Holes": {
        "env_cfg_entry_point": "gaitnet_sim.env.env_cfg:GaitNetHolesEnvCfg",
        "rsl_rl_cfg_entry_point": "gaitnet_sim.rl.agent_cfg:GaitNetPpoRunnerCfg",
    },
    "GaitNet-Pillars": {
        "env_cfg_entry_point": "gaitnet_sim.env.env_cfg:GaitNetPillarsEnvCfg",
        "rsl_rl_cfg_entry_point": "gaitnet_sim.rl.agent_cfg:GaitNetPillarsPpoRunnerCfg",
    },
}


def register() -> None:
    for task_id, kwargs in TASKS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs=dict(kwargs),
        )
