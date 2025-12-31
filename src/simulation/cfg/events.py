import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from gaitnet.sim2real import SimInterface
from gaitnet.util.vectorpool import VectorPool


def reset_controller(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset the controller for the specified environments."""
    # see if the cfg has a controllers field
    if not hasattr(env.cfg, "controllers"):
        raise AttributeError("The env cfg does not have a controllers field.")
    if env.cfg.controllers is None:  # type: ignore
        raise ValueError("The env cfg controllers field is None.")
    
    mask = np.zeros(env.num_envs, dtype=bool)
    mask[env_ids.cpu().numpy()] = True

    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    controllers.call(function=SimInterface.reset, mask=mask)
