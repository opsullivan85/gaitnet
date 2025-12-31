"""Utility to allow for running a generic action term slower than the physics rate.
"""

from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

class HierarchicalActionTerm(ActionTerm):
    """Hierarchical Action Term

    Allows for processing actions at a lower rate than the environment step rate.
    """
    def __init__(self, cfg: "HierarchicalActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # create the action term
        self.inner = cfg.action_cfg.class_type(cfg.action_cfg, self._env)
        """The action term being wrapped"""
        self._skip = cfg.skip
        self._step_count = 0
        # sanity check if term is valid type
        if not isinstance(self.inner, ActionTerm):
            raise TypeError(f"Returned object for the term 'action_cfg' is not of type ActionTerm.")

    def __getattr__(self, key):
        """Defers attribute access to the inner action term.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        return getattr(self.inner, key)
    
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        This class defers to the inner action term on a set interval.

        Note:
            This is called at every simulation step by the manager.
        """
        if self._step_count % self._skip == 0:
            self.inner.apply_actions()
        self._step_count += 1

    # the following need to be explicitly defined to avoid ABC errors

    @property
    def action_dim(self) -> int:
        return self.inner.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self.inner.raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.inner.processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        return self.inner.process_actions(actions)


@configclass
class HierarchicalActionCfg(ActionTermCfg):
    """Configuration for the Hierarchical Action Term
    """

    class_type: type[ActionTerm] = HierarchicalActionTerm

    action_cfg: ActionTermCfg = MISSING  # type: ignore
    """Configuration for the underlying action term."""

    skip: int = MISSING  # type: ignore
    """Number of physics steps to skip before running the action."""
