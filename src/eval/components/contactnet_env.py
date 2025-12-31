from typing import Any
import torch
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    RecorderManager,
    RewardManager,
    TerminationManager,
)
import copy

from gaitnet import get_logger
from gaitnet.gaitnet.actions.mpc_action import ManagerBasedEnv
from gaitnet.gaitnet.components.footstep_candidate_sampler import FootstepCandidateSampler
from gaitnet.gaitnet.components.gaitnet_observation_manager import (
    GaitNetObservationManager,
)
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
import gaitnet.constants as const

logger = get_logger()


class ContactNetEnv(GaitNetEnv):
    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # prepare the managers
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        # Note that this is the one change from the parent class
        self.observation_manager = GaitNetObservationManager(
            cfg=self.cfg.observations,
            env=self,
            footstep_option_generator=FootstepCandidateSampler(
                env=self, options_per_leg=1,  # just list the best options
                noise=False  # don't apply any noise
            ),
            num_footstep_options=1,
        )
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if (
            self.__class__ == ManagerBasedEnv
            and "startup" in self.event_manager.available_modes
        ):
            self.event_manager.apply(mode="startup")

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")