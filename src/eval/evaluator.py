from math import hypot, trunc
import torch
from isaaclab.assets import ArticulationData
from isaaclab.envs import ManagerBasedRLEnv, VecEnvObs, VecEnvStepReturn
from gaitnet import PROJECT_ROOT
import csv

data_folder = PROJECT_ROOT / "data" / "evaluations"
data_folder.mkdir(parents=True, exist_ok=True)


class Evaluator:
    def __init__(self, env: ManagerBasedRLEnv, observations: VecEnvObs, trials: int, name: str) -> None:
        self.env = env
        self._reset_from_obs(observations)
        self.remaining_trials = trials
        self.trials = trials
        self.data_file = data_folder / name
        self.data_file.unlink(missing_ok=True)
        with open(self.data_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "distance", "truncated"])

    def _reset(self):
        with open(self.data_file, "a") as f:
            writer = csv.writer(f)
            for distance, trunc in zip(self.terminal_distances.cpu().numpy(), self.truncations.cpu().numpy()):
                writer.writerow([self.trials - self.remaining_trials + 1, distance, int(trunc)])

        self.remaining_trials -= 1
        observations, _ = self.env.reset()
        self._reset_from_obs(observations)

    def _reset_from_obs(self, observations: VecEnvObs):
        self.initial_obs: torch.Tensor = observations["policy"]  # type: ignore
        self.dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.terminal_distances = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        self.truncations = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    @property
    def done(self) -> bool:
        return self.remaining_trials == 0
    
    def process(self, data: VecEnvStepReturn) -> None:
        observations, rew, terminated, truncated, info = data
        dones = torch.logical_or(truncated, terminated)

        robot_data: ArticulationData = self.env.scene["robot"].data

        self.truncations = torch.logical_or(self.truncations, truncated)

        self.dones = torch.logical_or(self.dones, dones)
        x_positions = robot_data.root_link_pos_w[:, 0]
        self.terminal_distances[~self.dones] = x_positions[~self.dones]
        if torch.all(self.dones):
            self._reset()


