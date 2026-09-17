"""Records how far each robot walked before it fell."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from isaaclab.assets import ArticulationData
from isaaclab.envs import ManagerBasedRLEnv, VecEnvObs, VecEnvStepReturn

from gaitnet import PROJECT_ROOT

data_folder = PROJECT_ROOT / "data" / "evaluations"
data_folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EvalGroup:
    """One output file and the contiguous block of environments that feeds it.

    A single scene holds every difficulty at once, so the environments have to be split
    back out per difficulty when the results are written.
    """

    file_name: str
    envs: slice


class Evaluator:
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        trials: int,
        groups: Sequence[EvalGroup],
    ) -> None:
        self.env = env
        self.groups = list(groups)
        self.remaining_trials = trials
        self.trials = trials

        self.data_files: dict[str, Path] = {}
        for group in self.groups:
            data_file = data_folder / group.file_name
            data_file.unlink(missing_ok=True)
            with open(data_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(["trial", "distance", "truncated"])
            self.data_files[group.file_name] = data_file

        self._reset_buffers()

    def _write_trial(self) -> None:
        trial = self.trials - self.remaining_trials + 1
        distances = self.terminal_distances.cpu().numpy()
        truncations = self.truncations.cpu().numpy()
        for group in self.groups:
            with open(self.data_files[group.file_name], "a") as f:
                writer = csv.writer(f)
                for distance, truncated in zip(distances[group.envs], truncations[group.envs]):
                    writer.writerow([trial, distance, int(truncated)])

    def _reset_buffers(self) -> None:
        self.dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.terminal_distances = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        self.truncations = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    @property
    def done(self) -> bool:
        return self.remaining_trials == 0

    def process(self, data: VecEnvStepReturn) -> VecEnvObs | None:
        """Record one step, and start the next trial once every robot is done.

        Args:
            data: The return value of `env.step`.

        Returns:
            The observations from the reset when a trial finished and another is left to
            run, otherwise None. The caller has to pick these up: acting on the
            pre-reset observations would put the first step of a trial one step stale.
        """
        observations, rew, terminated, truncated, info = data
        dones = torch.logical_or(truncated, terminated)

        robot_data: ArticulationData = self.env.scene["robot"].data

        self.truncations = torch.logical_or(self.truncations, truncated)
        self.dones = torch.logical_or(self.dones, dones)

        # measured from the sub-terrain the robot spawned on, since the difficulties are
        # laid out along the terrain's rows and so start at different world positions
        x_positions = robot_data.root_link_pos_w[:, 0] - self.env.scene.env_origins[:, 0]
        self.terminal_distances[~self.dones] = x_positions[~self.dones]

        if not torch.all(self.dones):
            return None

        self._write_trial()
        self.remaining_trials -= 1
        if self.done:
            return None

        observations, _ = self.env.reset()
        self._reset_buffers()
        return observations
