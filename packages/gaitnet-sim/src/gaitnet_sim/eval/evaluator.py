"""Per-robot outcomes of an evaluation trial: how far each robot walked before its first
episode ended, and which termination ended it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn


class Evaluator:
    def __init__(self, env: "ManagerBasedRLEnv", asset_name: str = "robot"):
        self.env = env
        self.asset_name = asset_name
        self.term_names: list[str] = list(env.termination_manager.active_terms)
        self.start()

    def start(self) -> None:
        """Begin a trial. Call after resetting the env."""
        n, device = self.env.num_envs, self.env.device
        self.done = torch.zeros(n, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(n, dtype=torch.bool, device=device)
        self.distance = torch.zeros(n, device=device)
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.reason = torch.full((n,), -1, dtype=torch.long, device=device)

    def record(self, step: "VecEnvStepReturn") -> bool:
        """Record one `env.step` return. True once every robot's first episode has ended."""
        _, _, terminated, truncated, _ = step
        ended = terminated | truncated
        running = ~self.done
        self.steps[running] += 1

        # Robots whose episode just ended were already reset, so their position now is a
        # fresh spawn: they keep the distance from the step before. Distance is along the
        # command (+x) from the sub-terrain the robot spawned on.
        position = self.env.scene[self.asset_name].data.root_link_pos_w.torch
        x = position[:, 0] - self.env.scene.env_origins[:, 0]
        walking = running & ~ended
        self.distance[walking] = x[walking]

        finished = running & ended
        if finished.any():
            manager = self.env.termination_manager
            fired = torch.stack([manager.get_term(name) for name in self.term_names], dim=1)
            self.reason[finished] = fired.float().argmax(dim=1)[finished]
            self.truncated[finished] = truncated[finished]
        self.done |= ended
        return bool(self.done.all())

    def rows(self) -> list[dict]:
        """One row per robot: env, distance (m), steps, truncated, terminated_by."""
        names = self.term_names
        return [
            {
                "env": env_id,
                "distance": round(distance, 4),
                "steps": steps,
                "truncated": int(truncated),
                "terminated_by": names[reason] if reason >= 0 else "",
            }
            for env_id, (distance, steps, truncated, reason) in enumerate(
                zip(
                    self.distance.tolist(),
                    self.steps.tolist(),
                    self.truncated.tolist(),
                    self.reason.tolist(),
                )
            )
        ]
