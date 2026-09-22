"""One planning tick: valid footholds -> candidates -> scores -> a footstep (or none)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.candidates import Candidates
from gaitnet_core.eligibility import step_eligible
from gaitnet_core.features import state_vector
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.networks.candidate_scorer import CandidateScorer
from gaitnet_core.robot_spec import RobotSpec
from gaitnet_core.samplers import CandidateSampler
from gaitnet_core.selection import FootstepDistribution, Scores, Selection, leg_marginals, select_deterministic
from gaitnet_core.state import Observation
from gaitnet_core.terrain import inner_heights, valid_footholds


@dataclass
class FootholdRules:
    """Which cells and legs are allowed. Part of the policy's contract, saved with it."""

    step_threshold: float = 0.02
    """Height difference between neighbouring cells that counts as an edge (m)."""
    edge_margin: int = 2
    """Cells within this many cells of an edge are invalid."""
    min_stance_after_step: int = 2
    """A leg may only lift off if this many legs stay in stance."""

    def valid(self, observation: Observation, spec: RobotSpec) -> torch.Tensor:
        """(N, L, *grid.size) cells each leg may step to this tick."""
        cells = valid_footholds(
            observation.terrain.heights,
            spec,
            observation.terrain.grid,
            step_threshold=self.step_threshold,
            edge_margin=self.edge_margin,
        )
        legs = step_eligible(observation.state.gait_timing, self.min_stance_after_step)
        return cells & legs.unsqueeze(-1).unsqueeze(-1)


@dataclass
class PlanResult:
    candidates: Candidates
    scores: Scores
    selection: Selection
    leg_marginals: torch.Tensor
    """(N, L) log step probability of each leg, comparable to `scores.noop_logit`."""
    is_step: torch.Tensor
    """(N,) bool"""
    leg: torch.Tensor
    """(N,) long, meaningless where not stepping"""
    target: torch.Tensor
    """(N, 3) foothold in the leg's hip yaw frame (m)"""

    def footstep_command(self) -> FootstepCommand:
        return FootstepCommand(
            active=self.is_step, leg=self.leg, target=self.target, duration=self.selection.duration
        )

    def env_action(self, nudge: Nudge | None = None) -> EnvAction:
        leg = torch.where(self.is_step, self.leg, torch.full_like(self.leg, NO_STEP_LEG))
        delta = nudge.command_delta if nudge is not None else torch.zeros_like(self.target)
        return EnvAction(
            choice_index=self.selection.index,
            duration=self.selection.duration,
            leg=leg,
            target=self.target,
            nudge=delta,
        )


def plan_from_scores(scores: Scores, candidates: Candidates, selection: Selection) -> PlanResult:
    is_step, leg, target = candidates.gather(selection.index)
    return PlanResult(
        candidates=candidates,
        scores=scores,
        selection=selection,
        leg_marginals=leg_marginals(scores, candidates),
        is_step=is_step,
        leg=leg,
        target=target,
    )


class FootstepPlanner:
    def __init__(
        self,
        network: CandidateScorer,
        spec: RobotSpec,
        grid: FootholdGrid,
        features: tuple[str, ...],
        sampler: CandidateSampler,
        rules: FootholdRules | None = None,
        duration_std: float = 0.05,
        max_rows_per_forward: int = 65536,
    ):
        """
        Args:
            features: robot state features the network was trained on
            duration_std: swing duration noise when planning stochastically
            max_rows_per_forward: robots are scored in chunks so (robots * candidates)
                stays under this, bounding peak activation memory with dense sampling
        """
        self.network = network
        self.spec = spec
        self.grid = grid
        self.features = tuple(features)
        self.sampler = sampler
        self.rules = rules or FootholdRules()
        self.duration_std = duration_std
        self.max_rows_per_forward = max_rows_per_forward

    def sample(self, observation: Observation, generator: torch.Generator | None = None) -> Candidates:
        valid = self.rules.valid(observation, self.spec)
        heights = inner_heights(observation.terrain.heights, self.grid)
        return self.sampler.sample(valid, self.grid, heights=heights, generator=generator)

    def score(self, observation: Observation, candidates: Candidates) -> Scores:
        state = state_vector(observation.state, self.features)
        per_robot = candidates.num_legs * candidates.per_leg + 1
        chunk = max(1, self.max_rows_per_forward // per_robot)
        parts = [
            self.network(state[i : i + chunk], candidates[i : i + chunk], observation.terrain.heights[i : i + chunk])
            for i in range(0, state.shape[0], chunk)
        ]
        return Scores(
            step_logits=torch.cat([p.step_logits for p in parts]),
            noop_logit=torch.cat([p.noop_logit for p in parts]),
            duration=torch.cat([p.duration for p in parts]),
        )

    @torch.no_grad()
    def plan(
        self,
        observation: Observation,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> PlanResult:
        candidates = self.sample(observation, generator)
        scores = self.score(observation, candidates)
        if deterministic:
            selection = select_deterministic(scores, candidates)
        else:
            fixed = getattr(self.network, "fixed_duration", None) is not None
            std = None if fixed else torch.tensor(self.duration_std, device=scores.duration.device)
            selection = FootstepDistribution(scores, candidates, std).sample()
            # the sampled duration is the network's mean plus unbounded noise
            low, high = self.spec.swing_duration_range
            is_step = selection.index != candidates.noop_index
            selection.duration = torch.where(is_step, selection.duration.clamp(low, high), 0.0)
        return plan_from_scores(scores, candidates, selection)
