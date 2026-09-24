"""Per-episode statistics of what the footstep action term executes, for the training log.

The term records one tick per env step and, as episodes end, hands their statistics to
`env.extras["log"]`, which RSL-RL averages into each iteration's metrics next to Isaac Lab's
`Episode_Reward/*`. Everything stays on the device: per-env running sums, pooled over the
envs that reset together and handed over as 0-dim tensors, so logging never waits on the GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from gaitnet_core.interfaces import FootstepCommand
from gaitnet_core.robot_spec import RobotSpec

if TYPE_CHECKING:
    from gaitnet_sim.robot_io import RobotIO

LOW_SUPPORT_FEET = 3
"""Fewer feet than this in contact is low support."""


class RatioMeter:
    """Named ratios of per-env running sums, pooled over the envs whose episodes end.

    Pooling (sum of numerators over sum of denominators) weights every tick alike and needs
    no per-env division, so an episode that never took a step doesn't need special-casing.
    """

    def __init__(self):
        self.names: tuple[str, ...] = ()
        self._sums: torch.Tensor | None = None
        """(N, M, 2) each ratio's numerator and denominator, summed over the episode so far."""

    def add(self, ratios: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """One tick's (N,) numerator and denominator increments, by name. Every call must
        name the same ratios in the same order."""
        terms = torch.stack([torch.stack(pair, dim=-1) for pair in ratios.values()], dim=1)
        if self._sums is None:
            self.names = tuple(ratios)
            # made outside inference mode, so a reset outside it (play, evaluation) can clear
            # it in place
            with torch.inference_mode(False):
                self._sums = torch.zeros_like(terms)
        self._sums += terms

    def pop(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each ratio pooled over `env_ids` as a 0-dim tensor, then those envs start over.
        A ratio whose denominator is still zero reads 0; empty before the first `add`."""
        if self._sums is None:
            return {}
        sums = self._sums[env_ids].sum(dim=0)
        ratios = sums[:, 0] / sums[:, 1].clamp_min(1e-9)
        self._sums[env_ids] = 0.0
        return dict(zip(self.names, ratios.unbind()))


class GaitMetrics:
    """The footstep behaviour logged per episode.

    - `Gait/steps_per_s`: footsteps started per second
    - `Gait/leg_share_<leg>`: the fraction of footsteps each leg took
    - `Gait/swing_duration`: mean commanded swing duration (s)
    - `Gait/duration_at_limit`: fraction of footsteps whose duration sits at an end of the
      robot's `swing_duration_range`, where the action term clamps it
    - `Gait/step_length`: mean horizontal distance from the stepping foot to its target at
      the moment the step is commanded (m)
    - `Gait/low_support_frac`: fraction of ticks with fewer than `LOW_SUPPORT_FEET` feet in
      contact
    - `Observer/nudged_frac`: fraction of ticks on which feedback observers changed the command
    - `Observer/nudge_xy`: mean size of the xy nudge on those ticks (m/s)
    """

    def __init__(self, io: "RobotIO", spec: RobotSpec, step_dt: float):
        self.io = io
        self.leg_names = spec.leg_names
        self.duration_range = spec.swing_duration_range
        self.step_dt = step_dt
        self._meter = RatioMeter()

    def record(self, footsteps: FootstepCommand, nudge: torch.Tensor) -> None:
        """One env step: the footsteps it started and the (N, 3) nudge it applied, with the
        feet where they are before the step executes."""
        active = footsteps.active.float()
        ones = torch.ones_like(active)
        # comparing against arange rather than one_hot, which syncs to check its indices
        leg = footsteps.leg.clamp(min=0)
        steps_by_leg = (
            leg.unsqueeze(-1) == torch.arange(len(self.leg_names), device=leg.device)
        ) * active.unsqueeze(-1)
        foot = self.io.foot_pos_hip().gather(1, leg.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        step_length = (footsteps.target[:, :2] - foot[:, :2]).norm(dim=-1)
        low, high = self.duration_range
        at_limit = ((footsteps.duration <= low) | (footsteps.duration >= high)).float()
        low_support = (self.io.foot_contact().sum(dim=-1) < LOW_SUPPORT_FEET).float()
        nudged = (nudge != 0).any(dim=-1).float()

        self._meter.add(
            {
                "Gait/steps_per_s": (active, self.step_dt * ones),
                **{
                    f"Gait/leg_share_{name}": (steps_by_leg[:, i], active)
                    for i, name in enumerate(self.leg_names)
                },
                "Gait/swing_duration": (footsteps.duration * active, active),
                "Gait/duration_at_limit": (at_limit * active, active),
                "Gait/step_length": (step_length * active, active),
                "Gait/low_support_frac": (low_support, ones),
                "Observer/nudged_frac": (nudged, ones),
                "Observer/nudge_xy": (nudge[:, :2].norm(dim=-1), nudged),
            }
        )

    def pop(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """The episodes of `env_ids` just ended: their statistics, pooled, as 0-dim tensors."""
        return self._meter.pop(env_ids)
