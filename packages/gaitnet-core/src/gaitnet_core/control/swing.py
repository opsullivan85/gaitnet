"""The arc a foot follows between liftoff and touchdown, batched over robots and legs.

A cubic Bezier in x and y over the whole swing, and two half-swing Beziers in z that
meet at an apex above the *higher* of the two ends, so a step up onto a raised foothold
comes down onto it rather than approaching it from below. Same curve as
`FootSwingTrajectory.computeSwingTrajectoryBezier` in `gaitnet_mpc`.
"""

from __future__ import annotations

import torch


def _bezier(start: torch.Tensor, end: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Cubic Bezier from `start` to `end` at `phase` in [0, 1], flat at both ends."""
    return start + (phase**3 + 3.0 * phase**2 * (1.0 - phase)) * (end - start)


def _bezier_rate(start: torch.Tensor, end: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """d/d(phase) of `_bezier`."""
    return 6.0 * phase * (1.0 - phase) * (end - start)


def swing_trajectory(
    start: torch.Tensor,
    end: torch.Tensor,
    apex_clearance: float,
    phase: torch.Tensor,
    duration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where a swinging foot should be, and how fast.

    Args:
        start: (N, L, 3) foot position at liftoff, in the frame the result comes back in.
        end: (N, L, 3) commanded touchdown position, same frame.
        apex_clearance: how far the apex clears the higher end of the step (m).
        phase: (N, L) progress through the swing in [0, 1].
        duration: (N, L) swing duration (s), only read where `phase` is non-zero.

    Returns:
        (N, L, 3) position (m) and (N, L, 3) velocity (m/s).
    """
    phase = phase.unsqueeze(-1)
    # a leg in stance has phase 0 and may have duration 0; the curve is flat there
    # anyway, so the clamp only keeps the division finite
    inverse_duration = 1.0 / duration.clamp_min(1e-6).unsqueeze(-1)

    position = _bezier(start, end, phase)
    velocity = _bezier_rate(start, end, phase) * inverse_duration

    apex = torch.maximum(start[..., 2], end[..., 2]) + apex_clearance
    rising = phase[..., 0] < 0.5
    # each half of the arc runs its own Bezier over a phase of twice the rate
    half_start = torch.where(rising, start[..., 2], apex)
    half_end = torch.where(rising, apex, end[..., 2])
    half_phase = torch.where(rising, phase[..., 0] * 2.0, phase[..., 0] * 2.0 - 1.0)
    height = _bezier(half_start, half_end, half_phase)
    height_rate = _bezier_rate(half_start, half_end, half_phase) * 2.0 * inverse_duration[..., 0]

    position = torch.cat([position[..., :2], height.unsqueeze(-1)], dim=-1)
    velocity = torch.cat([velocity[..., :2], height_rate.unsqueeze(-1)], dim=-1)
    return position, velocity
