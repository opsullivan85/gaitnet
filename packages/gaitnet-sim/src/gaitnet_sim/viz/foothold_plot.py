"""The foothold map as a 2x2 figure of per-leg heatmaps, and a `PlannerRuntime` on_plan
callback that writes it to PNGs as the planner runs. Needs matplotlib, not the simulator.

Panels are laid out as seen from above with the robot facing up (FL top left), each one the
leg's grid in its hip yaw frame: forward up, left to the left, the hip at the origin.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gaitnet_core.foothold_map import LOGIT_KINDS, FootholdMap, LogitKind, foothold_map
from gaitnet_core.planner import FootstepPlanner, PlanResult
from gaitnet_core.robot_spec import LEG_NAMES
from gaitnet_core.state import Observation

COLORMAP = "viridis"
UNREACHABLE_RGBA = (0.0, 0.0, 0.0, 0.7)
NEAR_EDGE_RGBA = (0.45, 0.45, 0.45, 0.6)
CHOSEN_COLOR = "#e8175d"
MUTED_COLOR = "#8a8a8a"


def color_limits(fmap: FootholdMap, index: int, kind: LogitKind) -> tuple[float, float]:
    """The colour scale's (low, high) for robot `index`: the range of its logits over the
    cells the terrain allows (every cell if none), and the no-op for corrected logits, which
    share its scale. Masked cells can score far outside it; they saturate."""
    logits = fmap.logits(kind)[index]
    ok = fmap.terrain_ok[index]
    values = logits[ok] if ok.any() else logits.flatten()
    values = values[torch.isfinite(values)]
    if kind == "corrected":
        values = torch.cat([values, fmap.noop_logit[index].reshape(1)])
    if values.numel() == 0:
        return -1.0, 1.0
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        low, high = low - 0.5, high + 0.5
    return low, high


def _as_seen_from_above(cells: np.ndarray) -> np.ndarray:
    """(X, Y, ...) grid-indexed array -> image rows running forward-to-back, columns
    left-to-right."""
    return cells[::-1, ::-1]


def draw_foothold_map(
    figure: Figure, fmap: FootholdMap, index: int = 0, kind: LogitKind = "raw", tick: int | None = None
) -> None:
    """Draw robot `index` of `fmap` onto `figure`, replacing whatever was there."""
    figure.clear()
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    grid = fmap.grid
    half_x, half_y = grid.half_extent
    pad = grid.resolution / 2
    # left > right: y (left) increases to the left
    extent = (half_y + pad, -half_y - pad, -half_x - pad, half_x + pad)

    logits = fmap.logits(kind)[index].float().cpu().numpy()
    reachable = fmap.reachable[index].cpu().numpy()
    away = fmap.away_from_edges[index].cpu().numpy()
    terrain_ok = reachable & away
    eligible = fmap.eligible[index].cpu().numpy()
    probabilities = fmap.step_probabilities()[index].cpu().numpy()
    best = fmap.best_cells()[index].cpu().numpy()
    centers = grid.cell_centers().numpy()
    chosen_leg = int(fmap.leg[index])
    target = fmap.target[index].cpu().numpy()
    low, high = color_limits(fmap, index, kind)

    image = None
    for leg, ax in enumerate(axes.flat):
        image = ax.imshow(
            _as_seen_from_above(logits[leg]), cmap=COLORMAP, vmin=low, vmax=high, extent=extent,
            interpolation="nearest",
        )
        veil = np.zeros((*grid.size, 4))
        veil[~away[leg]] = NEAR_EDGE_RGBA
        veil[~reachable[leg]] = UNREACHABLE_RGBA
        ax.imshow(_as_seen_from_above(veil), extent=extent, interpolation="nearest")

        ax.plot(0.0, 0.0, "+", color="white", markersize=8, markeredgewidth=1)
        if terrain_ok[leg].any():
            best_x, best_y = centers[best[leg, 0], best[leg, 1]]
            ax.plot(best_y, best_x, "o", markerfacecolor="none", markeredgecolor="white", markersize=9, markeredgewidth=1.5)
        if leg == chosen_leg:
            ax.plot(target[1], target[0], "X", markerfacecolor=CHOSEN_COLOR, markeredgecolor="white", markersize=13)

        name = LEG_NAMES[leg] if leg < len(LEG_NAMES) else f"leg {leg}"
        if eligible[leg] and terrain_ok[leg].any():
            ax.set_title(f"{name}   p(step) {probabilities[leg]:.2f}", fontsize=10)
        elif eligible[leg]:
            ax.set_title(f"{name}   no foothold", fontsize=10, color=MUTED_COLOR)
        else:
            ax.set_title(f"{name}   can't lift off", fontsize=10, color=MUTED_COLOR)
            for spine in ax.spines.values():
                spine.set_linestyle((0, (4, 3)))
                spine.set_edgecolor(MUTED_COLOR)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("y, left of the hip (m)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("x, ahead of the hip (m)", fontsize=8)

    bar = figure.colorbar(image, ax=axes, shrink=0.85, extend="both", pad=0.02)
    bar.set_label(f"{kind} step logit", fontsize=9)
    bar.ax.tick_params(labelsize=7)
    if kind == "corrected":
        noop = float(fmap.noop_logit[index])
        bar.ax.axhline(noop, color=CHOSEN_COLOR, linewidth=2)
        bar.ax.annotate(
            "no-op", xy=(0, noop), xycoords=("axes fraction", "data"), xytext=(-3, 0),
            textcoords="offset points", ha="right", va="center", fontsize=8,
        )

    parts = [f"robot {int(fmap.robot_ids[index])}"]
    if tick is not None:
        parts.append(f"tick {tick}")
    parts.append(f"p(no-op) {probabilities[-1]:.2f}")
    if chosen_leg >= 0:
        parts.append(f"steps {LEG_NAMES[chosen_leg]}, {float(fmap.step_duration[index]) * 1e3:.0f} ms swing")
    else:
        parts.append("holds")
    figure.suptitle("   ·   ".join(parts), fontsize=11)
    figure.legend(
        handles=[
            Patch(facecolor=UNREACHABLE_RGBA, label="out of reach"),
            Patch(facecolor=NEAR_EDGE_RGBA, label="near an edge"),
            Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor="black", label="leg's best"),
            Line2D([], [], linestyle="none", marker="X", markerfacecolor=CHOSEN_COLOR, markeredgecolor="white", markersize=9, label="chosen"),
            Line2D([], [], linestyle="none", marker="+", color="black", label="hip"),
        ],
        loc="outside lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )


class FootholdPlot:
    """Writes the foothold maps of `robot_ids` to `out_dir` as the planner runs:
    `robot<id>.png`, replaced every `every` ticks (a viewer that reloads on change shows it
    live), and with `keep_frames` a copy of each in `frames/`, for stepping through or
    making a video. Drawing takes a good fraction of a second per robot."""

    def __init__(
        self,
        planner: FootstepPlanner,
        robot_ids: Sequence[int],
        out_dir: str | Path,
        kind: LogitKind = "raw",
        every: int = 1,
        keep_frames: bool = False,
        dpi: int = 100,
    ):
        if kind not in LOGIT_KINDS:
            raise ValueError(f"unknown logit kind {kind!r}, expected one of {LOGIT_KINDS}")
        self.planner = planner
        self.robot_ids = list(robot_ids)
        self.out_dir = Path(out_dir)
        self.kind = kind
        self.every = max(1, every)
        self.keep_frames = keep_frames
        self.dpi = dpi
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if keep_frames:
            (self.out_dir / "frames").mkdir(exist_ok=True)
        self.figure = Figure(figsize=(8.0, 7.2), layout="constrained")
        self.ticks = 0

    def __call__(self, plan: PlanResult, observation: Observation) -> None:
        tick = self.ticks
        self.ticks += 1
        if tick % self.every:
            return
        fmap = foothold_map(self.planner, plan, observation, self.robot_ids)
        for index, robot in enumerate(self.robot_ids):
            draw_foothold_map(self.figure, fmap, index, self.kind, tick)
            latest = self.out_dir / f"robot{robot}.png"
            # written aside and moved into place, so a viewer never reads half a file
            partial = self.out_dir / f".robot{robot}.partial.png"
            self.figure.savefig(partial, dpi=self.dpi)
            os.replace(partial, latest)
            if self.keep_frames:
                shutil.copyfile(latest, self.out_dir / "frames" / f"robot{robot}_{tick:06d}.png")
