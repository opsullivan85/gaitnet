"""The parts of the planner's contract that the environment has to agree on: which robot,
the foothold grid the terrain is scanned on, and the foothold rules. A trained policy's
bundle records the same values, see `gaitnet_core.bundle`.

Terms read these from `env.cfg.gaitnet` at run time rather than through their own
params, so one override changes every term. The foothold scanners' ray patterns are built
from the default grid when the scene cfg is created; a different grid needs scanners to
match (see `gaitnet_sim.env.scene.foothold_scanner_cfg`), and `RobotIO` checks that they do.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.planner import FootholdRules
from gaitnet_core.robot_spec import ROBOTS, RobotSpec


@configclass
class GaitNetCfg:
    robot: str = "go1"
    """Key of `gaitnet_core.robot_spec.ROBOTS`."""

    grid_resolution: float = 0.015
    """Foothold cell size (m)."""
    grid_size: tuple[int, int] = (25, 25)
    """Foothold cells per leg along (x, y)."""
    grid_border: int = 3
    """Extra terrain cells scanned on each side, context for the edge margin."""
    grid_center: tuple[float, float] = (0.0, 0.08)
    """(x, y) of a left leg's grid centre from its hip, in the hip's yaw frame (m); right legs
    mirror y. 0.08 m outboard is the Go1's abduction link, where a vertical leg's foot is."""

    step_threshold: float = 0.02
    """Height difference between neighbouring cells that counts as an edge (m)."""
    edge_margin: int = 2
    """Cells within this many cells of an edge are not valid footholds."""
    min_stance_after_step: int = 2
    """A leg may only lift off if this many legs stay in stance."""

    def robot_spec(self) -> RobotSpec:
        return ROBOTS[self.robot]

    def foothold_grid(self) -> FootholdGrid:
        return FootholdGrid(
            resolution=self.grid_resolution,
            size=tuple(self.grid_size),
            border=self.grid_border,
            center=tuple(self.grid_center),
        )

    def foothold_rules(self) -> FootholdRules:
        return FootholdRules(
            step_threshold=self.step_threshold,
            edge_margin=self.edge_margin,
            min_stance_after_step=self.min_stance_after_step,
        )
