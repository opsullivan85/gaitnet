from __future__ import annotations
import numpy as np
from gaitnet import get_logger
from dataclasses import dataclass, field

logger = get_logger()


@dataclass(frozen=True)
class _Robot:
    num_legs: int = 4
    """Number of legs on the robot"""


robot = _Robot()
"""Robot constants"""


@dataclass(frozen=True)
class _GaitNet:
    num_footstep_options: int = 64
    """Number of footstep options to provide per leg"""
    cspace_dialation: int = 2
    """Number of times to apply max-pooling to the height scan to simulate c-space dialation"""
    valid_height_range: tuple[float, float] = (-0.5, 0)
    """(min, max) valid height range for footstep options.
    Note that these are negative of the values you would expect."""
    valid_swing_duration_range: tuple[float, float] = (0.1, 0.3)
    """(min, max) valid swing duration range for footstep options."""
    robot_state_dim: int = 53
    """Dimension of the robot state input to GaitNet (shared state).
    See `gaitnet.gaitnet.env_cfg.observations_utils` for the layout."""
    max_stance_time_obs: float = 0.5
    """Time since touchdown (s) is clipped to this in observations, since it is otherwise unbounded."""


gait_net = _GaitNet()
"""GaitNet constants"""


@dataclass(frozen=True)
class _FootstepScanner:
    grid_resolution: float = 0.015
    """Grid resolution used in footstep scanner observations"""
    total_robot_features: int = None  # type: ignore set in __post_init__
    """Number of features in footstep scanner observations. Assumes one scanner per leg."""
    grid_size: np.ndarray = field(default_factory=lambda: np.asarray((25, 25), dtype=int))
    """Grid size used in footstep scanner observations"""
    sensor_grid_size: np.ndarray = None  # type: ignore set in __post_init__
    """Grid size of the underlying raycaster sensors used for footstep scanner observations.
    This is larger than `grid_size` to account for c-space dialation."""

    def __post_init__(self):
        object.__setattr__(
            self,
            "total_robot_features",
            robot.num_legs * self.grid_size[0] * self.grid_size[1],
        )
        object.__setattr__(
            self,
            "sensor_grid_size",
            self.grid_size + 2 * gait_net.cspace_dialation,  # account for c-space dialation
        )
        self.grid_size.setflags(write=False)
        self.sensor_grid_size.setflags(write=False)


footstep_scanner = _FootstepScanner()
"""Footstep scanner constants"""

@dataclass(frozen=True)
class _Experiments:
    ablate_swing_duration: bool = False
    """If true, set all swing durations to a constant value for ablation study."""
    constant_swing_duration: float = 0.247
    """Constant swing duration to use if ablate_swing_duration is True."""
    swing_duration_logging: bool = False
    """If true, log swing duration statistics."""
    contact_schedule_logging: bool = False
    """If true, log contact schedule statistics, only for first robot."""


experiments = _Experiments()
"""Experiment constants"""


##### Checks

assert footstep_scanner.grid_size.shape == (2,), "Footstep scanner grid size must be 2D"

NO_STEP = -1  # special value for no step
