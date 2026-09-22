"""Self-contained policy files.

A bundle holds the actor's weights together with everything needed to use them: the
network's class and constructor arguments, the robot, the foothold grid and rules, the
state features in order, the sampler it was trained with, and the feedback observers it
was trained under (the policy saw their nudges, so it should run with them). Loading
checks the manifest against this code, so a policy can't silently run with a different
state layout or grid than it was trained on. Training logs bundles as run artifacts;
deployment loads one from a local file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from gaitnet_core.features import FEATURES, feature_dim
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.networks import NETWORKS, build_network
from gaitnet_core.observers import OBSERVERS, Observer, make_observers
from gaitnet_core.planner import FootholdRules, FootstepPlanner
from gaitnet_core.robot_spec import ROBOTS, RobotSpec
from gaitnet_core.samplers import CandidateSampler, make_sampler

FORMAT_VERSION = 2


class BundleError(ValueError):
    pass


@dataclass
class PolicyBundle:
    actor: nn.Module
    robot: RobotSpec
    grid: FootholdGrid
    features: tuple[str, ...]
    rules: FootholdRules
    train_sampler: dict
    """{"name": ..., **kwargs}, see `gaitnet_core.samplers.make_sampler`"""
    duration_std: float
    extra: dict
    """Free-form metadata: git commit, controller, run id, ..."""
    observers: dict[str, dict] = field(default_factory=dict)
    """{name: kwargs}, see `gaitnet_core.observers.make_observers`"""

    def make_observers(self) -> list[Observer]:
        return make_observers(self.observers)

    def planner(self, sampler: CandidateSampler | None = None) -> FootstepPlanner:
        """A planner for this policy. Defaults to exhaustive (dense) sampling."""
        if sampler is None:
            sampler = make_sampler("dense")
        return FootstepPlanner(
            network=self.actor,
            spec=self.robot,
            grid=self.grid,
            features=self.features,
            sampler=sampler,
            rules=self.rules,
            duration_std=self.duration_std,
        )


def _manifest(bundle: PolicyBundle) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "actor": {"class": type(bundle.actor).__name__, "config": bundle.actor.config},
        "robot": bundle.robot.name,
        "grid": bundle.grid.to_dict(),
        "features": list(bundle.features),
        "rules": asdict(bundle.rules),
        "train_sampler": dict(bundle.train_sampler),
        "duration_std": float(bundle.duration_std),
        "observers": {name: dict(kwargs or {}) for name, kwargs in bundle.observers.items()},
        "extra": dict(bundle.extra),
    }


def save_bundle(path: str | Path, bundle: PolicyBundle) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": _manifest(bundle), "actor_state_dict": bundle.actor.state_dict()}, path)
    return path


def check_manifest(manifest: dict) -> None:
    """Raise BundleError if this code can't run the policy the manifest describes."""
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"bundle format {manifest.get('format_version')}, this code reads {FORMAT_VERSION}")
    if manifest["actor"]["class"] not in NETWORKS:
        raise BundleError(f"unknown network class {manifest['actor']['class']}")
    if manifest["robot"] not in ROBOTS:
        raise BundleError(f"unknown robot {manifest['robot']}")
    unknown = [name for name in manifest["features"] if name not in FEATURES]
    if unknown:
        raise BundleError(f"unknown state features {unknown}")
    num_legs = ROBOTS[manifest["robot"]].num_legs
    expected = feature_dim(manifest["features"], num_legs)
    state_dim = manifest["actor"]["config"].get("state_dim")
    if state_dim is not None and state_dim != expected:
        raise BundleError(f"network expects a {state_dim} dim state, the features give {expected}")
    # networks that read terrain were built for one foothold grid
    network_grid = manifest["actor"]["config"].get("grid")
    if network_grid is not None and FootholdGrid.from_dict(network_grid) != FootholdGrid.from_dict(manifest["grid"]):
        raise BundleError(f"network reads terrain on grid {network_grid}, the bundle's grid is {manifest['grid']}")
    low, high = ROBOTS[manifest["robot"]].swing_duration_range
    network = manifest["actor"]["config"]
    network_low, network_high = network.get("duration_range", (low, high))
    fixed = network.get("fixed_duration")
    if network_low < low or network_high > high or (fixed is not None and not low <= fixed <= high):
        raise BundleError(
            f"network swing durations {fixed if fixed is not None else (network_low, network_high)} are outside"
            f" the robot's range {(low, high)}"
        )
    unknown = [name for name in manifest["observers"] if name not in OBSERVERS]
    if unknown:
        raise BundleError(f"unknown observers {unknown}")


def load_bundle(path: str | Path, map_location: str | torch.device = "cpu") -> PolicyBundle:
    data = torch.load(Path(path), map_location=map_location, weights_only=True)
    manifest = data["manifest"]
    check_manifest(manifest)
    actor = build_network(manifest["actor"]["class"], manifest["actor"]["config"])
    actor.load_state_dict(data["actor_state_dict"])
    actor.to(map_location).eval()
    return PolicyBundle(
        actor=actor,
        robot=ROBOTS[manifest["robot"]],
        grid=FootholdGrid.from_dict(manifest["grid"]),
        features=tuple(manifest["features"]),
        rules=FootholdRules(**manifest["rules"]),
        train_sampler=manifest["train_sampler"],
        duration_std=manifest["duration_std"],
        extra=manifest["extra"],
        observers=manifest["observers"],
    )
