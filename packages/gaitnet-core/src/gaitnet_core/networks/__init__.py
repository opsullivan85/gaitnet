"""Networks, looked up by class name so a checkpoint bundle can rebuild them.

A scoring network is called as `network(state, candidates, terrain)` and returns
`gaitnet_core.selection.Scores`. Networks that read the terrain patches set
`uses_terrain` and take the foothold grid in their config.
"""

from __future__ import annotations

import torch.nn as nn

from gaitnet_core.networks.candidate_scorer import CANDIDATE_FEATURES, CandidateScorer
from gaitnet_core.networks.critic import Critic
from gaitnet_core.networks.spatial import DenseSpatialCNN

NETWORKS: dict[str, type[nn.Module]] = {
    "CandidateScorer": CandidateScorer,
    "DenseSpatialCNN": DenseSpatialCNN,
    "Critic": Critic,
}


def build_network(name: str, config: dict) -> nn.Module:
    return NETWORKS[name](**config)


def uses_terrain(network: nn.Module) -> bool:
    return bool(getattr(network, "uses_terrain", False))


__all__ = [
    "CANDIDATE_FEATURES",
    "CandidateScorer",
    "Critic",
    "DenseSpatialCNN",
    "NETWORKS",
    "build_network",
    "uses_terrain",
]
