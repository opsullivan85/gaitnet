"""Networks, looked up by class name so a checkpoint bundle can rebuild them."""

from __future__ import annotations

import torch.nn as nn

from gaitnet_core.networks.candidate_scorer import CANDIDATE_FEATURES, CandidateScorer
from gaitnet_core.networks.critic import Critic

NETWORKS: dict[str, type[nn.Module]] = {
    "CandidateScorer": CandidateScorer,
    "Critic": Critic,
}


def build_network(name: str, config: dict) -> nn.Module:
    return NETWORKS[name](**config)


__all__ = ["CANDIDATE_FEATURES", "CandidateScorer", "Critic", "NETWORKS", "build_network"]
