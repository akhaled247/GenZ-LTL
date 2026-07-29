from typing import Any

import torch

from torch_ac.utils import DictList
from preprocessing.ma_layout import flatten_ma_obs, unflatten_ma_actions
from preprocessing.batched_sequences import BatchedReachAvoidSequences
from ltl.automata import LDBASequence
from ltl.logic import Assignment


def preprocess_obss_ma(
        obss: list[dict[str, Any]],
        propositions: list[str],
        agent_ids: list[int],
        device=None,
) -> DictList:
    from preprocessing.preprocessing import preprocess_features, preprocess_sequence

    features = []
    seqs = []
    epsilon_mask = []
    for obs, agent_id in zip(obss, agent_ids):
        features.append(obs["features"])
        seqs.append(list(reversed(obs["goal"])))
    for seq, obs in zip(seqs, obss):
        epsilon_enabled = seq[-1][0] == LDBASequence.EPSILON
        if epsilon_enabled and len(seq) > 1:
            next_avoid = seq[-2][1]
            assignment = Assignment({p: (p in obs['propositions']) for p in propositions}).to_frozen()
            epsilon_enabled &= assignment not in next_avoid
        epsilon_mask.append(epsilon_enabled)
    return DictList({
        "features": preprocess_features(features, device=device),
        "seq": BatchedReachAvoidSequences(
            [preprocess_sequence(seq, propositions) for seq in seqs], device=device,
        ),
        "epsilon_mask": torch.tensor(epsilon_mask, dtype=torch.bool).to(device),
        "agent_id": torch.tensor(agent_ids, dtype=torch.long).to(device),
    })


# Re-export layout helpers for callers that import from this module.
__all__ = [
    "preprocess_obss_ma",
    "flatten_ma_obs",
    "unflatten_ma_actions",
]
