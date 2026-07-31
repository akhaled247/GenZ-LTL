from typing import Any

import torch
from itertools import product

from torch_ac.utils import DictList
import numpy as np

from ltl.automata import LDBASequence
from ltl.logic import FrozenAssignment, Assignment
from preprocessing.vocab import VOCAB
from preprocessing.batched_sequences import BatchedReachAvoidSequences, ReachAvoidSet


def preprocess_obss(obss: list[dict[str, Any]], propositions: list[str], device=None) -> DictList:
    features = []
    seqs = []
    epsilon_mask = []
    for obs in obss:
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
        "seq": BatchedReachAvoidSequences([preprocess_sequence(seq, propositions) for seq in seqs], device=device),
        "current_subgoal": preprocess_current_subgoal(obss, propositions, device=device),
        "epsilon_mask": torch.tensor(epsilon_mask, dtype=torch.bool).to(device),
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)


def preprocess_current_subgoal(
        obss: list[dict[str, Any]],
        propositions: list[str],
        device=None,
) -> torch.Tensor:
    """One-hot over props for the active reach/avoid pair only (goal[0])."""
    rows = []
    for obs in obss:
        goal = obs.get("goal") or []
        if goal:
            reach, avoid = goal[0]
        else:
            reach, avoid = frozenset(), frozenset()
        rows.append(
            preprocess_assignments(reach, propositions)
            + preprocess_assignments(avoid, propositions)
        )
    return torch.tensor(np.array(rows), dtype=torch.float).to(device)


# one-hot encoding for assignments
def preprocess_sequence(seq: LDBASequence, propositions: list[str]) -> list[ReachAvoidSet]:
    return [(preprocess_assignments(reach, propositions), preprocess_assignments(avoid, propositions)) for reach, avoid in seq]

def preprocess_assignments(assignments: frozenset[FrozenAssignment] | type(LDBASequence.EPSILON),
                           propositions: list[str]) -> list[int]:
    """
    one-hot encoding of assignments
    """
    if assignments == LDBASequence.EPSILON:
        return []
    if not assignments:
        return [0.0] * len(propositions)
    assignments_arr = np.array([item for a in assignments for item in a.to_string()])
    props = np.array(propositions)
    assignments_emb = np.isin(props, assignments_arr).astype(float)
    return assignments_emb.tolist()
