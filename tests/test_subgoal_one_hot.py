"""Current-subgoal one-hot preprocessing and RCO model wiring."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from ltl.logic import Assignment, FrozenAssignment
from model.model import infer_model_safety_shapes
from preprocessing.preprocessing import preprocess_current_subgoal, preprocess_obss


def _fake_safety_state_dict(*, feat_dim=64, env_net_layers=None, actor_hidden=None, action_dim=2):
    env_net_layers = env_net_layers or [128, 96]
    actor_hidden = actor_hidden or [64, 64, 64]
    embedding_dim = env_net_layers[-1]
    state: dict[str, torch.Tensor] = {}
    in_dim = feat_dim
    for i, out_dim in enumerate(env_net_layers):
        state[f"env_net.mlp.{i * 2}.weight"] = torch.zeros(out_dim, in_dim)
        in_dim = out_dim
    state["actor.enc.0.weight"] = torch.zeros(actor_hidden[0], embedding_dim)
    prev = actor_hidden[0]
    for j, h in enumerate(actor_hidden[1:], start=1):
        state[f"actor.enc.{j * 2}.weight"] = torch.zeros(h, prev)
        prev = h
    state["actor.mu.0.weight"] = torch.zeros(action_dim, prev)
    state["actor.std.0.weight"] = torch.zeros(action_dim, prev)
    state["actor.epsilon_prob.0.weight"] = torch.zeros(1, prev)
    state["critic.0.weight"] = torch.zeros(64, embedding_dim)
    state["cost_critic.0.weight"] = torch.zeros(64, embedding_dim)
    state["lagrangian_net.0.weight"] = torch.zeros(64, embedding_dim)
    return state


def test_preprocess_current_subgoal_reach_only():
    props = ["surface_0", "entrapped_0"]
    reach = frozenset([Assignment.single_proposition("entrapped_0", props).to_frozen()])
    avoid = frozenset()
    obss = [{"goal": [(reach, avoid)], "features": [0.0], "propositions": []}]
    tensor = preprocess_current_subgoal(obss, props)
    assert tensor.shape == (1, 4)
    assert tensor[0, props.index("entrapped_0")] == 1.0
    assert tensor[0, props.index("surface_0")] == 0.0
    assert tensor[0, 2:].sum() == 0.0


def test_preprocess_obss_includes_current_subgoal():
    props = ["a", "b"]
    reach = frozenset([FrozenAssignment({"a": True, "b": False})])
    avoid = frozenset([FrozenAssignment({"a": False, "b": True})])
    obss = [{"goal": [(reach, avoid)], "features": [[1.0]], "propositions": ["a"]}]
    batch = preprocess_obss(obss, props)
    assert batch.current_subgoal.shape == (1, 4)
    assert batch.current_subgoal[0, 0] == 1.0
    assert batch.current_subgoal[0, 3] == 1.0


def test_infer_model_safety_shapes_with_subgoal_one_hot():
    state = _fake_safety_state_dict(feat_dim=64, env_net_layers=[128, 96])
    subgoal_dim = 6  # 3 props × 2
    state["actor.enc.0.weight"] = torch.zeros(64, 96 + subgoal_dim)
    state["critic.0.weight"] = torch.zeros(64, 96 + subgoal_dim)
    state["cost_critic.0.weight"] = torch.zeros(64, 96 + subgoal_dim)
    state["lagrangian_net.0.weight"] = torch.zeros(64, 96 + subgoal_dim)
    shapes = infer_model_safety_shapes(
        state, num_propositions=3, use_subgoal_one_hot=True,
    )
    assert shapes["use_env_net"] is True
    assert shapes["embedding_dim"] == 96 + subgoal_dim
    assert shapes["subgoal_dim"] == subgoal_dim
    assert shapes["feature_dim"] == 64
