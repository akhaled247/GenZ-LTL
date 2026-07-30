"""Checkpoint-aware ModelSafety shape inference (no MuJoCo)."""
from __future__ import annotations

import torch

from model.model import infer_model_safety_shapes


def _fake_safety_state_dict(
    *,
    feat_dim: int = 64,
    env_net_layers: list[int] | None = None,
    actor_hidden: list[int] | None = None,
    action_dim: int = 2,
) -> dict[str, torch.Tensor]:
    env_net_layers = env_net_layers or [128, 96]
    actor_hidden = actor_hidden or [64, 64, 64]
    embedding_dim = env_net_layers[-1] if env_net_layers else feat_dim

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


def test_infer_model_safety_shapes_env_net_96_embedding():
    state = _fake_safety_state_dict(feat_dim=64, env_net_layers=[128, 96])
    shapes = infer_model_safety_shapes(state)
    assert shapes["feat_dim"] == 64
    assert shapes["embedding_dim"] == 96
    assert shapes["env_net_layers"] == [128, 96]
    assert shapes["action_dim"] == 2
    assert shapes["actor_hidden"] == [64, 64, 64]
