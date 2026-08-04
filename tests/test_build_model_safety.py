"""Checkpoint-aware ModelSafety shape inference (no MuJoCo)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from gymnasium import spaces

from config.model_config import zones_safety
from model.model import build_model_safety, infer_model_safety_shapes


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
        state[f"env_net.mlp.{i * 2}.bias"] = torch.zeros(out_dim)
        in_dim = out_dim

    state["actor.enc.0.weight"] = torch.zeros(actor_hidden[0], embedding_dim)
    state["actor.enc.0.bias"] = torch.zeros(actor_hidden[0])
    prev = actor_hidden[0]
    for j, h in enumerate(actor_hidden[1:], start=1):
        state[f"actor.enc.{j * 2}.weight"] = torch.zeros(h, prev)
        state[f"actor.enc.{j * 2}.bias"] = torch.zeros(h)
        prev = h

    state["actor.mu.0.weight"] = torch.zeros(action_dim, prev)
    state["actor.mu.0.bias"] = torch.zeros(action_dim)
    state["actor.std.0.weight"] = torch.zeros(action_dim, prev)
    state["actor.std.0.bias"] = torch.zeros(action_dim)
    state["actor.epsilon_prob.0.weight"] = torch.zeros(1, prev)
    state["actor.epsilon_prob.0.bias"] = torch.zeros(1)

    def _mlp_block(prefix: str, in_size: int, layers: list[int]):
        dim = in_size
        for i, out in enumerate(layers):
            state[f"{prefix}.{i * 2}.weight"] = torch.zeros(out, dim)
            state[f"{prefix}.{i * 2}.bias"] = torch.zeros(out)
            dim = out
        state[f"{prefix}.{len(layers) * 2}.weight"] = torch.zeros(1, dim)
        state[f"{prefix}.{len(layers) * 2}.bias"] = torch.zeros(1)

    _mlp_block("critic", embedding_dim, [64, 64])
    _mlp_block("cost_critic", embedding_dim, [64, 64])
    _mlp_block("lagrangian_net", embedding_dim, [64, 64])
    return state


def test_infer_model_safety_shapes_env_net_96_embedding():
    state = _fake_safety_state_dict(feat_dim=64, env_net_layers=[128, 96])
    shapes = infer_model_safety_shapes(state)
    assert shapes["feat_dim"] == 64
    assert shapes["embedding_dim"] == 96
    assert shapes["feature_dim"] == 64
    assert shapes["use_env_net"] is True
    assert shapes["env_net_layers"] == [128, 96]
    assert shapes["action_dim"] == 2
    assert shapes["actor_hidden"] == [64, 64, 64]


def test_infer_model_safety_shapes_skips_env_net_when_output_mismatch():
    state = _fake_safety_state_dict(feat_dim=96, env_net_layers=[128, 64])
    state["actor.enc.0.weight"] = torch.zeros(64, 96)
    state["critic.0.weight"] = torch.zeros(64, 96)
    state["cost_critic.0.weight"] = torch.zeros(64, 96)
    state["lagrangian_net.0.weight"] = torch.zeros(64, 96)
    shapes = infer_model_safety_shapes(state)
    assert shapes["use_env_net"] is False
    assert shapes["feature_dim"] == 96
    assert shapes["env_net_layers"] is None


def test_build_model_safety_ignores_legacy_ltl_net_keys():
    """Zone/RCO ckpts may still contain unused ltl_net.* weights."""
    feat_dim = 48
    state = _fake_safety_state_dict(feat_dim=feat_dim, env_net_layers=[128, 64])
    state["ltl_net.embedding.weight"] = torch.zeros(10, 16)
    state["ltl_net.rnn.weight_ih_l0"] = torch.zeros(48, 16)

    env = MagicMock()
    env.observation_space = spaces.Dict({
        "features": spaces.Box(-np.inf, np.inf, (feat_dim,), dtype=np.float32),
    })
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    env.get_propositions = lambda: ["blue", "green", "yellow", "magenta"]
    task = MagicMock()
    task.lidar_conf.num_bins = 16
    env.unwrapped.task = task
    env.spec = MagicMock()
    env.spec.id = "PointLtlSafety2-v0"

    model = build_model_safety(
        env,
        {"model_state": state},
        zones_safety,
        deploy_meta={
            "raw_feature_dim": feat_dim,
            "use_env_net": True,
            "actor_input_dim": 64,
            "feat_recipe": "zone_compat",
        },
    )
    assert model is not None
    assert model.ltl_net is None
