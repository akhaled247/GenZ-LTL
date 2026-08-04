"""Tests for SAR deploy helpers (no MuJoCo)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from envs.sar_deploy import (
    ma_agent_cost_walls,
    ma_episode_success,
    ma_episode_violation,
    ma_step_saw_walls,
    resolve_sar_feat_shape,
    sar_preprocess_for_deploy,
)
from model.model import infer_model_safety_shapes
from tests.test_build_model_safety import _fake_safety_state_dict


def test_resolve_sar_feat_shape_prefers_raw_feature_dim():
    model = SimpleNamespace(raw_feature_dim=96, input_feat_dim=64)
    assert resolve_sar_feat_shape(model, lidar_bins=10) == (96,)


def test_resolve_sar_feat_shape_legacy_input_feat_dim():
    model = SimpleNamespace(input_feat_dim=96)
    assert resolve_sar_feat_shape(model, lidar_bins=10) == (96,)


def test_sar_preprocess_for_deploy_passes_feat_shape():
    env = MagicMock()
    env.spec.id = "PointLTL1MASAR1WC-v0"
    env.pre_process_obs_sar.return_value = np.zeros(96, dtype=np.float32)
    model = SimpleNamespace(raw_feature_dim=96, feat_recipe="legacy_v0")
    task = SimpleNamespace(lidar_conf=SimpleNamespace(num_bins=10))
    with patch("envs.seq_wrapper.sar_task", return_value=task):
        out = sar_preprocess_for_deploy(env, model, "reach", "avoid", agent_idx=1)
    env.pre_process_obs_sar.assert_called_once_with(
        "reach",
        "avoid",
        agent_idx=1,
        feat_shape=(96,),
        allow_legacy_padding=True,
    )
    assert out.shape == (96,)


def test_sar_preprocess_for_deploy_routes_pointltl_safety_to_zones():
    env = MagicMock()
    env.spec.id = "PointLtlSafety2-v0"
    env.pre_process_obs_zones.return_value = np.zeros(48, dtype=np.float32)
    model = SimpleNamespace(raw_feature_dim=48, feat_recipe="sar_v1")
    out = sar_preprocess_for_deploy(env, model, "reach", "avoid", agent_idx=0)
    env.pre_process_obs_zones.assert_called_once_with("reach", "avoid")
    env.pre_process_obs_sar.assert_not_called()
    assert out.shape == (48,)


def test_ma_episode_success_from_goal_met():
    agents = ["agent_0", "agent_1"]
    assert ma_episode_success({"goal_met": True}, agents)
    assert ma_episode_success({"agent_0": {"goal_met": True}}, agents)
    assert ma_episode_success({"success": True}, agents)
    assert not ma_episode_success({"violation": True}, agents)


def test_ma_episode_violation():
    agents = ["agent_0", "agent_1"]
    assert ma_episode_violation({"violation": True}, agents)
    assert ma_episode_violation({"agent_1": {"violation": True}}, agents)
    assert not ma_episode_violation({"success": True}, agents)


def test_ma_episode_violation_from_walls_cost():
    agents = ["agent_0", "agent_1"]
    assert ma_episode_violation({"agent_0": {"cost_walls": 1.0}}, agents)
    assert ma_episode_violation({"cost": 1.0}, agents)
    assert ma_episode_violation({"propositions": ["walls"]}, agents)
    assert ma_episode_violation({"success": True}, agents, saw_walls=True)
    assert ma_episode_violation({}, agents, saw_walls=True)
    assert not ma_episode_violation({"agent_0": {"cost_walls": 0.0}}, agents)


def test_ma_step_saw_walls():
    agents = ["agent_0", "agent_1"]
    assert not ma_step_saw_walls({}, agents)
    assert ma_step_saw_walls({"agent_0": {"cost_walls": 1.0}}, agents)
    assert not ma_step_saw_walls({"agent_1": {"cost_walls": 0.0}}, agents)


def test_ma_agent_cost_walls():
    agents = ["agent_0", "agent_1"]
    assert ma_agent_cost_walls({}, agents) == {"agent_0": 0.0, "agent_1": 0.0}
    assert ma_agent_cost_walls(
        {"agent_0": {"cost_walls": 1.0}, "agent_1": {"cost_collision": 1.0}},
        agents,
    ) == {"agent_0": 1.0, "agent_1": 0.0}


def test_classify_wall_geom_name():
    from envs.sar_deploy import classify_wall_geom_name, expected_pseudo_lidar

    assert classify_wall_geom_name("wall0") == "interior"
    assert classify_wall_geom_name("wall3") == "interior"
    assert classify_wall_geom_name("ltl_wall0") == "arena_ltl"
    assert classify_wall_geom_name("ltl_walls2") == "arena_ltl"
    assert classify_wall_geom_name("building0_ltl_walls0") == "building_perimeter"
    assert abs(expected_pseudo_lidar(0.175) - float(__import__("numpy").exp(-0.5 * 0.175))) < 1e-9
    assert abs(expected_pseudo_lidar(2.0) - float(__import__("numpy").exp(-1.0))) < 1e-9


def test_infer_shapes_raw_feature_dim_matches_preprocess_for_legacy_checkpoint():
    state = _fake_safety_state_dict(feat_dim=96, env_net_layers=[128, 64])
    state["actor.enc.0.weight"] = __import__("torch").zeros(64, 96)
    state["critic.0.weight"] = __import__("torch").zeros(64, 96)
    state["cost_critic.0.weight"] = __import__("torch").zeros(64, 96)
    state["lagrangian_net.0.weight"] = __import__("torch").zeros(64, 96)
    shapes = infer_model_safety_shapes(state)
    assert shapes["use_env_net"] is False
    assert shapes["feature_dim"] == 96
