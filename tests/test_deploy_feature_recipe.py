"""Tests for canonical deploy feature recipe."""
from __future__ import annotations

import pytest

from deploy.feature_recipe import (
    FEAT_RECIPE_LEGACY_V0,
    FEAT_RECIPE_SAR_V1,
    FEAT_RECIPE_ZONE_COMPAT,
    apply_zone_compat_deploy_meta,
    canonical_raw_dim,
    ensure_sar_v1_indep_lidars,
    infer_feat_recipe,
    is_zone_safety_train_env,
    resolve_zone_compat,
)


def test_infer_feat_recipe_sar_v1_at_canonical_dim():
    lidar_bins = 16
    raw = canonical_raw_dim(lidar_bins)
    assert infer_feat_recipe(raw, lidar_bins) == FEAT_RECIPE_SAR_V1


def test_infer_feat_recipe_legacy_when_mismatch():
    lidar_bins = 16
    assert infer_feat_recipe(96, lidar_bins) == FEAT_RECIPE_LEGACY_V0


def test_infer_feat_recipe_sar_v1_with_walls_lidar():
    lidar_bins = 16
    raw = canonical_raw_dim(lidar_bins, include_walls_lidar=True)
    assert raw == 80
    assert infer_feat_recipe(raw, lidar_bins) == FEAT_RECIPE_SAR_V1


def test_infer_feat_recipe_zone_compat_at_48():
    lidar_bins = 16
    raw = canonical_raw_dim(lidar_bins, zone_compat=True)
    assert raw == 48
    assert infer_feat_recipe(raw, lidar_bins) == FEAT_RECIPE_ZONE_COMPAT


def test_resolve_zone_compat_cli_and_recipe():
    assert is_zone_safety_train_env("PointLtlSafety2-v0")
    assert not is_zone_safety_train_env("PointLTL1MASAR1WC-v0")
    # SAR: CLI alone does not flip packing (would mismatch 64/80-d weights).
    assert resolve_zone_compat("PointLTL1MASAR1WC-v0", True) is False
    # SAR zc48 checkpoint: feat_recipe wins.
    assert resolve_zone_compat(
        "PointLTL1MASAR1WC-v0", False, feat_recipe=FEAT_RECIPE_ZONE_COMPAT,
    ) is True
    assert resolve_zone_compat("PointLtlSafety2-v0", True) is True
    assert resolve_zone_compat("PointLtlSafety2-v0", False) is False


def test_ensure_sar_v1_preserves_intentional_zone_compat():
    meta = ensure_sar_v1_indep_lidars(
        {
            "train_env": "PointLTL1MASAR1WC-v0",
            "raw_feature_dim": 48,
            "feat_recipe": FEAT_RECIPE_ZONE_COMPAT,
        },
        lidar_bins=16,
    )
    assert meta["feat_recipe"] == FEAT_RECIPE_ZONE_COMPAT
    assert meta["raw_feature_dim"] == 48


def test_ensure_sar_v1_restores_walls_on_stale_zone_recipe():
    meta = ensure_sar_v1_indep_lidars(
        {
            "train_env": "PointLTL1MASAR1WC-v0",
            "raw_feature_dim": 80,
            "feat_recipe": FEAT_RECIPE_ZONE_COMPAT,
        },
        lidar_bins=16,
    )
    assert meta["feat_recipe"] == FEAT_RECIPE_SAR_V1
    assert meta["raw_feature_dim"] == 80


def test_apply_zone_compat_noop_recipe_for_sar_v1_train():
    meta = apply_zone_compat_deploy_meta(
        {
            "train_env": "PointLTL1MASAR1WC-v0",
            "raw_feature_dim": 80,
            "feat_recipe": FEAT_RECIPE_SAR_V1,
        },
        lidar_bins=16,
    )
    assert meta["feat_recipe"] == FEAT_RECIPE_SAR_V1
    assert meta["raw_feature_dim"] == 80


def test_apply_zone_compat_keeps_sar_zc48():
    meta = apply_zone_compat_deploy_meta(
        {
            "train_env": "PointLTL1MASAR1WC-v0",
            "raw_feature_dim": 48,
            "feat_recipe": FEAT_RECIPE_ZONE_COMPAT,
        },
        lidar_bins=16,
    )
    assert meta["feat_recipe"] == FEAT_RECIPE_ZONE_COMPAT
    assert meta["raw_feature_dim"] == 48


def test_apply_zone_compat_for_zones_train():
    meta = apply_zone_compat_deploy_meta(
        {
            "train_env": "PointLtlSafety2-v0",
            "raw_feature_dim": 48,
            "feat_recipe": FEAT_RECIPE_SAR_V1,
        },
        lidar_bins=16,
    )
    assert meta["feat_recipe"] == FEAT_RECIPE_ZONE_COMPAT
    assert meta["raw_feature_dim"] == 48


def test_pre_process_obs_sar_rejects_padding_without_legacy():
    pytest.importorskip("numpy")
    import numpy as np
    from unittest.mock import MagicMock

    from envs.seq_wrapper import pre_process_obs_sar, sar_agent_obs_keys

    env = MagicMock()
    lidar_dim = 16
    task = MagicMock()
    task.lidar_conf.num_bins = lidar_dim
    task.agent_num = 1
    env.unwrapped.task = task
    keys = sar_agent_obs_keys(0)
    dims = [3, 3, 3, 3, 4]
    agent_obs = {k: np.zeros(d, dtype=np.float32) for k, d in zip(keys, dims)}
    agent_obs["terracotta_buildings_lidar_0"] = np.zeros(lidar_dim, dtype=np.float32)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("envs.seq_wrapper.sar_agent_obs", lambda _e, _i=0: agent_obs)
        mp.setattr("envs.seq_wrapper.sar_task", lambda _e: task)
        with pytest.raises(ValueError, match="legacy_v0"):
            pre_process_obs_sar(
                env, keys, frozenset(), frozenset(), (96,),
            )
