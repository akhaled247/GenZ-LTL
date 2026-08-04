"""Tests for canonical deploy feature recipe."""
from __future__ import annotations

import pytest

from deploy.feature_recipe import (
    FEAT_RECIPE_LEGACY_V0,
    FEAT_RECIPE_SAR_V1,
    FEAT_RECIPE_ZONE_COMPAT,
    canonical_raw_dim,
    infer_feat_recipe,
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
    agent_obs = {k: np.zeros(3, dtype=np.float32) for k in keys}
    agent_obs["terracotta_buildings_lidar_0"] = np.zeros(lidar_dim, dtype=np.float32)
    agent_obs["surface_casualtys_lidar_0"] = np.zeros(lidar_dim, dtype=np.float32)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "envs.seq_wrapper.sar_agent_obs",
            lambda _env, _idx=0: agent_obs,
        )
        mp.setattr(
            "envs.seq_wrapper.sar_task",
            lambda _env: task,
        )
        with pytest.raises(ValueError, match="legacy_v0"):
            pre_process_obs_sar(
                env,
                keys,
                frozenset(),
                frozenset(),
                (96,),
                allow_legacy_padding=False,
            )
