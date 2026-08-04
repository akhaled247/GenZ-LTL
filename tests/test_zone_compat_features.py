"""zone_compat SAR feature recipe (48-d Zone-like layout)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from deploy.feature_recipe import (
    FEAT_RECIPE_ZONE_COMPAT,
    apply_zone_compat_deploy_meta,
    canonical_raw_dim,
    infer_feat_recipe,
)
from envs.seq_wrapper import (
    lidar_for_assignments,
    pre_process_obs_sar,
    sar_agent_obs_keys,
    sar_feat_dim,
)
from ltl.logic import FrozenAssignment
from utils.deploy_meta import FEAT_RECIPE_ZONE_COMPAT as META_ZONE_COMPAT


def _mock_sar_env(lidar_dim: int = 4):
    env = MagicMock()
    task = MagicMock()
    task.lidar_conf.num_bins = lidar_dim
    task.agent_num = 1
    env.unwrapped.task = task
    return env, task


def test_sar_feat_dim_zone_compat_is_48_for_16_bins():
    assert sar_feat_dim(16, zone_compat=True) == 48
    assert canonical_raw_dim(16, zone_compat=True) == 48


def test_infer_feat_recipe_zone_compat_at_48():
    assert infer_feat_recipe(48, 16) == FEAT_RECIPE_ZONE_COMPAT
    assert META_ZONE_COMPAT == "zone_compat"


def test_apply_zone_compat_deploy_meta():
    meta = apply_zone_compat_deploy_meta({"raw_feature_dim": 64, "feat_recipe": "sar_v1"})
    assert meta["feat_recipe"] == "zone_compat"
    assert meta["raw_feature_dim"] == 48


def test_zone_compat_entrapped_uses_buildings_only():
    lidar_dim = 4
    building = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64)
    entrapped = np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "terracotta_buildings_lidar_0": building,
        "entrapped_casualtys_lidar_0": entrapped,
    }
    reach = frozenset([FrozenAssignment({"entrapped_0": True})])
    reach_obs = lidar_for_assignments(
        original_obs, reach, lidar_dim, zone_compat=True,
    )
    np.testing.assert_allclose(reach_obs, building)


def test_zone_compat_surface_unchanged():
    lidar_dim = 4
    building = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64)
    surface = np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "terracotta_buildings_lidar_0": building,
        "surface_casualtys_lidar_0": surface,
    }
    reach = frozenset([FrozenAssignment({"surface_0": True})])
    reach_obs = lidar_for_assignments(
        original_obs, reach, lidar_dim, zone_compat=True,
    )
    np.testing.assert_allclose(reach_obs, surface)


def test_zone_compat_walls_in_avoid():
    lidar_dim = 4
    walls = np.array([0.0, 0.7, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "walls_lidar_0": walls,
        "terracotta_buildings_lidar_0": np.zeros(lidar_dim),
    }
    avoid = frozenset([FrozenAssignment({"walls": True})])
    avoid_obs = lidar_for_assignments(
        original_obs, avoid, lidar_dim, zone_compat=True,
    )
    np.testing.assert_allclose(avoid_obs, walls)


def test_pre_process_zone_compat_omits_indep_buildings_and_walls():
    lidar_dim = 4
    env, task = _mock_sar_env(lidar_dim)
    keys = sar_agent_obs_keys(0)
    # Match real Point SAR agent block: 3+3+3+3+4 = 16
    dims = [3, 3, 3, 3, 4]
    agent_obs = {k: np.zeros(d, dtype=np.float32) for k, d in zip(keys, dims)}
    building = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    walls = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    entrapped = np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32)
    surface = np.array([0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    agent_obs["terracotta_buildings_lidar_0"] = building
    agent_obs["walls_lidar_0"] = walls
    agent_obs["entrapped_casualtys_lidar_0"] = entrapped
    agent_obs["surface_casualtys_lidar_0"] = surface

    reach = frozenset([FrozenAssignment({"entrapped_0": True})])
    avoid = frozenset([FrozenAssignment({"walls": True})])
    feat_dim = sar_feat_dim(lidar_dim, zone_compat=True)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("envs.seq_wrapper.sar_agent_obs", lambda _env, _idx=0: agent_obs)
        mp.setattr("envs.seq_wrapper.sar_task", lambda _env: task)
        feat = pre_process_obs_sar(
            env, keys, reach, avoid, (feat_dim,), zone_compat=True,
        )

    assert feat.shape == (feat_dim,)
    assert feat_dim == 16 + 2 * lidar_dim
    reach_slice = feat[16: 16 + lidar_dim]
    avoid_slice = feat[16 + lidar_dim:]
    np.testing.assert_allclose(reach_slice, building)
    np.testing.assert_allclose(avoid_slice, walls)


def test_strip_walls_avoid_lidar_zeros_walls_in_avoid_features():
    lidar_dim = 4
    env, task = _mock_sar_env(lidar_dim)
    keys = sar_agent_obs_keys(0)
    dims = [3, 3, 3, 3, 4]
    agent_obs = {k: np.zeros(d, dtype=np.float32) for k, d in zip(keys, dims)}
    building = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    walls = np.array([0.0, 0.9, 0.0, 0.0], dtype=np.float32)
    surface = np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32)
    agent_obs["terracotta_buildings_lidar_0"] = building
    agent_obs["walls_lidar_0"] = walls
    agent_obs["surface_casualtys_lidar_0"] = surface
    agent_obs["entrapped_casualtys_lidar_0"] = np.zeros(lidar_dim, dtype=np.float32)

    reach = frozenset([FrozenAssignment({"entrapped_0": True})])
    avoid = frozenset([
        FrozenAssignment({"walls": True}),
        FrozenAssignment({"surface_0": True}),
    ])
    feat_dim = sar_feat_dim(lidar_dim, zone_compat=True)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("envs.seq_wrapper.sar_agent_obs", lambda _env, _idx=0: agent_obs)
        mp.setattr("envs.seq_wrapper.sar_task", lambda _env: task)
        feat_on = pre_process_obs_sar(
            env, keys, reach, avoid, (feat_dim,),
            zone_compat=True, strip_walls_avoid_lidar=True,
        )
        feat_off = pre_process_obs_sar(
            env, keys, reach, avoid, (feat_dim,),
            zone_compat=True, strip_walls_avoid_lidar=False,
        )

    avoid_on = feat_on[16 + lidar_dim:]
    avoid_off = feat_off[16 + lidar_dim:]
    np.testing.assert_allclose(avoid_on, surface)
    np.testing.assert_allclose(avoid_off, np.maximum(walls, surface))


def test_lidar_for_assignments_skip_walls_prop():
    lidar_dim = 4
    walls = np.array([0.0, 0.8, 0.0, 0.0], dtype=np.float64)
    surface = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float64)
    original_obs = {
        "walls_lidar_0": walls,
        "surface_casualtys_lidar_0": surface,
    }
    avoid = frozenset([
        FrozenAssignment({"walls": True}),
        FrozenAssignment({"surface_0": True}),
    ])
    full = lidar_for_assignments(original_obs, avoid, lidar_dim)
    stripped = lidar_for_assignments(
        original_obs, avoid, lidar_dim, skip_props={"walls"},
    )
    np.testing.assert_allclose(full, np.maximum(walls, surface))
    np.testing.assert_allclose(stripped, surface)
