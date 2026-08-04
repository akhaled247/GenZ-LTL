"""Reach subgoal lidar for entrapped props (optional building max-pool)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from envs.seq_wrapper import (
    lidar_for_assignments,
    pre_process_obs_sar,
    sar_agent_obs_keys,
    sar_feat_dim,
)
from ltl.logic import FrozenAssignment


def _mock_sar_env(lidar_dim: int = 4):
    env = MagicMock()
    task = MagicMock()
    task.lidar_conf.num_bins = lidar_dim
    task.agent_num = 1
    env.unwrapped.task = task
    return env, task


def test_entrapped_reach_lidar_default_uses_casualty_only():
    lidar_dim = 4
    building = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64)
    entrapped = np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "terracotta_buildings_lidar_0": building,
        "entrapped_casualtys_lidar_0": entrapped,
    }
    reach = frozenset([FrozenAssignment({"entrapped_0": True})])

    reach_obs = lidar_for_assignments(original_obs, reach, lidar_dim)
    np.testing.assert_allclose(reach_obs, entrapped)


def test_entrapped_reach_lidar_max_pools_building_when_flag_on():
    lidar_dim = 4
    building = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64)
    entrapped = np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "terracotta_buildings_lidar_0": building,
        "entrapped_casualtys_lidar_0": entrapped,
    }
    reach = frozenset([FrozenAssignment({"entrapped_0": True})])

    reach_obs = lidar_for_assignments(
        original_obs, reach, lidar_dim, for_reach=True,
    )
    np.testing.assert_allclose(reach_obs, np.maximum(building, entrapped))


def test_surface_reach_lidar_does_not_include_building():
    lidar_dim = 4
    building = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64)
    surface = np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "terracotta_buildings_lidar_0": building,
        "surface_casualtys_lidar_0": surface,
    }
    reach = frozenset([FrozenAssignment({"surface_0": True})])

    reach_obs = lidar_for_assignments(
        original_obs, reach, lidar_dim, for_reach=True,
    )
    np.testing.assert_allclose(reach_obs, surface)


def test_entrapped_avoid_lidar_uses_casualty_only():
    lidar_dim = 4
    building = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64)
    entrapped = np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float64)
    original_obs = {
        "terracotta_buildings_lidar_0": building,
        "entrapped_casualtys_lidar_0": entrapped,
    }
    avoid = frozenset([FrozenAssignment({"entrapped_0": True})])

    avoid_obs = lidar_for_assignments(original_obs, avoid, lidar_dim)
    np.testing.assert_allclose(avoid_obs, entrapped)


def test_pre_process_obs_sar_entrapped_reach_respects_flag():
    lidar_dim = 4
    env, task = _mock_sar_env(lidar_dim)
    keys = sar_agent_obs_keys(0)
    dims = [3, 3, 3, 3, 4]
    agent_obs = {k: np.zeros(d, dtype=np.float32) for k, d in zip(keys, dims)}
    building = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    entrapped = np.array([0.0, 0.6, 0.0, 0.0], dtype=np.float32)
    agent_obs["terracotta_buildings_lidar_0"] = building
    agent_obs["entrapped_casualtys_lidar_0"] = entrapped

    reach = frozenset([FrozenAssignment({"entrapped_0": True})])
    feat_dim = sar_feat_dim(lidar_dim)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("envs.seq_wrapper.sar_agent_obs", lambda _env, _idx=0: agent_obs)
        mp.setattr("envs.seq_wrapper.sar_task", lambda _env: task)
        feat_off = pre_process_obs_sar(env, keys, reach, frozenset(), (feat_dim,))
        feat_on = pre_process_obs_sar(
            env, keys, reach, frozenset(), (feat_dim,), entr_bldg_obs=True,
        )

    agent_dim = 16
    reach_off = feat_off[agent_dim + lidar_dim: agent_dim + 2 * lidar_dim]
    reach_on = feat_on[agent_dim + lidar_dim: agent_dim + 2 * lidar_dim]
    np.testing.assert_allclose(reach_off, entrapped)
    np.testing.assert_allclose(reach_on, np.maximum(building, entrapped))


def test_pre_process_obs_sar_includes_walls_lidar_when_present():
    lidar_dim = 4
    env, task = _mock_sar_env(lidar_dim)
    keys = sar_agent_obs_keys(0)
    dims = [3, 3, 3, 3, 4]
    agent_obs = {k: np.zeros(d, dtype=np.float32) for k, d in zip(keys, dims)}
    building = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    walls = np.array([0.0, 0.7, 0.0, 0.0], dtype=np.float32)
    entrapped = np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32)
    agent_obs["terracotta_buildings_lidar_0"] = building
    agent_obs["walls_lidar_0"] = walls
    agent_obs["entrapped_casualtys_lidar_0"] = entrapped

    reach = frozenset([FrozenAssignment({"entrapped_0": True})])
    feat_dim = sar_feat_dim(lidar_dim, include_walls_lidar=True)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("envs.seq_wrapper.sar_agent_obs", lambda _env, _idx=0: agent_obs)
        mp.setattr("envs.seq_wrapper.sar_task", lambda _env: task)
        feat = pre_process_obs_sar(env, keys, reach, frozenset(), (feat_dim,))

    agent_len = 16
    walls_slice = feat[agent_len + lidar_dim: agent_len + 2 * lidar_dim]
    reach_slice = feat[agent_len + 2 * lidar_dim: agent_len + 3 * lidar_dim]
    assert feat.shape == (feat_dim,)
    np.testing.assert_allclose(walls_slice, walls)
    np.testing.assert_allclose(reach_slice, entrapped)
