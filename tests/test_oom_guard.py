"""Unit tests for GenZ vec OOM guardrails (no MuJoCo)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from envs.vec.oom_guard import (
    apply_oom_guardrails,
    compensate_steps_per_process,
    safe_num_procs,
)


def test_refuse_list_parallel(monkeypatch):
    monkeypatch.delenv("GENZ_FORCE_NUM_PROCS", raising=False)
    with pytest.raises(RuntimeError, match="safety_async"):
        safe_num_procs(
            24,
            env_name="PointLTL1MASAR1WC-v0",
            vec_backend="list",
            parallel=True,
        )


def test_refuse_list_sync_multi(monkeypatch):
    monkeypatch.delenv("GENZ_FORCE_NUM_PROCS", raising=False)
    with pytest.raises(RuntimeError, match="safety_async"):
        safe_num_procs(
            8,
            env_name="PointLTL1MASAR1WC-v0",
            vec_backend="list",
            parallel=False,
        )


def test_force_allows_list_parallel(monkeypatch):
    monkeypatch.setenv("GENZ_FORCE_NUM_PROCS", "1")
    monkeypatch.setenv("GENZ_MAX_PROCS", "64")
    d = safe_num_procs(
        4,
        env_name="PointLTL1MASAR1WC-v0",
        vec_backend="list",
        parallel=True,
    )
    assert d.num_procs == 4


def test_clamp_by_hard_cap(monkeypatch):
    monkeypatch.delenv("GENZ_FORCE_NUM_PROCS", raising=False)
    monkeypatch.setenv("GENZ_MAX_PROCS", "2")
    monkeypatch.setenv("GENZ_ENV_RSS_MB", "1")
    monkeypatch.setenv("GENZ_RAM_RESERVE_GB", "0.001")
    d = safe_num_procs(
        16,
        env_name="PointLTL1MASAR1WC-v0",
        vec_backend="safety_async",
        parallel=False,
    )
    assert d.clamped
    assert d.num_procs == 2


def test_compensate_steps_per_process_exact():
    # 4096 * 24 = 98304; 8 procs → 12288 * 8 = 98304
    assert compensate_steps_per_process(24, 8, 4096) == 12288


def test_compensate_steps_per_process_ceil():
    # 4096 * 24 = 98304; 7 procs → ceil(98304/7)=14044
    assert compensate_steps_per_process(24, 7, 4096) == 14044
    assert 14044 * 7 >= 4096 * 24


def test_compensate_noop_at_reference():
    assert compensate_steps_per_process(24, 24, 4096) == 4096


def test_compensate_num_procs_one():
    # Explicit --num_procs 1 still gets full 24-proc rollout budget
    assert compensate_steps_per_process(24, 1, 4096) == 98304


def test_apply_mutates_experiment(monkeypatch):
    monkeypatch.delenv("GENZ_FORCE_NUM_PROCS", raising=False)
    monkeypatch.setenv("GENZ_MAX_PROCS", "3")
    monkeypatch.setenv("GENZ_ENV_RSS_MB", "1")
    monkeypatch.setenv("GENZ_RAM_RESERVE_GB", "0.001")
    exp = SimpleNamespace(
        num_procs=12,
        env="PointLTL1MASAR1WC-v0",
        vec_backend="safety_async",
        parallel=False,
    )
    logs: list[str] = []
    apply_oom_guardrails(exp, log=logs.append)
    assert exp.num_procs == 3
    assert any("Clamped" in m for m in logs)


def test_apply_preserves_rollout_vs_ref_24(monkeypatch):
    monkeypatch.delenv("GENZ_FORCE_NUM_PROCS", raising=False)
    monkeypatch.delenv("GENZ_ROLLOUT_REF_PROCS", raising=False)
    monkeypatch.setenv("GENZ_MAX_PROCS", "3")
    monkeypatch.setenv("GENZ_ENV_RSS_MB", "1")
    monkeypatch.setenv("GENZ_RAM_RESERVE_GB", "0.001")
    exp = SimpleNamespace(
        num_procs=12,
        env="PointLTL1MASAR1WC-v0",
        vec_backend="safety_async",
        parallel=False,
    )
    algo = SimpleNamespace(steps_per_process=4096)
    logs: list[str] = []
    apply_oom_guardrails(exp, algo_config=algo, log=logs.append)
    assert exp.num_procs == 3
    # Target = 4096 * 24, not 4096 * requested(12)
    assert algo.steps_per_process == 32768
    assert algo.steps_per_process * exp.num_procs == 4096 * 24
    assert any("Preserved rollout" in m for m in logs)


def test_apply_scales_explicit_num_procs_one(monkeypatch):
    monkeypatch.delenv("GENZ_FORCE_NUM_PROCS", raising=False)
    monkeypatch.delenv("GENZ_ROLLOUT_REF_PROCS", raising=False)
    monkeypatch.setenv("GENZ_MAX_PROCS", "32")
    monkeypatch.setenv("GENZ_ENV_RSS_MB", "1")
    monkeypatch.setenv("GENZ_RAM_RESERVE_GB", "0.001")
    exp = SimpleNamespace(
        num_procs=1,
        env="PointLTL1MASAR1WC-v0",
        vec_backend="list",
        parallel=False,
    )
    algo = SimpleNamespace(steps_per_process=4096)
    logs: list[str] = []
    apply_oom_guardrails(exp, algo_config=algo, log=logs.append)
    assert exp.num_procs == 1
    assert algo.steps_per_process == 98304
    assert any("Preserved rollout" in m for m in logs)
