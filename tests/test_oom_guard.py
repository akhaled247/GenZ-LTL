"""Unit tests for GenZ vec OOM guardrails (no MuJoCo)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from envs.vec.oom_guard import apply_oom_guardrails, safe_num_procs


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
