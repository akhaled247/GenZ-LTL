"""Select GenZ env stepping backend (sync, pipe-parallel, async subprocess)."""

from __future__ import annotations

from typing import Any

import gymnasium

from torch_ac.utils.parallel_env import ParallelEnv
from torch_ac.utils.sync_env import SyncEnv


def build_env_runner(
    envs: list[gymnasium.Env],
    parallel: bool,
    vec_backend: str,
    async_factory_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Return env runner with SyncEnv-compatible ``reset`` / ``step`` API."""
    if vec_backend == "safety_async":
        from envs.vec.genz_async_vec import build_genz_async_vec

        kwargs = dict(async_factory_kwargs or {})
        return build_genz_async_vec(**kwargs)

    if vec_backend != "list":
        raise ValueError(f"Unknown vec_backend: {vec_backend}")

    if parallel:
        return ParallelEnv(envs)
    return SyncEnv(envs)
