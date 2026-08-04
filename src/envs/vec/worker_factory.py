"""Spawn-safe picklable env factory for GenZ async vector workers."""

from __future__ import annotations

from functools import partial
from typing import Callable

import gymnasium

from envs.vec.paths_bootstrap import ensure_genz_paths


def _make_genz_worker_env(
    env_name: str,
    curriculum_name: str,
    curriculum_stage: int,
    seed: int,
    rank: int,
    max_steps: int | None,
    sar_env_backend: str,
    safety: bool,
    sequence: bool,
    entr_bldg_obs: bool = False,
    zone_compat: bool = False,
) -> gymnasium.Env:
    """Top-level factory for multiprocessing workers (Windows spawn-safe)."""
    ensure_genz_paths()

    from sequence.samplers import CurriculumSampler, curricula
    from envs.env_utils import make_env, make_env_safety

    curriculum = curricula[curriculum_name]
    curriculum.stage_index = curriculum_stage
    sampler = CurriculumSampler.partial(curriculum)
    worker_seed = seed + rank
    if safety:
        env = make_env_safety(
            env_name,
            sampler,
            max_steps=max_steps,
            sequence=sequence,
            sar_env_backend=sar_env_backend,
            entr_bldg_obs=entr_bldg_obs,
            zone_compat=zone_compat,
        )
    else:
        env = make_env(
            env_name,
            sampler,
            max_steps=max_steps,
            sequence=sequence,
            sar_env_backend=sar_env_backend,
        )
    env.reset(seed=worker_seed)
    return env


def make_worker_env_thunk(
    env_name: str,
    curriculum_name: str,
    curriculum_stage: int,
    seed: int,
    rank: int,
    max_steps: int | None,
    sar_env_backend: str,
    safety: bool,
    sequence: bool,
    entr_bldg_obs: bool = False,
    zone_compat: bool = False,
) -> Callable[[], gymnasium.Env]:
    # partial of top-level fn — picklable under spawn (closures are not).
    return partial(
        _make_genz_worker_env,
        env_name,
        curriculum_name,
        curriculum_stage,
        seed,
        rank,
        max_steps,
        sar_env_backend,
        safety,
        sequence,
        entr_bldg_obs,
        zone_compat,
    )
