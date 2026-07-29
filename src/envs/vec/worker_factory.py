"""Spawn-safe picklable env factory for GenZ async vector workers."""

from __future__ import annotations

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
) -> gymnasium.Env:
    """Top-level factory for multiprocessing workers (Windows spawn-safe)."""
    ensure_genz_paths()

    from sequence.samplers import CurriculumSampler, curricula
    from envs.env_utils import make_env, make_env_safety

    curriculum = curricula[curriculum_name]
    curriculum.stage_index = curriculum_stage
    sampler = CurriculumSampler.partial(curriculum)
    worker_seed = seed + rank
    factory: Callable = make_env_safety if safety else make_env
    env = factory(
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
) -> Callable[[], gymnasium.Env]:
    def _thunk() -> gymnasium.Env:
        return _make_genz_worker_env(
            env_name,
            curriculum_name,
            curriculum_stage,
            seed,
            rank,
            max_steps,
            sar_env_backend,
            safety,
            sequence,
        )

    return _thunk
