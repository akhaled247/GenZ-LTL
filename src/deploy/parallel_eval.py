"""Episode-shard parallel MA eval (seed+i preserved)."""
from __future__ import annotations

import multiprocessing as mp
import warnings
from typing import Any, Callable, Sequence

import numpy as np

from envs.vec.oom_guard import safe_num_eval_workers


def shard_episode_indices(num_episodes: int, num_workers: int) -> list[list[int]]:
    workers = max(1, min(int(num_workers), int(num_episodes)))
    base, rem = divmod(int(num_episodes), workers)
    shards: list[list[int]] = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        if size:
            shards.append(list(range(start, start + size)))
            start += size
    return shards


def aggregate_ma_shards(
    shards: Sequence[tuple[int, int, int, list[int], list[float]]],
    *,
    seed: int,
    num_episodes: int,
) -> tuple[int, int, float]:
    num_successes = sum(s[0] for s in shards)
    num_violations = sum(s[1] for s in shards)
    num_unreachable = sum(s[2] for s in shards)
    steps: list[int] = []
    rets: list[float] = []
    for s in shards:
        steps.extend(s[3])
        rets.extend(s[4])
    average_steps = float(np.mean(steps)) if steps else float("nan")
    adr = float(np.mean(rets)) if rets else 0.0
    print(
        f"{seed}: {num_successes / num_episodes:.3f},"
        f"{num_violations / num_episodes:.3f},"
        f"{num_unreachable / num_episodes:.3f},"
        f"{adr:.3f},{average_steps:.3f}"
    )
    return num_successes, num_violations, average_steps


def clamp_eval_workers(
    num_workers: int,
    *,
    env_name: str,
    num_episodes: int,
) -> int:
    """Apply RAM/CPU/hard caps; never exceed episode count."""
    requested = max(1, int(num_workers))
    decision = safe_num_eval_workers(requested, env_name=env_name)
    print(f"[oom_guard] {decision.reason}")
    if decision.clamped:
        warnings.warn(decision.reason, stacklevel=2)
    workers = max(1, min(decision.num_procs, int(num_episodes)))
    if workers < decision.num_procs:
        print(
            f"[oom_guard] Further capped num_workers {decision.num_procs} → {workers} "
            f"(num_episodes={num_episodes})."
        )
    return workers


def run_sharded_ma_eval(
    worker_fn: Callable[[dict[str, Any]], tuple[int, int, int, list[int], list[float]]],
    base_kwargs: dict[str, Any],
    *,
    num_episodes: int,
    num_workers: int,
    env_name: str | None = None,
) -> tuple[int, int, float]:
    env = env_name or str(base_kwargs.get("eval_env") or base_kwargs.get("train_env") or "")
    workers = clamp_eval_workers(num_workers, env_name=env, num_episodes=num_episodes)
    shards = shard_episode_indices(num_episodes, workers)
    if len(shards) == 1:
        result = worker_fn({**base_kwargs, "episode_indices": shards[0]})
        return aggregate_ma_shards([result], seed=base_kwargs["seed"], num_episodes=num_episodes)

    payloads = [{**base_kwargs, "episode_indices": shard} for shard in shards]
    # Windows-safe spawn; worker_fn must be importable top-level.
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(payloads)) as pool:
        results = pool.map(worker_fn, payloads)
    return aggregate_ma_shards(results, seed=base_kwargs["seed"], num_episodes=num_episodes)
