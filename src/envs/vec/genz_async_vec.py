"""Multiprocess vector env for GenZ (spawn-safe; matches SyncEnv step contract)."""

from __future__ import annotations

import sys
from multiprocessing import Pipe, get_context
from typing import Any, Callable

import numpy as np


def _mp_context() -> str:
    return "fork" if sys.platform != "win32" else "spawn"


def _genz_worker(conn, env_fn: Callable[[], Any]) -> None:
    env = env_fn()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "kill":
            conn.close()
            return
        else:
            raise NotImplementedError(cmd)


class GenZSafetyAsyncEnv:
    """All envs in subprocesses; same step/reset contract as SyncEnv / ParallelEnv."""

    def __init__(self, env_fns: list[Callable[[], Any]]):
        assert len(env_fns) >= 1, "No environment given."
        self.num_envs = len(env_fns)
        self._locals: list[Any] = []
        self._processes: list[Any] = []

        ctx = get_context(_mp_context())
        for env_fn in env_fns:
            local, remote = Pipe()
            proc = ctx.Process(target=_genz_worker, args=(remote, env_fn))
            proc.daemon = True
            proc.start()
            remote.close()
            self._locals.append(local)
            self._processes.append(proc)

    def __del__(self) -> None:
        for local in self._locals:
            try:
                local.send(("kill", None))
            except (BrokenPipeError, OSError):
                pass

    def reset(self) -> list[Any]:
        for local in self._locals:
            local.send(("reset", None))
        return [local.recv() for local in self._locals]

    def step(self, actions: np.ndarray) -> tuple[tuple, tuple, tuple, tuple]:
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, -1)
        results = []
        for local, action in zip(self._locals, actions):
            local.send(("step", np.asarray(action)))
            results.append(local.recv())
        return tuple(zip(*results))

    def close(self) -> None:
        for local in self._locals:
            try:
                local.send(("kill", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._processes:
            proc.join(timeout=1)


def build_genz_async_vec(
    n_envs: int,
    env_name: str,
    curriculum_name: str,
    curriculum_stage: int,
    seed: int,
    max_steps: int | None,
    sar_env_backend: str,
    safety: bool = True,
    sequence: bool = True,
    entr_bldg_obs: bool = False,
) -> GenZSafetyAsyncEnv:
    from envs.vec.worker_factory import make_worker_env_thunk

    seed_offset = 100 * seed
    env_fns = [
        make_worker_env_thunk(
            env_name,
            curriculum_name,
            curriculum_stage,
            seed_offset + rank,
            rank,
            max_steps,
            sar_env_backend,
            safety,
            sequence,
            entr_bldg_obs,
        )
        for rank in range(n_envs)
    ]
    return GenZSafetyAsyncEnv(env_fns)
