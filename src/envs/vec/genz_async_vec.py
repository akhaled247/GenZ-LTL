"""Multiprocess vector env for GenZ (spawn-safe; matches SyncEnv step contract)."""

from __future__ import annotations

import os
import traceback
from multiprocessing import Pipe, get_context
from typing import Any, Callable

import numpy as np


def _mp_context() -> str:
    # Always spawn: parent already has MuJoCo probe + CUDA (model.to) before
    # workers start. fork after that → silent worker death → ConnectionResetError.
    method = os.environ.get("GENZ_MP_START_METHOD", "spawn").strip().lower()
    if method not in {"spawn", "fork", "forkserver"}:
        raise ValueError(f"Invalid GENZ_MP_START_METHOD={method!r}")
    return method


def _genz_worker(conn, env_fn: Callable[[], Any]) -> None:
    try:
        env = env_fn()
    except BaseException:
        try:
            conn.send(("__error__", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
        conn.close()
        return

    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            return
        try:
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
        except BaseException:
            try:
                conn.send(("__error__", traceback.format_exc()))
            except (BrokenPipeError, OSError):
                pass
            conn.close()
            return


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

    def _recv(self, local: Any, idx: int) -> Any:
        try:
            msg = local.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            proc = self._processes[idx]
            exitcode = proc.exitcode
            raise RuntimeError(
                f"Async env worker {idx} died during IPC "
                f"(exitcode={exitcode}). Parent already had MuJoCo/CUDA; "
                f"if this persists set GENZ_MP_START_METHOD=spawn (default). "
                f"Underlying: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__error__":
            raise RuntimeError(f"Async env worker {idx} failed:\n{msg[1]}")
        return msg

    def reset(self) -> list[Any]:
        for local in self._locals:
            local.send(("reset", None))
        return [self._recv(local, i) for i, local in enumerate(self._locals)]

    def step(self, actions: np.ndarray) -> tuple[tuple, tuple, tuple, tuple]:
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, -1)
        results = []
        for i, (local, action) in enumerate(zip(self._locals, actions)):
            local.send(("step", np.asarray(action)))
            results.append(self._recv(local, i))
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
    zone_compat: bool = False,
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
            zone_compat,
        )
        for rank in range(n_envs)
    ]
    return GenZSafetyAsyncEnv(env_fns)
