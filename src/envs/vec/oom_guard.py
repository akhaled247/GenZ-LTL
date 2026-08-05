"""RAM / process guardrails for GenZ vec envs (avoid machine-level OOM)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable


# Conservative RSS per MuJoCo Safety-Gym worker (parent probe + each spawn).
# Override: GENZ_ENV_RSS_MB. Leave headroom: GENZ_RAM_RESERVE_GB (default 6).
_DEFAULT_SAR_RSS_MB = 750
_DEFAULT_POINT_RSS_MB = 400
_DEFAULT_RESERVE_GB = 6.0
_DEFAULT_MAX_PROCS_HARD = 32
# CLI ``steps_per_process`` is defined at this worker count (healthy SAR recipe).
# Override: GENZ_ROLLOUT_REF_PROCS.
_DEFAULT_ROLLOUT_REF_PROCS = 24


@dataclass(frozen=True)
class OomGuardDecision:
    num_procs: int
    clamped: bool
    reason: str


def rollout_ref_procs() -> int:
    raw = os.environ.get("GENZ_ROLLOUT_REF_PROCS", "").strip()
    if raw:
        return max(1, int(raw))
    return _DEFAULT_ROLLOUT_REF_PROCS


def compensate_steps_per_process(
    reference_num_procs: int,
    actual_num_procs: int,
    steps_per_process: int,
) -> int:
    """Scale ``steps_per_process`` so frames/update match the reference rollout.

    Reference rollout is ``steps_per_process * reference_num_procs`` (default
    reference = 24). If ``actual_num_procs`` differs (user chose 1, or OOM
    clamped 24→8), raise/lower ``steps_per_process`` (ceil) so each update still
    sees ≈ that many timesteps.
    """
    if (
        reference_num_procs < 1
        or actual_num_procs < 1
        or steps_per_process < 1
        or actual_num_procs == reference_num_procs
    ):
        return steps_per_process
    target = steps_per_process * reference_num_procs
    return (target + actual_num_procs - 1) // actual_num_procs


def available_ram_bytes() -> int | None:
    """Best-effort MemAvailable (Linux) / GlobalMemoryStatusEx (Windows)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except Exception:
        pass
    return None


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def estimate_env_rss_bytes(env_name: str) -> int:
    override = os.environ.get("GENZ_ENV_RSS_MB", "").strip()
    if override:
        return max(64, int(override)) * 1024 * 1024
    name = (env_name or "").upper()
    mb = _DEFAULT_SAR_RSS_MB if "SAR" in name or "MASAR" in name else _DEFAULT_POINT_RSS_MB
    return mb * 1024 * 1024


def estimate_eval_worker_rss_bytes(env_name: str) -> int:
    """RSS for one MA eval spawn worker (MuJoCo env + policy + LDBA)."""
    override = os.environ.get("GENZ_EVAL_WORKER_RSS_MB", "").strip()
    if override:
        return max(64, int(override)) * 1024 * 1024
    # Heavier than train env-only workers: each spawn loads ckpt + search.
    return int(estimate_env_rss_bytes(env_name) * 1.5)


def reserve_bytes() -> int:
    gb = float(os.environ.get("GENZ_RAM_RESERVE_GB", str(_DEFAULT_RESERVE_GB)))
    return max(1.0, gb) * 1024 ** 3


def force_num_procs() -> bool:
    return os.environ.get("GENZ_FORCE_NUM_PROCS", "").strip().lower() in {"1", "true", "yes"}


def max_procs_hard_cap() -> int:
    raw = os.environ.get("GENZ_MAX_PROCS", "").strip()
    if raw:
        return max(1, int(raw))
    return _DEFAULT_MAX_PROCS_HARD


def safe_num_procs(
    requested: int,
    *,
    env_name: str,
    vec_backend: str,
    parallel: bool,
) -> OomGuardDecision:
    """Clamp / refuse dangerous vec configs before MuJoCo workers spawn."""
    if requested < 1:
        raise ValueError(f"num_procs must be >= 1, got {requested}")

    # list + parallel: parent builds N full envs then forks → classic machine OOM.
    if vec_backend == "list" and parallel and requested > 1 and not force_num_procs():
        raise RuntimeError(
            "Refusing --vec_backend list --parallel with num_procs>1: parent creates "
            f"{requested} MuJoCo envs then forks (often OOMs the host / breaks CUDA). "
            "Use --vec_backend safety_async (fixed send-all/recv-all). "
            "Override only if you accept the risk: GENZ_FORCE_NUM_PROCS=1."
        )

    # list SyncEnv: still N sims in one process.
    if vec_backend == "list" and not parallel and requested > 1 and not force_num_procs():
        raise RuntimeError(
            "Refusing --vec_backend list with num_procs>1: all envs live in one process "
            f"({requested} MuJoCo sims → host OOM risk). Use --vec_backend safety_async. "
            "Override: GENZ_FORCE_NUM_PROCS=1."
        )

    caps: list[tuple[int, str]] = [(requested, "requested")]
    hard = max_procs_hard_cap()
    caps.append((hard, f"GENZ_MAX_PROCS/hard={hard}"))

    cpus = cpu_count()
    # Leave a couple cores for parent + OS when many workers.
    cpu_cap = max(1, cpus - 1) if requested > 1 else requested
    caps.append((cpu_cap, f"cpu_count-1={cpu_cap}"))

    avail = available_ram_bytes()
    per_env = estimate_env_rss_bytes(env_name)
    headroom = reserve_bytes()
    if avail is not None:
        # safety_async: probe in parent + N workers ≈ (N+1) * per_env
        # list paths already refused above when N>1 unless forced.
        multiplier = (requested + 1) if vec_backend == "safety_async" else requested
        budget = max(0, avail - headroom)
        ram_cap = max(1, int(budget // per_env)) if per_env > 0 else requested
        # If estimate says only room for probe, still allow 1 worker.
        if vec_backend == "safety_async":
            ram_cap = max(1, ram_cap - 1)  # account for probe in parent
        caps.append(
            (
                ram_cap,
                f"ram≈{avail / 1024**3:.1f}GiB avail, reserve={headroom / 1024**3:.1f}GiB, "
                f"~{per_env / 1024**2:.0f}MiB/env",
            )
        )

    chosen = min(c[0] for c in caps)
    binding = next(c for c in caps if c[0] == chosen)
    if chosen < requested and not force_num_procs():
        return OomGuardDecision(
            num_procs=chosen,
            clamped=True,
            reason=(
                f"Clamped num_procs {requested} → {chosen} ({binding[1]}). "
                f"Override: GENZ_FORCE_NUM_PROCS=1 or lower GENZ_ENV_RSS_MB / GENZ_RAM_RESERVE_GB."
            ),
        )
    if chosen < requested and force_num_procs():
        return OomGuardDecision(
            num_procs=requested,
            clamped=False,
            reason=(
                f"GENZ_FORCE_NUM_PROCS=1: keeping num_procs={requested} despite cap "
                f"{chosen} ({binding[1]})."
            ),
        )
    return OomGuardDecision(
        num_procs=requested,
        clamped=False,
        reason=f"num_procs={requested} within caps ({binding[1]}).",
    )


def safe_num_eval_workers(
    requested: int,
    *,
    env_name: str,
) -> OomGuardDecision:
    """Clamp MA eval ``num_workers`` before spawn Pool (each worker = full env+ckpt)."""
    if requested < 1:
        raise ValueError(f"num_workers must be >= 1, got {requested}")
    if requested == 1:
        return OomGuardDecision(
            num_procs=1,
            clamped=False,
            reason="num_workers=1 (serial).",
        )

    caps: list[tuple[int, str]] = [(requested, "requested")]
    hard = max_procs_hard_cap()
    caps.append((hard, f"GENZ_MAX_PROCS/hard={hard}"))

    cpus = cpu_count()
    cpu_cap = max(1, cpus - 1)
    caps.append((cpu_cap, f"cpu_count-1={cpu_cap}"))

    avail = available_ram_bytes()
    per_worker = estimate_eval_worker_rss_bytes(env_name)
    headroom = reserve_bytes()
    if avail is not None:
        budget = max(0, avail - headroom)
        ram_cap = max(1, int(budget // per_worker)) if per_worker > 0 else requested
        caps.append(
            (
                ram_cap,
                f"ram≈{avail / 1024**3:.1f}GiB avail, reserve={headroom / 1024**3:.1f}GiB, "
                f"~{per_worker / 1024**2:.0f}MiB/eval-worker",
            )
        )

    chosen = min(c[0] for c in caps)
    binding = next(c for c in caps if c[0] == chosen)
    if chosen < requested and not force_num_procs():
        return OomGuardDecision(
            num_procs=chosen,
            clamped=True,
            reason=(
                f"Clamped num_workers {requested} → {chosen} ({binding[1]}). "
                f"Override: GENZ_FORCE_NUM_PROCS=1 or set GENZ_EVAL_WORKER_RSS_MB / "
                f"GENZ_RAM_RESERVE_GB."
            ),
        )
    if chosen < requested and force_num_procs():
        return OomGuardDecision(
            num_procs=requested,
            clamped=False,
            reason=(
                f"GENZ_FORCE_NUM_PROCS=1: keeping num_workers={requested} despite cap "
                f"{chosen} ({binding[1]})."
            ),
        )
    return OomGuardDecision(
        num_procs=requested,
        clamped=False,
        reason=f"num_workers={requested} within caps ({binding[1]}).",
    )


def apply_oom_guardrails(
    experiment,
    *,
    algo_config=None,
    log: Callable[[str], None] | None = None,
) -> OomGuardDecision:
    """Mutate ``experiment.num_procs`` (and optionally algo ``steps_per_process``).

    After OOM clamp, scale ``algo_config.steps_per_process`` so frames/update
    stay ≈ ``spp * GENZ_ROLLOUT_REF_PROCS`` (default 24) — including when the
    user passes ``--num_procs 1`` (or any count ≠ 24).
    """
    log = log or (lambda msg: warnings.warn(msg, stacklevel=2))
    requested = int(experiment.num_procs)
    decision = safe_num_procs(
        requested,
        env_name=str(getattr(experiment, "env", "")),
        vec_backend=str(getattr(experiment, "vec_backend", "list")),
        parallel=bool(getattr(experiment, "parallel", False)),
    )
    log(f"[oom_guard] {decision.reason}")
    if decision.clamped:
        experiment.num_procs = decision.num_procs

    if algo_config is not None and hasattr(algo_config, "steps_per_process"):
        ref = rollout_ref_procs()
        old_spp = int(algo_config.steps_per_process)
        actual = int(experiment.num_procs)
        new_spp = compensate_steps_per_process(ref, actual, old_spp)
        if new_spp != old_spp:
            algo_config.steps_per_process = new_spp
            old_rollout = old_spp * ref
            new_rollout = new_spp * actual
            log(
                f"[oom_guard] Preserved rollout vs {ref} procs: steps_per_process "
                f"{old_spp} -> {new_spp} "
                f"(frames/update {old_rollout} -> {new_rollout}; num_procs={actual})"
            )
    return decision


def mem_available_below_reserve() -> bool:
    avail = available_ram_bytes()
    if avail is None:
        return False
    return avail < reserve_bytes()
