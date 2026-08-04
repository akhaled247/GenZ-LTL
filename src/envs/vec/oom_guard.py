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


@dataclass(frozen=True)
class OomGuardDecision:
    num_procs: int
    clamped: bool
    reason: str


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


def apply_oom_guardrails(
    experiment,
    *,
    log: Callable[[str], None] | None = None,
) -> OomGuardDecision:
    """Mutate ``experiment.num_procs`` in place; raise on refused configs."""
    log = log or (lambda msg: warnings.warn(msg, stacklevel=2))
    decision = safe_num_procs(
        int(experiment.num_procs),
        env_name=str(getattr(experiment, "env", "")),
        vec_backend=str(getattr(experiment, "vec_backend", "list")),
        parallel=bool(getattr(experiment, "parallel", False)),
    )
    if decision.clamped:
        log(f"[oom_guard] {decision.reason}")
        experiment.num_procs = decision.num_procs
    else:
        log(f"[oom_guard] {decision.reason}")
    return decision


def mem_available_below_reserve() -> bool:
    avail = available_ram_bytes()
    if avail is None:
        return False
    return avail < reserve_bytes()
