"""SAR deploy helpers: Rabinizer preflight, MA episode helpers (re-exports deploy recipe)."""
from __future__ import annotations

import os
import subprocess
from typing import Any

from deploy.feature_recipe import (
    allow_legacy_padding,
    resolve_feat_shape,
    sar_preprocess_for_deploy,
)
from envs.seq_wrapper import sar_feat_dim, sar_task

GENZ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RABINIZER_REL = os.path.join("rabinizer4", "bin", "ltl2ldba")


def genz_root() -> str:
    return GENZ_ROOT


def rabinizer_path() -> str:
    return os.path.join(GENZ_ROOT, RABINIZER_REL)


def resolve_sar_feat_shape(model: Any, lidar_bins: int) -> tuple[int, ...]:
    return resolve_feat_shape(model, lidar_bins)


def check_rabinizer() -> None:
    """Fail fast if Rabinizer wrapper is missing or not executable."""
    path = rabinizer_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Rabinizer not found at {path}. "
            "Ensure GenZ-LTL/rabinizer4 is present (see README)."
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(
            f"Rabinizer not executable: {path}\n"
            f"Run: chmod +x {path}"
        )
    try:
        subprocess.run(
            ["java", "-version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Java is required for Rabinizer (Java 11+). Install Java and retry."
        ) from exc


def ma_step_done(
    terminated: Any,
    truncated: Any,
    agents: list[str],
) -> bool:
    """Paper protocol: episode ends when any agent terminates or truncates."""
    if isinstance(terminated, dict):
        term = any(bool(terminated[a]) for a in agents)
        trunc = any(bool(truncated[a]) for a in agents)
        return term or trunc
    return bool(terminated) or bool(truncated)


def ma_episode_success(info: dict[str, Any], agents: list[str]) -> bool:
    if info.get("success"):
        return True
    for agent in agents:
        ai = info.get(agent, {})
        if isinstance(ai, dict) and ai.get("success"):
            return True
    return False


def ma_episode_violation(info: dict[str, Any], agents: list[str]) -> bool:
    if info.get("violation"):
        return True
    for agent in agents:
        ai = info.get(agent, {})
        if isinstance(ai, dict) and ai.get("violation"):
            return True
    return False
