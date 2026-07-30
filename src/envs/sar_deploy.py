"""SAR deploy helpers: feature width, Rabinizer preflight, MA episode helpers."""
from __future__ import annotations

import os
import subprocess
from typing import Any

from envs.seq_wrapper import sar_feat_dim, sar_task

GENZ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RABINIZER_REL = os.path.join("rabinizer4", "bin", "ltl2ldba")


def genz_root() -> str:
    return GENZ_ROOT


def rabinizer_path() -> str:
    return os.path.join(GENZ_ROOT, RABINIZER_REL)


def sar_preprocess_for_deploy(
    env: Any,
    model: Any,
    reach: Any,
    avoid: Any,
    *,
    agent_idx: int = 0,
) -> Any:
    """Build SAR feature vector at deploy width (checkpoint / deploy_meta)."""
    lidar_bins = sar_task(env).lidar_conf.num_bins
    feat_shape = resolve_sar_feat_shape(model, lidar_bins)
    if hasattr(env, "pre_process_obs_sar"):
        return env.pre_process_obs_sar(
            reach, avoid, agent_idx=agent_idx, feat_shape=feat_shape,
        )
    if hasattr(env, "pre_process_obs_zones"):
        return env.pre_process_obs_zones(reach, avoid)
    return env.pre_process_obs_letter(reach, avoid)


def resolve_sar_feat_shape(model: Any, lidar_bins: int) -> tuple[int, ...]:
    """Feature vector width for ``pre_process_obs_sar`` (before optional env_net)."""
    raw = getattr(model, "raw_feature_dim", None)
    if raw is not None:
        return (int(raw),)
    legacy = getattr(model, "input_feat_dim", None)
    if legacy is not None:
        return (int(legacy),)
    return (sar_feat_dim(lidar_bins),)


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
