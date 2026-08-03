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
    """True if Büchi success or SAR mission complete (goal_met / all rescued)."""
    if info.get("success") or info.get("goal_met"):
        return True
    for agent in agents:
        ai = info.get(agent, {})
        if isinstance(ai, dict) and (ai.get("success") or ai.get("goal_met")):
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


def ma_step_saw_walls(info: dict[str, Any], agents: list[str]) -> bool:
    """True if any agent reported ``cost_walls`` on this step."""
    for agent in agents:
        ai = info.get(agent, {})
        if isinstance(ai, dict) and float(ai.get("cost_walls", 0) or 0) > 0:
            return True
    return False


def ma_agent_cost_walls(info: dict[str, Any], agents: list[str]) -> dict[str, float]:
    """Per-agent wall cost on the current step (0 when absent)."""
    out: dict[str, float] = {}
    for agent in agents:
        ai = info.get(agent, {})
        if isinstance(ai, dict):
            out[agent] = float(ai.get("cost_walls", 0) or 0)
    return out


def _info_goal_met(info: dict[str, Any]) -> bool:
    if info.get("goal_met"):
        return True
    for value in info.values():
        if isinstance(value, dict) and value.get("goal_met"):
            return True
    return False


def print_ma_episode_done_debug(
    env: Any,
    info: dict[str, Any],
    *,
    step: int | None = None,
    saw_walls: bool | None = None,
) -> None:
    """Print termination diagnostics when an MA deploy episode ends."""
    task = sar_task(env)
    num_agents = getattr(task, "agent_num", 2)
    agents = [f"agent_{i}" for i in range(num_agents)]
    surface_geom = getattr(task, "surface_casualtys", None)
    entrapped_geom = getattr(task, "entrapped_casualtys", None)
    surface_rescued = list(surface_geom.rescued) if surface_geom is not None else None
    entrapped_rescued = list(entrapped_geom.rescued) if entrapped_geom is not None else None

    header = f"[MA done debug] step={step}" if step is not None else "[MA done debug]"
    goal_met = _info_goal_met(info)
    cost_walls = ma_agent_cost_walls(info, agents)
    wall_violation = saw_walls if saw_walls is not None else ma_step_saw_walls(info, agents)

    print(header)
    print(f"  success (Büchi): {info.get('success')}")
    print(f"  goal_met (mission): {goal_met}")
    print(f"  violation (LTL): {info.get('violation')}")
    print(f"  wall_violation (WC): {wall_violation}")
    print(f"  cost_walls (final step): {cost_walls}")
    if info.get("cost") is not None:
        print(f"  cost (WC): {info.get('cost')}")
    print(f"  propositions: {info.get('propositions')}")
    print(f"  surface_casualtys.rescued: {surface_rescued}")
    print(f"  entrapped_casualtys.rescued: {entrapped_rescued}")
