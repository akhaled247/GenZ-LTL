"""SAR deploy helpers: Rabinizer preflight, MA episode helpers (re-exports deploy recipe)."""
from __future__ import annotations

import os
import subprocess
from typing import Any

import numpy as np

from deploy.feature_recipe import (
    allow_legacy_padding,
    resolve_feat_shape,
    sar_preprocess_for_deploy,
)
from envs.seq_wrapper import (
    lidar_for_assignments,
    sar_agent_obs,
    sar_task,
    walls_lidar_key,
)

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


def ma_episode_violation(
    info: dict[str, Any],
    agents: list[str],
    *,
    saw_walls: bool | None = None,
) -> bool:
    """True if Büchi violation or WC wall hit (parity with SafePO saw_walls→V)."""
    if info.get("violation"):
        return True
    for agent in agents:
        ai = info.get(agent, {})
        if isinstance(ai, dict) and ai.get("violation"):
            return True
    if saw_walls:
        return True
    if "walls" in (info.get("propositions") or []):
        return True
    if float(info.get("cost", 0) or 0) > 0:
        return True
    return ma_step_saw_walls(info, agents)


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


def _fmt_lidar(arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=float).reshape(-1)
    return f"max={a.max():.3f} {np.array2string(a, precision=3, suppress_small=True)}"


def classify_wall_geom_name(name: str) -> str:
    """Bucket a MuJoCo geom name that participates in wall-cost matching."""
    n = str(name)
    if "building" in n and "wall" in n:
        return "building_perimeter"
    if n.startswith("ltl_wall") or "ltl_wall" in n:
        return "arena_ltl"
    if n.startswith("wall") and "ltl" not in n:
        return "interior"
    if "wall" in n:
        return "other_wall"
    return "non_wall"


def expected_pseudo_lidar(dist: float, exp_gain: float = 0.5) -> float:
    """Match ``_accumulate_pseudo_lidar_reading`` with ``max_dist=None``."""
    return float(np.exp(-float(exp_gain) * float(dist)))


def collect_gremlin_wall_contacts(task: Any) -> list[dict[str, Any]]:
    """Live MuJoCo contacts: gremlin*obj ↔ geom with ``wall`` in the name."""
    model = getattr(task, "model", None)
    data = getattr(task, "data", None)
    if model is None or data is None:
        engine = getattr(task, "engine", None)
        if engine is not None:
            model = getattr(engine, "model", model)
            data = getattr(engine, "data", data)
    if model is None or data is None:
        return []
    out: list[dict[str, Any]] = []
    ncon = int(getattr(data, "ncon", 0))
    for con in data.contact[:ncon]:
        name1 = model.geom(con.geom1).name
        name2 = model.geom(con.geom2).name
        pair = (name1, name2)
        if not any("gremlin" in n for n in pair):
            continue
        if not any("wall" in n for n in pair):
            continue
        gremlin_name = name1 if "gremlin" in name1 else name2
        wall_name = name2 if "gremlin" in name1 else name1
        agent_id = None
        # Names look like gremlin0obj
        digits = "".join(ch for ch in gremlin_name if ch.isdigit())
        if digits:
            agent_id = int(digits)
        out.append(
            {
                "agent_id": agent_id,
                "gremlin": gremlin_name,
                "wall": wall_name,
                "wall_class": classify_wall_geom_name(wall_name),
            }
        )
    return out


def nearest_interior_wall_debug(task: Any, agent_idx: int) -> dict[str, Any] | None:
    """Closest interior ``Walls`` surface distance + expected lidar reading."""
    walls = getattr(task, "walls", None)
    if walls is None or not getattr(walls, "num", 0):
        return None
    agent = getattr(task, "agent", None)
    if agent is None:
        return None
    agent_xy = np.asarray(agent.get_agent_pos(agent_idx), dtype=float)[:2]
    exp_gain = float(getattr(getattr(task, "lidar_conf", None), "exp_gain", 0.5) or 0.5)
    best: dict[str, Any] | None = None
    for row in range(int(walls.num)):
        if hasattr(walls, "closest_surface_pos"):
            surf = np.asarray(walls.closest_surface_pos(agent_idx, row), dtype=float)
        else:
            surf = np.asarray(walls.pos[row], dtype=float)
        dist = float(np.linalg.norm(agent_xy - surf[:2]))
        los = None
        if hasattr(task, "_lidar_line_of_sight"):
            try:
                los = bool(task._lidar_line_of_sight(agent_idx, surf, walls, row))
            except Exception:  # noqa: BLE001 — debug only
                los = None
        cand = {
            "row": row,
            "dist": dist,
            "expected_lidar": expected_pseudo_lidar(dist, exp_gain),
            "los": los,
            "surface_xy": surf[:2].tolist(),
        }
        if best is None or dist < best["dist"]:
            best = cand
    return best


def _geom_name(task: Any, geom_id: int | None) -> str | None:
    if geom_id is None:
        return None
    model = getattr(task, "model", None)
    if model is None:
        return None
    try:
        return str(model.geom(int(geom_id)).name)
    except Exception:  # noqa: BLE001 — debug only
        return None


def first_hit_along_agent_to_target(
    task: Any,
    agent_idx: int,
    target_pos: np.ndarray,
) -> dict[str, Any] | None:
    """First lidar-observable geom on agent→target ray (LOS failure probe)."""
    if not hasattr(task, "_lidar_ray_first_observable_geom"):
        return None
    agent = getattr(task, "agent", None)
    if agent is None:
        return None
    agent_pos = np.asarray(agent.get_agent_pos(agent_idx), dtype=float)
    target = np.asarray(target_pos, dtype=float)
    if target.shape == (2,):
        target = np.concatenate([target, [float(agent_pos[2])]])
    vec = target - agent_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return {"geom_id": None, "geom_name": None, "hit_dist": 0.0, "wall_class": None}
    vec = vec / dist
    try:
        hit_geom, hit_dist = task._lidar_ray_first_observable_geom(
            agent_idx, agent_pos, vec, dist + 1e-4,
        )
    except Exception:  # noqa: BLE001 — debug only
        return None
    name = _geom_name(task, hit_geom)
    return {
        "geom_id": hit_geom,
        "geom_name": name,
        "hit_dist": float(hit_dist) if hit_dist is not None else None,
        "wall_class": classify_wall_geom_name(name) if name else None,
    }


def remaining_surface_debug(task: Any, agent_idx: int) -> list[dict[str, Any]]:
    """Per unrescued surface casualty: dist, LOS, expected lidar, live obs, sticky."""
    surface = getattr(task, "surface_casualtys", None)
    if surface is None:
        return []
    rescued = list(getattr(surface, "rescued", []) or [])
    agent = getattr(task, "agent", None)
    if agent is None:
        return []
    agent_xy = np.asarray(agent.get_agent_pos(agent_idx), dtype=float)[:2]
    exp_gain = float(getattr(getattr(task, "lidar_conf", None), "exp_gain", 0.5) or 0.5)
    last_seen = getattr(task, "_surface_last_seen", {}).get(agent_idx, {})
    sticky_active = getattr(task, "_surface_sticky_active", {}).get(agent_idx, set())
    rows: list[dict[str, Any]] = []
    for row, is_rescued in enumerate(rescued):
        if is_rescued:
            # Still report on rescue frame when lidar-skip lag keeps them visible.
            skip = getattr(task, "_casualty_lidar_skip_rows", {}).get(surface.name, frozenset())
            if row in skip:
                continue
        pos = np.asarray(surface.pos[row], dtype=float)
        dist = float(np.linalg.norm(agent_xy - pos[:2]))
        los = None
        first_hit = None
        if hasattr(task, "_lidar_line_of_sight"):
            try:
                target = pos if pos.shape[0] >= 3 else np.r_[pos[:2], 0.0]
                los = bool(task._lidar_line_of_sight(agent_idx, target, surface, row))
                if not los:
                    first_hit = first_hit_along_agent_to_target(task, agent_idx, target)
            except Exception:  # noqa: BLE001 — debug only
                los = None
        seen_xy = last_seen.get(row)
        rows.append(
            {
                "row": row,
                "dist": dist,
                "expected_lidar": expected_pseudo_lidar(dist, exp_gain),
                "los": los,
                "xy": pos[:2].tolist(),
                "sticky": row in sticky_active,
                "last_seen_xy": (
                    np.asarray(seen_xy, dtype=float)[:2].tolist() if seen_xy is not None else None
                ),
                "first_hit": first_hit,
                "rescued_live": bool(is_rescued),
            }
        )
    return rows


def print_ma_episode_done_debug(
    env: Any,
    info: dict[str, Any],
    *,
    step: int | None = None,
    saw_walls: bool | None = None,
    reach: Any = None,
    avoid: Any = None,
    entr_bldg_obs: bool = False,
    zone_compat: bool = False,
    strip_walls_avoid_lidar: bool = False,
) -> None:
    """Print termination diagnostics when an MA deploy episode ends."""
    task = sar_task(env)
    num_agents = getattr(task, "agent_num", 2)
    agents = [f"agent_{i}" for i in range(num_agents)]
    surface_geom = getattr(task, "surface_casualtys", None)
    entrapped_geom = getattr(task, "entrapped_casualtys", None)
    surface_rescued = list(surface_geom.rescued) if surface_geom is not None else None
    entrapped_rescued = list(entrapped_geom.rescued) if entrapped_geom is not None else None

    # Prefer live wrapper/model flags so debug matches agent features.
    if hasattr(env, "strip_walls_avoid_lidar"):
        strip_walls_avoid_lidar = bool(getattr(env, "strip_walls_avoid_lidar"))
    avoid_skip = {"walls"} if strip_walls_avoid_lidar else None

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
    if strip_walls_avoid_lidar:
        print("  strip_walls_avoid_lidar: True (walls omitted from avoid *features*)")

    # --- root-cause probes (wall cost vs lidar, blank all_surface reach) ---
    contacts = collect_gremlin_wall_contacts(task)
    print(f"  gremlin↔wall contacts now: {len(contacts)}")
    for c in contacts:
        print(
            f"    agent_{c['agent_id']}: {c['gremlin']} ↔ {c['wall']} "
            f"[{c['wall_class']}]"
        )
    if not contacts and wall_violation:
        print(
            "    (none live — cost may be first-frame-only / contact already cleared)"
        )

    if reach is not None or avoid is not None:
        lidar_dim = int(task.lidar_conf.num_bins)
        print(f"  reach set: {reach}")
        print(f"  avoid set: {avoid}")
        for agent_idx in range(num_agents):
            original_obs = sar_agent_obs(env, agent_idx)
            reach_obs = lidar_for_assignments(
                original_obs,
                reach or frozenset(),
                lidar_dim,
                agent_idx=agent_idx,
                num_agents=num_agents,
                for_reach=entr_bldg_obs,
                zone_compat=zone_compat,
            )
            avoid_obs = lidar_for_assignments(
                original_obs,
                avoid or frozenset(),
                lidar_dim,
                agent_idx=agent_idx,
                num_agents=num_agents,
                zone_compat=zone_compat,
                skip_props=avoid_skip,
            )
            print(f"  agent_{agent_idx} reach_lidar: {_fmt_lidar(reach_obs)}")
            print(f"  agent_{agent_idx} avoid_lidar: {_fmt_lidar(avoid_obs)}")
            surf_key = f"surface_casualtys_lidar_{agent_idx}"
            if surf_key in original_obs:
                surf_arr = np.asarray(original_obs[surf_key], dtype=float)
                peak_bin = int(np.argmax(surf_arr)) if surf_arr.size else -1
                print(
                    f"  agent_{agent_idx} surface_lidar: "
                    f"{_fmt_lidar(surf_arr)} peak_bin={peak_bin}"
                )
            walls_key = walls_lidar_key(agent_idx)
            walls_max = None
            if walls_key in original_obs:
                walls_arr = np.asarray(original_obs[walls_key], dtype=float)
                walls_max = float(walls_arr.max())
                print(
                    f"  agent_{agent_idx} walls_lidar: "
                    f"{_fmt_lidar(walls_arr)}"
                )
            nearest = nearest_interior_wall_debug(task, agent_idx)
            if nearest is not None:
                mismatch = (
                    walls_max is not None
                    and abs(walls_max - nearest["expected_lidar"]) > 0.15
                )
                print(
                    f"  agent_{agent_idx} nearest interior wall: "
                    f"row={nearest['row']} dist={nearest['dist']:.3f} "
                    f"expected_lidar={nearest['expected_lidar']:.3f} "
                    f"los={nearest['los']} "
                    f"walls_lidar_max={walls_max} "
                    f"{'MISMATCH' if mismatch else 'ok'}"
                )
            for row in remaining_surface_debug(task, agent_idx):
                hit = row.get("first_hit") or {}
                hit_bits = ""
                if row.get("los") is False and hit:
                    hit_bits = (
                        f" first_hit={hit.get('geom_name')} "
                        f"[{hit.get('wall_class')}] "
                        f"hit_dist={hit.get('hit_dist')}"
                    )
                print(
                    f"  agent_{agent_idx} remaining surface_{row['row']}: "
                    f"dist={row['dist']:.3f} expected_lidar={row['expected_lidar']:.3f} "
                    f"los={row['los']} sticky={row['sticky']} "
                    f"last_seen_xy={row['last_seen_xy']} "
                    f"rescued_live={row['rescued_live']} xy={row['xy']}"
                    f"{hit_bits}"
                )
