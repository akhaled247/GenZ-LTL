"""Roll out SAR policies and draw top-down multi-agent trajectories."""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]  # GenZ-LTL
_SPECRL = ROOT.parent / "SpecRLBench"
_SAR_SG = _SPECRL / "specbench" / "envs" / "zones" / "safety-gymnasium"
# Prefer SpecRL SAR safety-gymnasium fork over site-packages stock package.
sys.path[:0] = [
    str(ROOT / "src"),
    str(_SPECRL),
    str(_SAR_SG),
]

from deploy.coordinator import MultiAgentSARCoordinator  # noqa: E402
from deploy.eval_stack import build_sar_ltl_eval_stack  # noqa: E402
from deploy.feature_recipe import (  # noqa: E402
    apply_zone_compat_deploy_meta,
    attach_model_deploy_fields,
    ensure_sar_v1_indep_lidars,
    resolve_zone_compat,
)
from deploy.loading import load_model_for_deploy  # noqa: E402
from envs import make_env_safety  # noqa: E402
from envs.env_utils import get_env_attr  # noqa: E402
from envs.sar_deploy import check_rabinizer  # noqa: E402
from envs.seq_wrapper import (  # noqa: E402
    lidar_for_assignments,
    sar_agent_obs,
    sar_task,
)
from ltl import FixedSampler  # noqa: E402
from ltl.automata import LDBASequence  # noqa: E402
from model.agent import Agent  # noqa: E402
from sequence.search import ExhaustiveSearchSafety, NoPathsException  # noqa: E402
from sequence.search.exhaustive_search import strip_walls_from_reach_set  # noqa: E402
from utils.deploy_meta import FEAT_RECIPE_ZONE_COMPAT  # noqa: E402
from utils.train_device import resolve_training_device  # noqa: E402
from visualize.sar import draw_sar_trajectories, snapshot_sar_scene  # noqa: E402


DEFAULT_FORMULA = "(!surface_0 U entrapped_0) & F surface_0"


def _agent_num_from_env_id(env_id: str) -> int | None:
    m = re.search(r"MASAR(\d+)", env_id)
    if not m:
        return None
    return int(m.group(1))


def _episode_outcome(info: dict) -> str:
    if "success" in info:
        return "success"
    if "violation" in info:
        return "violation"
    return "not_finish"


def _assignment_props(assignment) -> set[str]:
    return set(assignment.get_true_propositions())


def _reach_props(reach) -> set[str]:
    if reach == LDBASequence.EPSILON:
        return set()
    props: set[str] = set()
    for assignment in reach:
        props |= _assignment_props(assignment)
    return props


def _avoid_props(avoid) -> set[str]:
    props: set[str] = set()
    for assignment in avoid:
        props |= _assignment_props(assignment)
    return props


def _props_csv(props: set[str] | frozenset[str]) -> str:
    return ",".join(sorted(props)) if props else "-"


def _format_stage(reach, avoid) -> str:
    if reach == LDBASequence.EPSILON:
        r = "ε"
    else:
        r = _props_csv(_reach_props(reach))
    a = _props_csv(_avoid_props(avoid))
    return f"R={{{r}}} A={{{a}}}"


def _format_sequence(seq) -> str:
    if seq is None:
        return "seq=None"
    stages = [_format_stage(reach, avoid) for reach, avoid in seq]
    return " → ".join(stages) if stages else "seq=[]"


def _active_reach_avoid(seq, propositions):
    """Return sanitized first-stage reach/avoid used for features (or None)."""
    if seq is None or len(seq) == 0:
        return None
    reach, avoid = seq[0]
    if reach == LDBASequence.EPSILON:
        return reach, avoid
    stripped = strip_walls_from_reach_set(reach, avoid, propositions)
    return stripped if stripped is not None else (reach, avoid)


def _casualty_xy_for_prop(task, prop: str) -> np.ndarray | None:
    """World XY of the casualty geom named by ``surface_i`` / ``entrapped_i``."""
    m = re.fullmatch(r"(surface|entrapped)_(\d+)", prop)
    if not m:
        return None
    kind, idx_s = m.group(1), int(m.group(2))
    geom = getattr(task, f"{kind}_casualtys", None)
    if geom is None or int(getattr(geom, "num", 0) or 0) <= idx_s:
        return None
    try:
        engine = geom.engine
        prefix = geom.name[:-1]
        body = engine.data.body(f"{prefix}{idx_s}")
        return np.asarray(body.xpos, dtype=float)[:2].copy()
    except Exception:  # noqa: BLE001
        pos = np.asarray(geom.pos[idx_s], dtype=float)
        return pos[:2].copy()


def _reach_goal_xy(task, reach) -> tuple[np.ndarray | None, list[str]]:
    if reach == LDBASequence.EPSILON:
        return None, []
    props = sorted(_reach_props(reach))
    for prop in props:
        xy = _casualty_xy_for_prop(task, prop)
        if xy is not None:
            return xy, props
    return None, props


def _agent_mat_xy(task, agent_idx: int = 0) -> np.ndarray:
    """2x2 world←ego rotation (columns = agent local axes in world)."""
    mat = np.asarray(task.agent.get_agent_mat(agent_idx), dtype=float)
    return mat[:2, :2]


def _lidar_peak_world_dir(lidar: np.ndarray, agent_mat_xy: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Peak lidar bin → unit world direction + strength + bin index."""
    arr = np.asarray(lidar, dtype=float).reshape(-1)
    peak_bin = int(np.argmax(arr))
    strength = float(arr[peak_bin])
    n = max(1, arr.size)
    ego_theta = (peak_bin / n) * (2.0 * math.pi)
    ego = np.array([math.cos(ego_theta), math.sin(ego_theta)], dtype=float)
    world = agent_mat_xy @ ego
    norm = float(np.linalg.norm(world))
    if norm < 1e-8:
        return np.zeros(2, dtype=float), strength, peak_bin
    return world / norm, strength, peak_bin


def _bearing_delta_deg(peak_dir: np.ndarray, goal_dir: np.ndarray) -> float | None:
    pn = float(np.linalg.norm(peak_dir))
    gn = float(np.linalg.norm(goal_dir))
    if pn < 1e-8 or gn < 1e-8:
        return None
    cos_a = float(np.clip(np.dot(peak_dir / pn, goal_dir / gn), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def _entrapped_masked(task, agent_idx: int, original_obs: dict) -> bool:
    """True when SpecRL wrapper zeros entrapped lidar (outside / not entered)."""
    key = f"entrapped_casualtys_lidar_{agent_idx}"
    if key not in original_obs:
        return False
    if float(np.max(original_obs[key])) > 1e-8:
        return False
    try:
        from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
            agent_inside_building_idx,
        )
        inside = agent_inside_building_idx(task, agent_idx) is not None
    except Exception:  # noqa: BLE001
        inside = False
    entered = bool(getattr(task, "_buildings_entered", set()))
    return not (inside or entered)


def _print_zone_sar_diff_map(*, train_env: str, eval_env: str, zone_compat: bool) -> None:
    if "PointLtlSafety" not in train_env or "MASAR" not in eval_env:
        return
    print(
        "=== Zone→SAR feature / semantics map ===\n"
        "  Zone train (PointLtlSafety*): features = agent(16) | reach_color | avoid_color\n"
        "    reach/avoid = spatially separate colored zones; avoid≠goal direction.\n"
        f"  SAR eval ({eval_env}) zone_compat={zone_compat}:\n"
        "    packing = agent(16) | reach | avoid  (no indep buildings/walls slices)\n"
        "    entrapped reach → max(terracotta_buildings_lidar, entrapped lidar)\n"
        "    walls stripped from avoid *features* by default (Büchi still has walls)\n"
        "  Zone policy learned: approach reach, flee avoid. If avoid≈goal bearing → flees goal.\n"
        "  Native zones 100% OK does NOT imply Zone→SAR OK (avoid semantics change).\n"
        "  SAR-native (sar_v1): buildings/walls are indep channels; reach entrapped is\n"
        "    MASKED to 0 until agent enters a building (SafetyGymWrapperMASAR).\n"
        "========================================"
    )


def _diagnose_reach_avoid(
    *,
    env,
    seq,
    propositions,
    agent_idx: int = 0,
    zone_compat: bool = False,
    entr_bldg_obs: bool = False,
    strip_walls_avoid_lidar: bool = False,
) -> dict:
    """Snapshot first-stage reach/avoid + lidar peaks vs true casualty bearing."""
    task = sar_task(env)
    active = _active_reach_avoid(seq, propositions)
    out: dict = {
        "sequence": _format_sequence(seq),
        "stage0": None,
        "reach_props": [],
        "avoid_props": [],
        "goal_xy": None,
        "origin": None,
        "peak_dir": None,
        "goal_dir": None,
        "avoid_dir": None,
        "buildings_dir": None,
        "peak_strength": 0.0,
        "avoid_strength": 0.0,
        "buildings_strength": 0.0,
        "peak_bin": -1,
        "delta_deg": None,
        "avoid_delta_deg": None,
        "buildings_delta_deg": None,
        "entrapped_masked": False,
        "avoid_goal_conflict": False,
        "warn": False,
        "ok": False,
        "note": "",
    }
    if active is None:
        out["note"] = "no active reach/avoid"
        return out
    reach, avoid = active
    out["stage0"] = _format_stage(reach, avoid)
    if reach != LDBASequence.EPSILON:
        out["reach_props"] = sorted(_reach_props(reach))
    out["avoid_props"] = sorted(_avoid_props(avoid))

    origin = np.asarray(task.agent.get_agent_pos(agent_idx), dtype=float)[:2].copy()
    out["origin"] = origin
    goal_xy, props = _reach_goal_xy(task, reach)
    out["goal_xy"] = goal_xy
    out["reach_props"] = props or out["reach_props"]

    if reach == LDBASequence.EPSILON:
        out["note"] = "epsilon reach stage"
        return out

    lidar_dim = int(task.lidar_conf.num_bins)
    original_obs = sar_agent_obs(env, agent_idx)
    num_agents = int(getattr(task, "agent_num", 1) or 1)
    mat_xy = _agent_mat_xy(task, agent_idx)
    masked = _entrapped_masked(task, agent_idx, original_obs)
    out["entrapped_masked"] = masked

    reach_lidar = lidar_for_assignments(
        original_obs, reach, lidar_dim, agent_idx=agent_idx, num_agents=num_agents,
        for_reach=entr_bldg_obs,
        zone_compat=zone_compat,
    )
    avoid_skip = {"walls"} if strip_walls_avoid_lidar else None
    avoid_lidar = lidar_for_assignments(
        original_obs, avoid, lidar_dim, agent_idx=agent_idx, num_agents=num_agents,
        zone_compat=zone_compat,
        skip_props=avoid_skip,
    )
    peak_dir, strength, peak_bin = _lidar_peak_world_dir(reach_lidar, mat_xy)
    avoid_dir, avoid_strength, _ = _lidar_peak_world_dir(avoid_lidar, mat_xy)
    out["peak_dir"] = peak_dir
    out["peak_strength"] = strength
    out["peak_bin"] = peak_bin
    out["avoid_dir"] = avoid_dir
    out["avoid_strength"] = avoid_strength
    out["strip_walls_avoid_lidar"] = bool(strip_walls_avoid_lidar)

    bldg_key = f"terracotta_buildings_lidar_{agent_idx}"
    if bldg_key in original_obs:
        bldg_dir, bldg_strength, _ = _lidar_peak_world_dir(original_obs[bldg_key], mat_xy)
        out["buildings_dir"] = bldg_dir
        out["buildings_strength"] = bldg_strength

    # Decompose avoid into walls vs surface (pathology: walls often ≈ building bearing).
    walls_key = f"walls_lidar_{agent_idx}"
    surf_key = f"surface_casualtys_lidar_{agent_idx}"
    out["walls_avoid_strength"] = 0.0
    out["surface_avoid_strength"] = 0.0
    out["walls_avoid_delta_deg"] = None
    out["surface_avoid_delta_deg"] = None
    out["indep_walls_strength"] = 0.0
    if walls_key in original_obs and "walls" in out["avoid_props"]:
        w_dir, w_s, _ = _lidar_peak_world_dir(original_obs[walls_key], mat_xy)
        out["walls_avoid_strength"] = w_s
        out["indep_walls_strength"] = w_s  # same sensor; also indep channel in sar_v1 L1+
        out["walls_avoid_dir"] = w_dir
    if surf_key in original_obs and any(p.startswith("surface_") for p in out["avoid_props"]):
        s_dir, s_s, _ = _lidar_peak_world_dir(original_obs[surf_key], mat_xy)
        out["surface_avoid_strength"] = s_s
        out["surface_avoid_dir"] = s_dir

    # Feature packing summary (sar_v1 L1 = 80d with walls duplicated into avoid).
    include_walls = walls_key in original_obs and not zone_compat
    out["feat_layout"] = (
        "agent|reach|avoid (zone_compat 48d)"
        if zone_compat
        else (
            "agent|buildings|walls|reach|avoid (sar_v1 L1+ 80d)"
            if include_walls
            else "agent|buildings|reach|avoid (sar_v1 64d)"
        )
    )
    out["walls_duplicated"] = bool(
        include_walls and "walls" in out["avoid_props"] and not strip_walls_avoid_lidar
    )
    out["zone_compat"] = bool(zone_compat)
    if strip_walls_avoid_lidar:
        out["feat_layout"] = (out["feat_layout"] or "") + " [walls stripped from avoid lidar]"

    if goal_xy is None:
        out["note"] = f"no geom for reach props {props}"
        out["warn"] = True
        return out

    goal_vec = goal_xy - origin
    gn = float(np.linalg.norm(goal_vec))
    if gn < 1e-8:
        out["note"] = "agent already on goal"
        return out
    goal_dir = goal_vec / gn
    out["goal_dir"] = goal_dir
    delta = _bearing_delta_deg(peak_dir, goal_dir)
    out["delta_deg"] = delta
    out["avoid_delta_deg"] = _bearing_delta_deg(avoid_dir, goal_dir)
    if out["buildings_dir"] is not None:
        out["buildings_delta_deg"] = _bearing_delta_deg(out["buildings_dir"], goal_dir)
    if out.get("walls_avoid_dir") is not None:
        out["walls_avoid_delta_deg"] = _bearing_delta_deg(out["walls_avoid_dir"], goal_dir)
    if out.get("surface_avoid_dir") is not None:
        out["surface_avoid_delta_deg"] = _bearing_delta_deg(out["surface_avoid_dir"], goal_dir)

    notes: list[str] = []
    # Expected SAR partial-obs mask (not a bug by itself).
    if masked and any(p.startswith("entrapped_") for p in out["reach_props"]):
        bdd = out["buildings_delta_deg"]
        bds = out["buildings_strength"]
        notes.append(
            f"entrapped lidar MASKED outside building (expected); "
            f"buildings peak strength={bds:.3f}"
            + (f" Δ={bdd:.0f}°" if bdd is not None else "")
        )
        if bds > 1e-6 and bdd is not None and bdd < 45.0:
            out["ok"] = True
        elif not zone_compat:
            # sar_v1 still has indep buildings channel — reach=0 is fine.
            out["ok"] = True

    if strength < 1e-6 and not masked:
        notes.append("reach lidar all-zero (unexpected — check keys / zone_compat)")
        out["warn"] = True
    elif strength < 1e-6 and zone_compat:
        notes.append("zone_compat reach all-zero (buildings key missing?)")
        out["warn"] = True
    elif delta is not None and delta > 90.0 and strength > 1e-6:
        notes.append(f"reach lidar peak OPPOSITE goal (Δ={delta:.0f}°)")
        out["warn"] = True
    elif delta is not None and strength > 1e-6:
        notes.append(f"reach lidar ≈ goal (Δ={delta:.0f}°)")
        out["ok"] = True

    # Walls-in-avoid co-aligned with building goal (feature avoid after optional strip).
    ad = out["avoid_delta_deg"]
    wad = out["walls_avoid_delta_deg"]
    if strip_walls_avoid_lidar and wad is not None and wad < 45.0 and out["walls_avoid_strength"] > 0.05:
        notes.append(
            f"walls sensor still ≈ goal (Δ={wad:.0f}° s={out['walls_avoid_strength']:.3f}) "
            f"but STRIPPED from avoid features (ablation on)"
        )
    if (
        avoid_strength > 0.05
        and ad is not None
        and ad < 45.0
        and "walls" in out["avoid_props"]
        and not strip_walls_avoid_lidar
    ):
        out["avoid_goal_conflict"] = True
        out["warn"] = True
        if zone_compat:
            notes.append(
                f"AVOID≈GOAL (zone_compat): avoid Δ={ad:.0f}° s={avoid_strength:.3f}; "
                f"walls in avoid — Zone policy learned flee-avoid → flees building"
            )
        else:
            notes.append(
                f"AVOID≈GOAL (SAR-native): avoid Δ={ad:.0f}° s={avoid_strength:.3f} "
                f"(walls_s={out['walls_avoid_strength']:.3f} "
                f"Δ={out['walls_avoid_delta_deg'] if out['walls_avoid_delta_deg'] is not None else -1:.0f}°; "
                f"surf_s={out['surface_avoid_strength']:.3f}); "
                f"walls also in indep channel (duplicated={out['walls_duplicated']}) — "
                f"approach building = strong avoid+walls lidar; lag/cost may bias retreat"
            )
    elif avoid_strength > 1e-6 and ad is not None:
        notes.append(
            f"avoid peak Δ={ad:.0f}° s={avoid_strength:.3f} "
            f"(walls_s={out['walls_avoid_strength']:.3f}, surf_s={out['surface_avoid_strength']:.3f})"
        )

    out["note"] = "; ".join(notes) if notes else ""
    return out


def _episode_title(outcome: str, diag: dict) -> str:
    stage = diag.get("stage0") or "?"
    delta = diag.get("delta_deg")
    delta_s = f" RΔ{delta:.0f}°" if delta is not None and diag.get("peak_strength", 0) > 1e-6 else ""
    if diag.get("entrapped_masked") and diag.get("buildings_delta_deg") is not None:
        delta_s = f" BΔ{diag['buildings_delta_deg']:.0f}°"
    go = diag.get("motion_delta_deg")
    go_s = f" goΔ{go:.0f}°" if go is not None else ""
    flag = ""
    if diag.get("motion_opposite"):
        flag = " GO≠GOAL"
    elif diag.get("avoid_goal_conflict"):
        flag = " AVOID≈GOAL"
    elif diag.get("warn"):
        flag = " ⚠"
    elif diag.get("entrapped_masked"):
        flag = " masked"
    return f"{outcome} | {stage}{delta_s}{go_s}{flag}"


def _print_diag(ep: int, formula: str, diag: dict, outcome: str) -> None:
    go = diag.get("motion_delta_deg")
    go_s = f"goΔ={go:.0f}°" if go is not None else "goΔ=n/a"
    act = diag.get("first_action")
    act_s = f"act0={np.asarray(act).round(3).tolist()}" if act is not None else "act0=n/a"
    print(
        f"[ep {ep}] {outcome} | formula={formula}\n"
        f"  sequence: {diag.get('sequence')}\n"
        f"  feature stage0: {diag.get('stage0')} | layout={diag.get('feat_layout', '?')}\n"
        f"  reach_s={diag.get('peak_strength'):.3f} "
        f"bldg_s={diag.get('buildings_strength'):.3f} "
        f"avoid_s={diag.get('avoid_strength'):.3f} "
        f"(walls={diag.get('walls_avoid_strength', 0):.3f} "
        f"surf={diag.get('surface_avoid_strength', 0):.3f}) | {go_s} | {act_s}\n"
        f"  {diag.get('note')}"
    )
    # Auto-trace when avoidance dominates building cue or agent fled goal.
    avoid_dom = (
        diag.get("avoid_strength", 0) > diag.get("buildings_strength", 0)
        and diag.get("avoid_strength", 0) > 0.05
    )
    fled = bool(diag.get("motion_opposite"))
    if diag.get("trace_stage0") or avoid_dom or fled:
        _print_stage0_trace(diag)


def _print_stage0_trace(diag: dict) -> None:
    """Dump stage0 cues when avoid dominates buildings or motion flees goal."""
    wad = diag.get("walls_avoid_delta_deg")
    sad = diag.get("surface_avoid_delta_deg")
    bdd = diag.get("buildings_delta_deg")
    print(
        "  [stage0-trace]\n"
        f"    buildings: s={diag.get('buildings_strength', 0):.3f} "
        f"Δgoal={bdd if bdd is not None else float('nan'):.0f}°\n"
        f"    walls→avoid: s={diag.get('walls_avoid_strength', 0):.3f} "
        f"Δgoal={wad if wad is not None else float('nan'):.0f}° "
        f"(duplicated_indep={diag.get('walls_duplicated')}, "
        f"stripped={diag.get('strip_walls_avoid_lidar')})\n"
        f"    surface→avoid: s={diag.get('surface_avoid_strength', 0):.3f} "
        f"Δgoal={sad if sad is not None else float('nan'):.0f}°\n"
        f"    reach(entrapped): s={diag.get('peak_strength', 0):.3f} "
        f"masked={diag.get('entrapped_masked')}\n"
        f"    RCO note: ModelSafety scores seq as V−λC; high wall cost-to-go near "
        f"building can prefer heading away even when buildings lidar points at goal."
    )


def _set_strip_walls_avoid_lidar(env, model, enabled: bool) -> None:
    """Propagate eval ablation: walls stay in Büchi/WC but leave avoid *features*."""
    enabled = bool(enabled)
    if model is not None:
        model.strip_walls_avoid_lidar = enabled
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "strip_walls_avoid_lidar"):
            cur.strip_walls_avoid_lidar = enabled
        cur = getattr(cur, "env", None)


def _set_entr_bldg_obs(env, model, enabled: bool) -> None:
    """Pool buildings into entrapped reach lidar (SAR-native seeking cue)."""
    enabled = bool(enabled)
    if model is not None:
        model.entr_bldg_obs = enabled
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "entr_bldg_obs"):
            cur.entr_bldg_obs = enabled
        cur = getattr(cur, "env", None)


def _print_run_aggregate(diags: list[dict], outcomes: list[str]) -> None:
    """End-of-run stats for avoid≈goal / flee pattern."""
    n = len(diags)
    if n == 0:
        return
    avoid_near = 0  # |avoidΔgoal| < 20°
    walls_near = 0
    go_opp = 0
    align_flee = 0  # avoid near goal AND go opposite
    align_ok = 0
    oppose_ok = 0
    go_deltas = []
    avoid_deltas = []
    for d, oc in zip(diags, outcomes):
        ad = d.get("avoid_delta_deg")
        wad = d.get("walls_avoid_delta_deg")
        gd = d.get("motion_delta_deg")
        if ad is not None:
            avoid_deltas.append(ad)
            if ad < 20.0:
                avoid_near += 1
                if gd is not None and gd > 90.0:
                    align_flee += 1
                elif oc == "success":
                    align_ok += 1
            elif ad > 90.0 and oc == "success":
                oppose_ok += 1
        if wad is not None and wad < 20.0:
            walls_near += 1
        if gd is not None:
            go_deltas.append(gd)
            if gd > 90.0:
                go_opp += 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        "=== run aggregate (stage0) ===\n"
        f"  n={n}  avoidΔgoal<20°: {avoid_near}/{n} ({100 * avoid_near / n:.0f}%)  "
        f"wallsΔgoal<20°: {walls_near}/{n} ({100 * walls_near / n:.0f}%)\n"
        f"  goΔ>90° (fled goal): {go_opp}/{n} ({100 * go_opp / n:.0f}%)\n"
        f"  when avoid≈goal (<20°): fled={align_flee}  success={align_ok}  |  "
        f"when avoid opposed (>90°) success={oppose_ok}\n"
        f"  mean avoidΔgoal={_mean(avoid_deltas):.1f}°  mean goΔ={_mean(go_deltas):.1f}°\n"
        "=============================="
    )


def _enrich_diag_with_motion(diag: dict, paths: dict[str, list[np.ndarray]], *, agent_key: str = "agent_0") -> dict:
    """Attach early-path travel direction vs goal (after rollout has points)."""
    pts = paths.get(agent_key) or []
    goal_xy = diag.get("goal_xy")
    origin = diag.get("origin")
    diag["motion_dir"] = None
    diag["motion_delta_deg"] = None
    diag["motion_opposite"] = False
    if origin is None:
        origin = np.asarray(pts[0], dtype=float)[:2] if pts else None
        diag["origin"] = origin
    if origin is None or goal_xy is None or len(pts) < 2:
        return diag

    origin = np.asarray(origin, dtype=float)[:2]
    goal_xy = np.asarray(goal_xy, dtype=float)[:2]
    goal_vec = goal_xy - origin
    gn = float(np.linalg.norm(goal_vec))
    if gn < 1e-8:
        return diag
    goal_dir = goal_vec / gn
    diag["goal_dir"] = goal_dir

    # Use first segment that moves enough; else mean of early steps.
    motion = np.zeros(2, dtype=float)
    for i in range(1, len(pts)):
        step = np.asarray(pts[i], dtype=float)[:2] - np.asarray(pts[i - 1], dtype=float)[:2]
        if float(np.linalg.norm(step)) > 0.02:
            # Average first ~0.5–1.0 units of travel for stable heading.
            traveled = 0.0
            acc = np.zeros(2, dtype=float)
            for j in range(i, len(pts)):
                d = np.asarray(pts[j], dtype=float)[:2] - np.asarray(pts[j - 1], dtype=float)[:2]
                acc += d
                traveled += float(np.linalg.norm(d))
                if traveled >= 0.6:
                    break
            motion = acc
            break
    if float(np.linalg.norm(motion)) < 1e-8:
        # Fallback: net displacement start→end (coarse).
        motion = np.asarray(pts[-1], dtype=float)[:2] - origin
    mn = float(np.linalg.norm(motion))
    if mn < 1e-8:
        note = diag.get("note") or ""
        diag["note"] = (note + "; ").lstrip("; ") + "agent barely moved"
        return diag

    motion_dir = motion / mn
    diag["motion_dir"] = motion_dir
    md = _bearing_delta_deg(motion_dir, goal_dir)
    diag["motion_delta_deg"] = md
    if md is not None and md > 90.0:
        diag["motion_opposite"] = True
        diag["warn"] = True
        note = diag.get("note") or ""
        diag["note"] = (
            (note + "; " if note else "")
            + f"agent went OPPOSITE goal (goΔ={md:.0f}°)"
        )
    elif md is not None:
        note = diag.get("note") or ""
        diag["note"] = (note + "; " if note else "") + f"agent vs goal goΔ={md:.0f}°"
    return diag


def _collect_agent_xy(task, num_agents: int) -> dict[str, np.ndarray]:
    return {
        f"agent_{i}": np.asarray(task.agent.get_agent_pos(i), dtype=float)[:2].copy()
        for i in range(num_agents)
    }


def _append_xy(
    paths: dict[str, list[np.ndarray]],
    xy_by_agent: dict[str, np.ndarray],
) -> None:
    for key, xy in xy_by_agent.items():
        paths.setdefault(key, []).append(xy)


def _grid_shape(n: int) -> tuple[int, int]:
    cols = min(4, max(1, n))
    rows = int(math.ceil(n / cols))
    return cols, rows


def _overlay_from_diag(diag: dict) -> dict | None:
    if diag.get("origin") is None:
        return None
    peak = diag.get("peak_dir")
    # When entrapped masked, show buildings peak as the effective approach cue.
    if diag.get("entrapped_masked") and diag.get("buildings_strength", 0) > 1e-6:
        peak = diag.get("buildings_dir")
    return {
        "origin": diag["origin"],
        "goal_xy": diag.get("goal_xy"),
        "peak_dir": peak if diag.get("peak_strength", 0) > 1e-6 or diag.get("entrapped_masked") else None,
        "goal_dir": diag.get("goal_dir"),
        "avoid_dir": diag.get("avoid_dir") if diag.get("avoid_strength", 0) > 1e-6 else None,
        "motion_dir": diag.get("motion_dir"),
        "arrow_scale": 1.4,
    }


def _rollout_sa(
    *,
    train_env: str,
    eval_env: str,
    exp: str,
    seed: int,
    formula: str,
    num_episodes: int,
    deterministic: bool,
    zone_compat: bool,
    device: str,
    debug_reach_avoid: bool,
    trace_stage0: bool = False,
    strip_walls_avoid_lidar: bool = False,
    entr_bldg_obs: bool | None = None,
):
    check_rabinizer()
    env, model, search, props, _algo = build_sar_ltl_eval_stack(
        train_env,
        exp,
        seed,
        formula,
        eval_env=eval_env,
        flat=True,
        zone_compat=zone_compat,
    )
    if device != "cpu":
        model = model.to(resolve_training_device(device))
    agent = Agent(env, model, search=search, propositions=props, verbose=debug_reach_avoid)
    # Match feature-recipe flags used by Agent.forward / sar_preprocess_for_deploy.
    zone_compat_eff = bool(
        getattr(model, "feat_recipe", None) == FEAT_RECIPE_ZONE_COMPAT or zone_compat
    )
    # Ablations (eval-time). zone_compat forces entr_bldg off (buildings already = reach).
    if entr_bldg_obs is None:
        entr_bldg_eff = bool(getattr(model, "entr_bldg_obs", False)) and not zone_compat_eff
    else:
        entr_bldg_eff = bool(entr_bldg_obs) and not zone_compat_eff
    _set_entr_bldg_obs(env, model, entr_bldg_eff)
    _set_strip_walls_avoid_lidar(env, model, strip_walls_avoid_lidar)
    if debug_reach_avoid:
        _print_zone_sar_diff_map(
            train_env=train_env, eval_env=eval_env, zone_compat=zone_compat_eff,
        )
        print(
            f"Ablations: zone_compat={zone_compat_eff} "
            f"strip_walls_avoid_lidar={strip_walls_avoid_lidar} "
            f"entr_bldg_obs={entr_bldg_eff}"
        )

    scenes, paths_list, titles, overlays = [], [], [], []
    diags, outcomes = [], []
    success = violation = unreachable = 0
    pbar = trange(num_episodes)
    for i in pbar:
        obs, info = env.reset(seed=seed + i), {}
        agent.reset()
        task = sar_task(env)
        num_agents = int(getattr(task, "agent_num", 1) or 1)
        scenes.append(snapshot_sar_scene(task))
        paths: dict[str, list[np.ndarray]] = {}
        _append_xy(paths, _collect_agent_xy(task, num_agents))
        diag = {
            "sequence": "seq=None",
            "stage0": None,
            "note": "no steps",
            "delta_deg": None,
            "peak_bin": -1,
            "peak_strength": 0.0,
            "avoid_strength": 0.0,
            "buildings_strength": 0.0,
            "origin": None,
            "warn": False,
            "entrapped_masked": False,
        }
        done = False
        first = True
        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=deterministic)
                if first:
                    diag = _diagnose_reach_avoid(
                        env=env, seq=agent.sequence, propositions=props, agent_idx=0,
                        zone_compat=zone_compat_eff, entr_bldg_obs=entr_bldg_eff,
                        strip_walls_avoid_lidar=strip_walls_avoid_lidar,
                    )
                    diag["first_action"] = np.asarray(action).flatten().copy()
                    diag["trace_stage0"] = bool(trace_stage0)
                    first = False
                action = np.asarray(action).flatten()
                if action.shape == (1,):
                    action = action[0]
                obs, _reward, done, info = env.step(action)
                _append_xy(paths, _collect_agent_xy(task, num_agents))
            except NoPathsException:
                unreachable += 1
                done = True
        paths_list.append(paths)
        outcome = _episode_outcome(info)
        diag = _enrich_diag_with_motion(diag, paths)
        titles.append(_episode_title(outcome, diag))
        overlays.append(_overlay_from_diag(diag))
        diags.append(diag)
        outcomes.append(outcome)
        if debug_reach_avoid:
            _print_diag(i, formula, diag, outcome)
        if "success" in info:
            success += 1
        elif "violation" in info:
            violation += 1
        pbar.set_postfix({"S": success / (i + 1), "V": violation / (i + 1)})

    env.close()
    print(f"Formula: {formula}, Success: {success}, Violation: {violation}, Unreachable: {unreachable}")
    _print_run_aggregate(diags, outcomes)
    print(
        "Note: walls forced into Büchi avoid by sanitize — (!walls U ...) no-ops. "
        "Use --strip-walls-avoid-lidar to drop walls from avoid *features* only (WC still terminates). "
        "SAR entrapped reach masked until enter building; --entr-bldg-obs pools buildings into reach."
    )
    if "PointLtlSafety" in train_env and "MASAR" in eval_env and not zone_compat_eff:
        print(
            "Hint: Zone→SAR needs --zone-compat (48-d packing). "
            "Then try --strip-walls-avoid-lidar to restore avoid≠goal."
        )
    return scenes, paths_list, titles, overlays


def _rollout_ma(
    *,
    train_env: str,
    eval_env: str,
    exp: str,
    seed: int,
    formula: str,
    num_episodes: int,
    deterministic: bool,
    zone_compat: bool,
    device: str,
    debug_reach_avoid: bool,
    trace_stage0: bool = False,
    strip_walls_avoid_lidar: bool = False,
    entr_bldg_obs: bool | None = None,
):
    check_rabinizer()
    if device != "cpu":
        device = resolve_training_device(device)

    model, deploy_meta, _store = load_model_for_deploy(
        train_env, exp, seed, formula, device=device,
    )
    deploy_meta = dict(deploy_meta)
    deploy_meta.setdefault("train_env", train_env)
    deploy_meta = ensure_sar_v1_indep_lidars(deploy_meta, lidar_bins=16)
    zone_compat = resolve_zone_compat(train_env, zone_compat)
    if zone_compat:
        deploy_meta = apply_zone_compat_deploy_meta(deploy_meta, lidar_bins=16)
    attach_model_deploy_fields(model, deploy_meta)

    if entr_bldg_obs is None:
        entr_bldg_eff = bool(deploy_meta.get("entr_bldg_obs", False)) and not zone_compat
    else:
        entr_bldg_eff = bool(entr_bldg_obs) and not zone_compat
    sampler = FixedSampler.partial(formula)
    env = make_env_safety(
        eval_env,
        sampler,
        flat=False,
        sar_env_backend="specrl",
        max_steps=2500,
        entr_bldg_obs=entr_bldg_eff,
        zone_compat=zone_compat,
    )
    _set_entr_bldg_obs(env, model, entr_bldg_eff)
    _set_strip_walls_avoid_lidar(env, model, strip_walls_avoid_lidar)
    num_agents = int(getattr(sar_task(env), "agent_num", 2) or 2)
    props = get_env_attr(env, "get_propositions")()
    search = ExhaustiveSearchSafety(env, model, props, num_loops=2, device=device)
    coordinator = MultiAgentSARCoordinator(
        env, model, search, props, num_agents, verbose=debug_reach_avoid, device=device,
    )
    if debug_reach_avoid:
        _print_zone_sar_diff_map(
            train_env=train_env, eval_env=eval_env, zone_compat=zone_compat,
        )
        print(
            f"Ablations: zone_compat={zone_compat} "
            f"strip_walls_avoid_lidar={strip_walls_avoid_lidar} "
            f"entr_bldg_obs={entr_bldg_eff}"
        )

    scenes, paths_list, titles, overlays = [], [], [], []
    diags, outcomes = [], []
    success = violation = unreachable = 0
    pbar = trange(num_episodes)
    for i in pbar:
        obs, info = env.reset(seed=seed + i), {}
        coordinator.reset()
        task = sar_task(env)
        scenes.append(snapshot_sar_scene(task))
        paths: dict[str, list[np.ndarray]] = {}
        _append_xy(paths, _collect_agent_xy(task, num_agents))
        diag = {
            "sequence": "seq=None",
            "stage0": None,
            "note": "no steps",
            "delta_deg": None,
            "peak_bin": -1,
            "peak_strength": 0.0,
            "avoid_strength": 0.0,
            "buildings_strength": 0.0,
            "origin": None,
            "warn": False,
            "entrapped_masked": False,
        }
        done = False
        first = True
        while not done:
            try:
                action = coordinator.get_action(obs, info, deterministic=deterministic)
                if first:
                    diag = _diagnose_reach_avoid(
                        env=env, seq=coordinator.sequence, propositions=props, agent_idx=0,
                        zone_compat=zone_compat, entr_bldg_obs=entr_bldg_eff,
                        strip_walls_avoid_lidar=strip_walls_avoid_lidar,
                    )
                    if isinstance(action, dict):
                        diag["first_action"] = np.asarray(
                            action.get("agent_0", next(iter(action.values())))
                        ).flatten().copy()
                    else:
                        diag["first_action"] = np.asarray(action).flatten().copy()
                    diag["trace_stage0"] = bool(trace_stage0)
                    first = False
                obs, _reward, done, info = env.step(action)
                _append_xy(paths, _collect_agent_xy(task, num_agents))
            except NoPathsException:
                unreachable += 1
                done = True
        paths_list.append(paths)
        outcome = _episode_outcome(info)
        diag = _enrich_diag_with_motion(diag, paths)
        titles.append(_episode_title(outcome, diag))
        overlays.append(_overlay_from_diag(diag))
        diags.append(diag)
        outcomes.append(outcome)
        if debug_reach_avoid:
            _print_diag(i, formula, diag, outcome)
        if "success" in info:
            success += 1
        elif "violation" in info:
            violation += 1
        pbar.set_postfix({"S": success / (i + 1), "V": violation / (i + 1)})

    env.close()
    print(f"Formula: {formula}, Success: {success}, Violation: {violation}, Unreachable: {unreachable}")
    _print_run_aggregate(diags, outcomes)
    print(
        "Note: use --strip-walls-avoid-lidar / --entr-bldg-obs for Phase-0 ablations. "
        "WC termination unchanged when walls stripped from avoid features."
    )
    return scenes, paths_list, titles, overlays


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw SAR top-down multi-agent trajectories")
    parser.add_argument("--env", type=str, default="PointLTL1MASAR1WC-v0")
    parser.add_argument("--train-env", type=str, default=None)
    parser.add_argument("--eval-env", type=str, default=None)
    parser.add_argument("--exp", type=str, default="GenZ-LTL")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--formula", type=str, default=DEFAULT_FORMULA)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--zone-compat",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--debug-reach-avoid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-episode Büchi reach/avoid + lidar peak vs true goal bearing; "
             "overlay arrows on plot (magenta=peak lidar, green=true goal).",
    )
    parser.add_argument(
        "--trace-stage0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Always print stage0 channel decomposition (walls/surface/buildings). "
             "Also auto-prints when avoid_s>bldg_s or goΔ>90°.",
    )
    parser.add_argument(
        "--strip-walls-avoid-lidar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Eval ablation: omit walls from avoid *feature* lidar (Büchi/WC cost unchanged). "
             "Restores Zone avoid≠goal assumption for Zone→SAR.",
    )
    parser.add_argument(
        "--entr-bldg-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pool buildings into entrapped reach lidar (SAR-native). "
             "Ignored under zone_compat (buildings already remapped to reach).",
    )
    args = parser.parse_args()

    eval_env = args.eval_env or args.env
    train_env = args.train_env or args.env
    formula = args.formula
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    n_agents = _agent_num_from_env_id(eval_env)
    use_ma = (n_agents is not None and n_agents > 1) or (
        train_env != eval_env and (_agent_num_from_env_id(eval_env) or 1) > 1
    )
    # Cold SA→MA: train MASAR1, eval MASAR2
    if "MASAR" in eval_env and (_agent_num_from_env_id(eval_env) or 1) > 1:
        use_ma = True

    # Auto zone_compat when Zone train → SAR eval unless user forced False after True default...
    # Keep explicit: user must pass --zone-compat for Zone→SAR (as before).
    rollout = _rollout_ma if use_ma else _rollout_sa
    scenes, paths_list, titles, overlays = rollout(
        train_env=train_env,
        eval_env=eval_env,
        exp=args.exp,
        seed=seed,
        formula=formula,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        zone_compat=args.zone_compat,
        device=args.device,
        debug_reach_avoid=args.debug_reach_avoid,
        trace_stage0=args.trace_stage0,
        strip_walls_avoid_lidar=args.strip_walls_avoid_lidar,
        entr_bldg_obs=args.entr_bldg_obs,
    )

    cols, rows = _grid_shape(len(scenes))
    fig = draw_sar_trajectories(
        scenes, paths_list, titles, cols, rows,
        overlays=overlays if args.debug_reach_avoid else None,
    )
    tag = ""
    if args.strip_walls_avoid_lidar:
        tag += "_nowallsavoid"
    if args.entr_bldg_obs:
        tag += "_entrbldg"
    if args.zone_compat:
        tag += "_zc"
    out = args.out or (
        f"experiments/rco/{train_env}/{args.exp}/{eval_env}_s{seed}{tag}_trajectories.png"
    )
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
