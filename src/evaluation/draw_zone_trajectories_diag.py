"""Zone-native reach/avoid bearing diag (sidecar).

Confirms pretrained Zone policy sees avoid ≠ reach/goal (contrast SAR zone_compat).

Revert: delete this file. Does NOT modify draw_zone_trajectories.py.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from config import model_configs
from envs import make_env, make_env_safety
from envs.env_utils import get_env_attr, find_builder
from ltl import FixedSampler
from ltl.automata import LDBASequence
from model.agent import Agent
from model.model import build_model, build_model_safety
from sequence.search import ExhaustiveSearchSafety, NoPathsException
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories


DEFAULT_ENV = "PointLtlSafety2-v0"
DEFAULT_EXP = "GenZ-LTL"
DEFAULT_FORMULA = "!green U ((blue | magenta) & (!green U yellow))"


def _props_from_assignments(assignments) -> list[str]:
    if assignments == LDBASequence.EPSILON:
        return []
    names: list[str] = []
    for a in assignments:
        for p in a.to_string():
            if p and p not in names:
                names.append(p)
    return names


def _format_stage(reach, avoid) -> str:
    r = ",".join(_props_from_assignments(reach)) or ("ε" if reach == LDBASequence.EPSILON else "-")
    a = ",".join(_props_from_assignments(avoid)) or "-"
    return f"R={{{r}}} A={{{a}}}"


def _zone_lidar_keys(assignments, *, for_avoid: bool = False) -> list[str]:
    """Match LDBAWrapper.pre_process_obs_zones key construction."""
    if assignments == LDBASequence.EPSILON:
        return []
    keys = []
    for a in assignments:
        labels = a.to_string()
        if not labels:
            continue
        if for_avoid:
            keys.append("_".join(sorted(labels)) + "_zones_lidar")
        else:
            keys.append(labels[0] + "_zones_lidar")
    return keys


def _max_lidar(original_obs: dict, keys: list[str], lidar_dim: int) -> np.ndarray:
    available = [np.asarray(original_obs[k], dtype=float).reshape(-1) for k in keys if k in original_obs]
    if not available:
        return np.zeros(lidar_dim, dtype=float)
    return np.max(np.vstack(available), axis=0)


def _lidar_peak_world_dir(lidar: np.ndarray, agent_mat_xy: np.ndarray) -> tuple[np.ndarray, float, int]:
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


def _bearing_delta_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-8 or bn < 1e-8:
        return None
    cos_a = float(np.clip(np.dot(a / an, b / bn), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def _goal_xy_for_reach(zone_positions: dict, reach_props: list[str]) -> np.ndarray | None:
    """Map color prop → layout key containing that color and 'zone'."""
    for color in reach_props:
        for key, xy in zone_positions.items():
            if color in key and "zone" in key:
                return np.asarray(xy, dtype=float)[:2].copy()
    return None


def _task_from_env(env):
    builder = find_builder(env)
    if builder is not None and hasattr(builder, "task"):
        return builder.task
    return get_env_attr(env, "task")


def _diagnose_zone_stage0(env, seq, zone_positions: dict) -> dict:
    out: dict = {
        "stage0": None,
        "reach_props": [],
        "avoid_props": [],
        "origin": None,
        "goal_xy": None,
        "goal_dir": None,
        "reach_dir": None,
        "avoid_dir": None,
        "motion_dir": None,
        "reach_s": 0.0,
        "avoid_s": 0.0,
        "avoid_delta_reach_deg": None,
        "avoid_delta_goal_deg": None,
        "reach_delta_goal_deg": None,
        "motion_delta_goal_deg": None,
        "avoid_near_reach": False,
        "motion_opposite": False,
        "note": "",
    }
    if seq is None or len(seq) == 0:
        out["note"] = "no sequence"
        return out
    reach, avoid = seq[0]
    out["stage0"] = _format_stage(reach, avoid)
    out["reach_props"] = _props_from_assignments(reach)
    out["avoid_props"] = _props_from_assignments(avoid)

    task = _task_from_env(env)
    original_obs = task.original_obs
    lidar_dim = int(task.lidar_conf.num_bins)
    origin = np.asarray(task.agent.pos, dtype=float)[:2].copy()
    out["origin"] = origin
    mat_xy = np.asarray(task.agent.mat, dtype=float).reshape(3, 3)[:2, :2]

    reach_keys = _zone_lidar_keys(reach, for_avoid=False)
    avoid_keys = _zone_lidar_keys(avoid, for_avoid=True)
    reach_lidar = _max_lidar(original_obs, reach_keys, lidar_dim)
    avoid_lidar = _max_lidar(original_obs, avoid_keys, lidar_dim)
    reach_dir, reach_s, _ = _lidar_peak_world_dir(reach_lidar, mat_xy)
    avoid_dir, avoid_s, _ = _lidar_peak_world_dir(avoid_lidar, mat_xy)
    out["reach_dir"] = reach_dir
    out["avoid_dir"] = avoid_dir
    out["reach_s"] = reach_s
    out["avoid_s"] = avoid_s

    goal_xy = _goal_xy_for_reach(zone_positions, out["reach_props"])
    out["goal_xy"] = goal_xy
    notes = []
    if goal_xy is not None:
        gvec = goal_xy - origin
        gn = float(np.linalg.norm(gvec))
        if gn > 1e-8:
            goal_dir = gvec / gn
            out["goal_dir"] = goal_dir
            out["reach_delta_goal_deg"] = _bearing_delta_deg(reach_dir, goal_dir)
            out["avoid_delta_goal_deg"] = _bearing_delta_deg(avoid_dir, goal_dir)

    out["avoid_delta_reach_deg"] = _bearing_delta_deg(avoid_dir, reach_dir)
    ad = out["avoid_delta_reach_deg"]
    if ad is not None and ad < 20.0 and avoid_s > 0.05 and reach_s > 0.05:
        out["avoid_near_reach"] = True
        notes.append(f"AVOID≈REACH (Δ={ad:.0f}°) — unexpected on Zone")
    elif ad is not None:
        notes.append(f"avoidΔreach={ad:.0f}° (Zone expect large)")

    rdg = out["reach_delta_goal_deg"]
    if rdg is not None and reach_s > 1e-6:
        notes.append(f"reach≈goal Δ={rdg:.0f}°")
    if avoid_s < 1e-6:
        notes.append("avoid lidar weak/empty")
    out["note"] = "; ".join(notes)
    return out


def _enrich_motion(diag: dict, traj: list[np.ndarray]) -> dict:
    diag["motion_dir"] = None
    diag["motion_delta_goal_deg"] = None
    diag["motion_opposite"] = False
    origin = diag.get("origin")
    goal_dir = diag.get("goal_dir")
    if origin is None or goal_dir is None or len(traj) < 2:
        return diag
    origin = np.asarray(origin, dtype=float)[:2]
    motion = np.zeros(2, dtype=float)
    traveled = 0.0
    for j in range(1, len(traj)):
        d = np.asarray(traj[j], dtype=float)[:2] - np.asarray(traj[j - 1], dtype=float)[:2]
        motion += d
        traveled += float(np.linalg.norm(d))
        if traveled >= 0.6:
            break
    if float(np.linalg.norm(motion)) < 1e-8:
        motion = np.asarray(traj[-1], dtype=float)[:2] - origin
    mn = float(np.linalg.norm(motion))
    if mn < 1e-8:
        return diag
    motion_dir = motion / mn
    diag["motion_dir"] = motion_dir
    md = _bearing_delta_deg(motion_dir, goal_dir)
    diag["motion_delta_goal_deg"] = md
    if md is not None and md > 90.0:
        diag["motion_opposite"] = True
        diag["note"] = (diag.get("note") or "") + f"; agent OPPOSITE goal goΔ={md:.0f}°"
    elif md is not None:
        diag["note"] = (diag.get("note") or "") + f"; goΔ={md:.0f}°"
    return diag


def _print_aggregate(diags: list[dict], outcomes: list[str]) -> None:
    n = len(diags)
    if n == 0:
        return
    near = sum(1 for d in diags if (d.get("avoid_delta_reach_deg") or 999) < 20.0)
    go_opp = sum(1 for d in diags if d.get("motion_opposite"))
    avoid_d = [d["avoid_delta_reach_deg"] for d in diags if d.get("avoid_delta_reach_deg") is not None]
    go_d = [d["motion_delta_goal_deg"] for d in diags if d.get("motion_delta_goal_deg") is not None]
    reach_goal = [d["reach_delta_goal_deg"] for d in diags if d.get("reach_delta_goal_deg") is not None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        "=== Zone native aggregate (confirm avoid ≠ reach) ===\n"
        f"  n={n}  avoidΔreach<20°: {near}/{n} ({100 * near / n:.0f}%)  "
        f"[expect LOW on Zone; HIGH on broken SAR zone_compat]\n"
        f"  goΔ>90° fled goal: {go_opp}/{n} ({100 * go_opp / n:.0f}%)  "
        f"[expect LOW on Zone]\n"
        f"  mean avoidΔreach={mean(avoid_d):.1f}°  mean reachΔgoal={mean(reach_goal):.1f}°  "
        f"mean goΔ={mean(go_d):.1f}°\n"
        f"  outcomes: S={outcomes.count('success')} V={outcomes.count('violation')} "
        f"NF={outcomes.count('not_finish')}\n"
        "====================================================="
    )


def _draw_overlay(ax, diag: dict) -> None:
    origin = diag.get("origin")
    if origin is None:
        return
    o = np.asarray(origin, dtype=float)[:2]
    scale = 1.2

    def arrow(direction, color, ls="-"):
        if direction is None:
            return
        d = np.asarray(direction, dtype=float)[:2]
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            return
        d = d / n
        tip = o + scale * d
        ax.plot([o[0], tip[0]], [o[1], tip[1]], color=color, lw=2.0, linestyle=ls, zorder=12)
        ax.annotate(
            "",
            xy=tip,
            xytext=o + 0.85 * scale * d,
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
            zorder=13,
        )

    if diag.get("goal_xy") is not None:
        g = np.asarray(diag["goal_xy"], dtype=float)[:2]
        ax.plot([o[0], g[0]], [o[1], g[1]], color="#2e7d32", lw=1.0, ls=":", alpha=0.6, zorder=11)
        ax.plot(g[0], g[1], marker="x", color="#2e7d32", markersize=8, zorder=12)
    arrow(diag.get("motion_dir"), "#1565c0")
    arrow(diag.get("reach_dir"), "#c2185b")
    arrow(diag.get("avoid_dir"), "#ef6c00", ls=":")
    arrow(diag.get("goal_dir"), "#2e7d32", ls="--")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zone reach/avoid bearing diag (sidecar; revert by deleting this file)",
    )
    parser.add_argument("--env", type=str, default=DEFAULT_ENV)
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--formula", type=str, default=DEFAULT_FORMULA)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    env_name = args.env
    exp = args.exp
    seed = args.seed
    formula = args.formula

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    if "Safety" in env_name:
        env = make_env_safety(env_name, sampler, render_mode=None, max_steps=None)
    else:
        env = make_env(env_name, sampler, render_mode=None, max_steps=None)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed, None)
    model_store.load_vocab()
    training_status = model_store.load_training_status(map_location="cpu")
    if "Safety" in env_name:
        model = build_model_safety(env, training_status, config)
    else:
        model = build_model(env, training_status, config)

    props = env.get_propositions()
    search = ExhaustiveSearchSafety(env, model, props, num_loops=2)
    agent = Agent(env, model, search=search, propositions=props, verbose=False)

    print(
        "=== Zone diag (sidecar) ===\n"
        "Expect: avoidΔreach mostly LARGE, goΔ>90° rare.\n"
        "If that holds, Zone ckpt is fine; SAR zone_compat avoid≈goal is the transfer break.\n"
        "Revert: delete evaluation/draw_zone_trajectories_diag.py"
    )

    trajectories = []
    zone_poss = []
    titles = []
    diags = []
    outcomes = []
    success = violation = unreachable = 0

    env.reset(seed=seed)
    pbar = trange(args.num_episodes)
    for i in pbar:
        obs, info = env.reset(), {}
        agent.reset()
        done = False
        zone_radius = env.zone_radius
        zpos = env.zone_positions
        zone_poss.append(zpos)
        agent_traj: list[np.ndarray] = []
        diag = {"note": "no steps", "stage0": None}
        first = True

        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=args.deterministic)
                if first:
                    diag = _diagnose_zone_stage0(env, agent.sequence, zpos)
                    first = False
                action = action.flatten()
                obs, _reward, done, info = env.step(action)
                agent_traj.append(np.asarray(env.agent_pos[:2], dtype=float).copy())
            except NoPathsException:
                unreachable += 1
                done = True

        if not agent_traj and hasattr(env, "agent_pos"):
            agent_traj.append(np.asarray(env.agent_pos[:2], dtype=float).copy())

        diag = _enrich_motion(diag, agent_traj)
        trajectories.append(agent_traj)

        if "success" in info:
            outcome = "success"
            success += 1
        elif "violation" in info:
            outcome = "violation"
            violation += 1
        else:
            outcome = "not_finish"
        outcomes.append(outcome)
        diags.append(diag)

        ad = diag.get("avoid_delta_reach_deg")
        gd = diag.get("motion_delta_goal_deg")
        flag = ""
        if diag.get("avoid_near_reach"):
            flag = " AVOID≈REACH"
        elif diag.get("motion_opposite"):
            flag = " GO≠GOAL"
        titles.append(
            f"{outcome} | {diag.get('stage0') or '?'} "
            f"aΔr={ad if ad is not None else float('nan'):.0f}° "
            f"goΔ={gd if gd is not None else float('nan'):.0f}°{flag}"
        )
        print(
            f"[ep {i}] {outcome} | {diag.get('stage0')}\n"
            f"  reach_s={diag.get('reach_s', 0):.3f} avoid_s={diag.get('avoid_s', 0):.3f} | "
            f"{diag.get('note')}"
        )
        pbar.set_postfix({"S": success / (i + 1), "V": violation / (i + 1)})

    print(f"Formula: {formula}, Success: {success}, Violation: {violation}, Unreachable: {unreachable}")
    _print_aggregate(diags, outcomes)

    env.close()
    cols = 4 if len(zone_poss) > 4 else max(1, len(zone_poss))
    rows = 1 if len(zone_poss) <= 4 else int(math.ceil(len(zone_poss) / cols))
    fig = draw_trajectories(zone_poss, zone_radius, trajectories, titles, cols, rows)
    for ax, diag in zip(fig.axes, diags):
        _draw_overlay(ax, diag)
    # Legend on first panel only.
    if fig.axes:
        ax0 = fig.axes[0]
        ax0.plot([], [], color="#1565c0", lw=2.0, label="agent went")
        ax0.plot([], [], color="#c2185b", lw=2.0, label="reach peak")
        ax0.plot([], [], color="#ef6c00", lw=2.0, ls=":", label="avoid peak")
        ax0.plot([], [], color="#2e7d32", lw=2.0, ls="--", label="true reach goal")
        ax0.legend(loc="upper right", fontsize=7, framealpha=0.7)

    out = args.out or f"{env_name}_{exp}_s{seed}_zone_diag_trajectories.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
