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
from envs.seq_wrapper import sar_task  # noqa: E402
from ltl import FixedSampler  # noqa: E402
from model.agent import Agent  # noqa: E402
from sequence.search import ExhaustiveSearchSafety, NoPathsException  # noqa: E402
from utils.train_device import resolve_training_device  # noqa: E402
from visualize.sar import draw_sar_trajectories, snapshot_sar_scene  # noqa: E402


DEFAULT_FORMULA = "(!surface_0 U entrapped_0) & F surface_0"


def _agent_num_from_env_id(env_id: str) -> int | None:
    m = re.search(r"MASAR(\d+)", env_id)
    if not m:
        return None
    return int(m.group(1))


def _episode_title(formula: str, info: dict) -> str:
    if "success" in info:
        return f"{formula}_success"
    if "violation" in info:
        return f"{formula}_violation"
    return f"{formula}_not_finish"


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
    agent = Agent(env, model, search=search, propositions=props, verbose=False)

    scenes, paths_list, titles = [], [], []
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
        done = False
        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=deterministic)
                action = action.flatten()
                if action.shape == (1,):
                    action = action[0]
                obs, _reward, done, info = env.step(action)
                _append_xy(paths, _collect_agent_xy(task, num_agents))
            except NoPathsException:
                unreachable += 1
                done = True
        paths_list.append(paths)
        titles.append(_episode_title(formula, info))
        if "success" in info:
            success += 1
        elif "violation" in info:
            violation += 1
        pbar.set_postfix({"S": success / (i + 1), "V": violation / (i + 1)})

    env.close()
    print(f"Formula: {formula}, Success: {success}, Violation: {violation}, Unreachable: {unreachable}")
    return scenes, paths_list, titles


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

    entr_bldg_obs = bool(deploy_meta.get("entr_bldg_obs", False)) and not zone_compat
    sampler = FixedSampler.partial(formula)
    env = make_env_safety(
        eval_env,
        sampler,
        flat=False,
        sar_env_backend="specrl",
        max_steps=2500,
        entr_bldg_obs=entr_bldg_obs,
        zone_compat=zone_compat,
    )
    num_agents = int(getattr(sar_task(env), "agent_num", 2) or 2)
    props = get_env_attr(env, "get_propositions")()
    search = ExhaustiveSearchSafety(env, model, props, num_loops=2, device=device)
    coordinator = MultiAgentSARCoordinator(
        env, model, search, props, num_agents, verbose=False, device=device,
    )

    scenes, paths_list, titles = [], [], []
    success = violation = unreachable = 0
    pbar = trange(num_episodes)
    for i in pbar:
        obs, info = env.reset(seed=seed + i), {}
        coordinator.reset()
        task = sar_task(env)
        scenes.append(snapshot_sar_scene(task))
        paths: dict[str, list[np.ndarray]] = {}
        _append_xy(paths, _collect_agent_xy(task, num_agents))
        done = False
        while not done:
            try:
                action = coordinator.get_action(obs, info, deterministic=deterministic)
                obs, _reward, done, info = env.step(action)
                _append_xy(paths, _collect_agent_xy(task, num_agents))
            except NoPathsException:
                unreachable += 1
                done = True
        paths_list.append(paths)
        titles.append(_episode_title(formula, info))
        if "success" in info:
            success += 1
        elif "violation" in info:
            violation += 1
        pbar.set_postfix({"S": success / (i + 1), "V": violation / (i + 1)})

    env.close()
    print(f"Formula: {formula}, Success: {success}, Violation: {violation}, Unreachable: {unreachable}")
    return scenes, paths_list, titles


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

    rollout = _rollout_ma if use_ma else _rollout_sa
    scenes, paths_list, titles = rollout(
        train_env=train_env,
        eval_env=eval_env,
        exp=args.exp,
        seed=seed,
        formula=formula,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        zone_compat=args.zone_compat,
        device=args.device,
    )

    cols, rows = _grid_shape(len(scenes))
    fig = draw_sar_trajectories(scenes, paths_list, titles, cols, rows)
    out = args.out or f"{eval_env}_{args.exp}_s{seed}_trajectories.png"
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
