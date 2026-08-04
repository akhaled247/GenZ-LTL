"""MA SAR deploy rollout (paper §5.3)."""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from deploy.coordinator import MultiAgentSARCoordinator
from deploy.env_check import assert_sar_wc_paper_protocol
from deploy.feature_recipe import (
    apply_zone_compat_deploy_meta,
    attach_model_deploy_fields,
    ensure_sar_v1_indep_lidars,
    resolve_zone_compat,
)
from deploy.loading import load_model_for_deploy
from deploy.parallel_eval import aggregate_ma_shards, run_sharded_ma_eval
from envs import make_env_safety
from envs.env_utils import get_env_attr
from envs.sar_deploy import (
    check_rabinizer,
    ma_episode_success,
    ma_episode_violation,
    ma_step_saw_walls,
    print_ma_episode_done_debug,
)
from envs.seq_wrapper import sar_task
from ltl import FixedSampler
from sequence.search import ExhaustiveSearchSafety, NoPathsException
from utils.train_device import resolve_training_device
from utils.deploy_meta import (
    MA_EVAL_ENV_DEFAULT,
    MA_EVAL_FORMULA_DEFAULT,
    FEAT_RECIPE_LEGACY_V0,
    warn_if_fragile_ma_formula,
)

TRAIN_ENV = "PointLTL0MASAR1WC-v0"
EVAL_ENV = MA_EVAL_ENV_DEFAULT


def _ma_sar_shard_worker(kwargs: dict) -> tuple[int, int, int, list[int], list[float]]:
    return _simulate_ma_sar_episodes(**kwargs)


def _simulate_ma_sar_episodes(
    eval_env: str,
    train_env: str,
    gamma: float,
    exp: str,
    seed: int,
    formula: str,
    render: bool,
    deterministic: bool,
    debug_done: bool,
    device: str,
    zone_compat: bool,
    episode_indices: Sequence[int],
) -> tuple[int, int, int, list[int], list[float]]:
    check_rabinizer()

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if device != "cpu":
        device = resolve_training_device(device)

    model, deploy_meta, _store = load_model_for_deploy(train_env, exp, seed, formula, device=device)
    deploy_meta = dict(deploy_meta)
    deploy_meta.setdefault("train_env", train_env)
    deploy_meta = ensure_sar_v1_indep_lidars(deploy_meta, lidar_bins=16)
    zone_compat = resolve_zone_compat(train_env, zone_compat)

    if zone_compat:
        deploy_meta = apply_zone_compat_deploy_meta(deploy_meta, lidar_bins=16)
    else:
        if formula == MA_EVAL_FORMULA_DEFAULT:
            formula = deploy_meta.get("ma_eval_formula", formula)
        if eval_env == EVAL_ENV:
            eval_env = deploy_meta.get("eval_env", eval_env)
    attach_model_deploy_fields(model, deploy_meta)

    warn_if_fragile_ma_formula(formula)

    if deploy_meta.get("feat_recipe") == FEAT_RECIPE_LEGACY_V0:
        print(
            "WARNING: legacy_v0 feature recipe — prefer retraining with sar_v1 "
            "for paper-protocol deploy."
        )

    entr_bldg_obs = bool(deploy_meta.get("entr_bldg_obs", False)) and not zone_compat
    sampler = FixedSampler.partial(formula)
    env = make_env_safety(
        eval_env,
        sampler,
        flat=False,
        render_mode="human" if render else None,
        sar_env_backend="specrl",
        max_steps=2500,
        entr_bldg_obs=entr_bldg_obs,
        zone_compat=zone_compat,
    )
    assert_sar_wc_paper_protocol(env)

    num_agents = getattr(sar_task(env), "agent_num", 2)
    agent_keys = [f"agent_{i}" for i in range(num_agents)]

    props = get_env_attr(env, "get_propositions")()
    search = ExhaustiveSearchSafety(env, model, props, num_loops=2, device=device)
    coordinator = MultiAgentSARCoordinator(
        env, model, search, props, num_agents, verbose=render, device=device
    )

    num_successes = 0
    num_violations = 0
    num_unreachable = 0
    steps: list[int] = []
    rets: list[float] = []

    indices = list(episode_indices)
    pbar = range(len(indices))
    if not render:
        pbar = tqdm(pbar, desc=f"eps[{indices[0]}-{indices[-1]}]" if indices else "eps")

    for local_i in pbar:
        i = indices[local_i]
        obs, info = env.reset(seed=seed + i), {}
        if render:
            print(obs["goal"])
        coordinator.reset()
        done = False
        num_steps = 0
        saw_walls = False
        while not done:
            try:
                action = coordinator.get_action(obs, info, deterministic=deterministic)
            except NoPathsException:
                num_unreachable += 1
                rets.append(0.0)
                break
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if ma_step_saw_walls(info, agent_keys):
                saw_walls = True
            if done:
                if render or debug_done:
                    print_ma_episode_done_debug(
                        env,
                        info,
                        step=num_steps,
                        saw_walls=saw_walls,
                        reach=coordinator.last_reach,
                        avoid=coordinator.last_avoid,
                        entr_bldg_obs=entr_bldg_obs,
                        zone_compat=zone_compat,
                    )
                success = ma_episode_success(info, agent_keys)
                violation = (
                    not success
                    and ma_episode_violation(info, agent_keys, saw_walls=saw_walls)
                )
                if success:
                    num_successes += 1
                    steps.append(num_steps)
                elif violation:
                    num_violations += 1
                rets.append(int(success) * gamma ** (num_steps - 1))
                if not render:
                    done_so_far = local_i + 1
                    pbar.set_postfix({
                        's': seed,
                        "S": num_successes / done_so_far,
                        "V": num_violations / done_so_far,
                        "ADR": np.mean(rets),
                        "AS": np.mean(steps) if steps else 0,
                    })

    env.close()
    return num_successes, num_violations, num_unreachable, steps, rets


def simulate_ma_sar(
    eval_env: str,
    train_env: str,
    gamma: float,
    exp: str,
    seed: int,
    num_episodes: int,
    formula: str,
    render: bool,
    deterministic: bool = True,
    debug_done: bool = False,
    device: str = "cpu",
    zone_compat: bool = False,
    num_workers: int = 1,
    episode_indices: Sequence[int] | None = None,
):
    if render and num_workers > 1:
        raise ValueError("render=True requires num_workers=1")

    if episode_indices is not None:
        shard = _simulate_ma_sar_episodes(
            eval_env, train_env, gamma, exp, seed, formula,
            render, deterministic, debug_done, device, zone_compat, episode_indices,
        )
        return aggregate_ma_shards(
            [shard], seed=seed, num_episodes=len(episode_indices),
        )

    workers = max(1, int(num_workers))
    if workers == 1:
        shard = _simulate_ma_sar_episodes(
            eval_env, train_env, gamma, exp, seed, formula,
            render, deterministic, debug_done, device, zone_compat, range(num_episodes),
        )
        return aggregate_ma_shards([shard], seed=seed, num_episodes=num_episodes)

    base = {
        "eval_env": eval_env,
        "train_env": train_env,
        "gamma": gamma,
        "exp": exp,
        "seed": seed,
        "formula": formula,
        "render": False,
        "deterministic": deterministic,
        "debug_done": debug_done,
        "device": device,
        "zone_compat": zone_compat,
    }
    return run_sharded_ma_eval(
        _ma_sar_shard_worker,
        base,
        num_episodes=num_episodes,
        num_workers=workers,
        env_name=eval_env,
    )
