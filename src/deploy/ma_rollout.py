"""MA SAR deploy rollout (paper §5.3)."""
from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import tqdm

from deploy.coordinator import MultiAgentSARCoordinator
from deploy.env_check import assert_sar_wc_paper_protocol
from deploy.loading import load_model_for_deploy
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
):
    check_rabinizer()

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if device != "cpu":
        device = resolve_training_device(device)

    model, deploy_meta, _store = load_model_for_deploy(train_env, exp, seed, formula, device=device)

    if formula == MA_EVAL_FORMULA_DEFAULT:
        formula = deploy_meta.get("ma_eval_formula", formula)
    if eval_env == EVAL_ENV:
        eval_env = deploy_meta.get("eval_env", eval_env)

    warn_if_fragile_ma_formula(formula)

    if deploy_meta.get("feat_recipe") == FEAT_RECIPE_LEGACY_V0:
        print(
            "WARNING: legacy_v0 feature recipe — prefer retraining with sar_v1 (48-dim) "
            "for paper-protocol deploy."
        )

    sampler = FixedSampler.partial(formula)
    env = make_env_safety(
        eval_env,
        sampler,
        flat=False,
        render_mode="human" if render else None,
        sar_env_backend="specrl",
        max_steps=2500,
        entr_bldg_obs=bool(deploy_meta.get("entr_bldg_obs", False)),
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

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)

    for i in pbar:
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
                        env, info, step=num_steps, saw_walls=saw_walls,
                    )
                success = ma_episode_success(info, agent_keys)
                violation = ma_episode_violation(info, agent_keys)
                if success:
                    num_successes += 1
                    steps.append(num_steps)
                elif violation:
                    num_violations += 1
                rets.append(int(success) * gamma ** (num_steps - 1))
                if not render:
                    pbar.set_postfix({
                        "S": num_successes / (i + 1),
                        "V": num_violations / (i + 1),
                        "ADR": np.mean(rets),
                        "AS": np.mean(steps) if steps else 0,
                    })

    env.close()
    average_steps = np.mean(steps) if steps else float("nan")
    adr = np.mean(rets) if rets else 0.0
    print(
        f"{seed}: {num_successes / num_episodes:.3f},"
        f"{num_violations / num_episodes:.3f},"
        f"{num_unreachable / num_episodes:.3f},"
        f"{adr:.3f},{average_steps:.3f}"
    )
    return num_successes, num_violations, average_steps
