"""Evaluate a MASAR1WC-trained shared RCO policy on MASAR2WC with automaton coordinator."""
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env_safety
from envs.env_utils import get_env_attr
from envs.sar_deploy import (
    check_rabinizer,
    ma_episode_success,
    ma_episode_violation,
    sar_preprocess_for_deploy,
)
from envs.seq_wrapper import sar_task
from ltl import FixedSampler
from model.model import build_model_safety
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearchSafety, NoPathsException
from utils.deploy_meta import (
    MA_EVAL_ENV_DEFAULT,
    MA_EVAL_FORMULA_DEFAULT,
    build_deploy_meta,
    load_deploy_meta,
    save_deploy_meta,
)
from utils.model_store import ModelStore
import argparse

TRAIN_ENV = 'PointLTL0MASAR1WC-v0'
EVAL_ENV = MA_EVAL_ENV_DEFAULT


class MultiAgentSARAgent:
    """Shared-policy coordinator: one Büchi search, per-agent SAR feature reduction."""

    def __init__(self, env, model, search, propositions, num_agents: int, verbose=False):
        self.env = env
        self.model = model
        self.search = search
        self.propositions = propositions
        self.num_agents = num_agents
        self.verbose = verbose
        self.sequence = None
        self.current_goal_steps = 0
        self.timeout = 300
        self._forward_agent = Agent(env, model, search, propositions, verbose=verbose)

    def reset(self):
        self.sequence = None
        self.current_goal_steps = 0
        self._forward_agent.reset()

    def get_action(self, obs, info, deterministic=False) -> dict[str, np.ndarray]:
        if 'ldba_state_changed' in info or self.sequence is None:
            prev_seq = self.sequence
            self.sequence = self.search(obs['ldba'], obs['ldba_states'], obs)
            if self.sequence != prev_seq:
                self.current_goal_steps = 0
            if self.verbose:
                print(f'Selected sequence: {self.sequence}')
        else:
            self.current_goal_steps += 1
            if self.current_goal_steps >= self.timeout:
                unfeasible_states = [
                    s for s, accepting in zip(obs['ldba_states'], obs['ldba_states_accepting'])
                    if not accepting
                ]
                if unfeasible_states:
                    true_props = set()
                    for a in self.sequence[0][0]:
                        true_props = true_props.union(a.get_true_propositions())
                    from ltl.logic import Assignment
                    reach_assignment = Assignment.where(
                        *true_props, propositions=obs['ldba'].propositions,
                    ).to_frozen()
                    obs['ldba'].mark_unfeasible(unfeasible_states, reach_assignment)
                    prev_seq = self.sequence
                    self.sequence = self.search(obs['ldba'], obs['ldba_states'], obs)
                    assert self.sequence != prev_seq
                self.current_goal_steps = 0

        assert self.sequence is not None
        reach, avoid = self.sequence[0]
        actions = {}
        for agent_idx in range(self.num_agents):
            obs_i = copy.deepcopy(obs)
            obs_i['goal'] = self.sequence
            obs_i['features'] = sar_preprocess_for_deploy(
                self.env, self.model, reach, avoid, agent_idx=agent_idx,
            )
            self._forward_agent.agent_idx = agent_idx
            action = self._forward_agent.forward(obs_i, deterministic).flatten()
            actions[f'agent_{agent_idx}'] = action
        return actions


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
):
    check_rabinizer()

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model_store = ModelStore(train_env, exp, seed, None)
    training_status = model_store.load_training_status(map_location='cpu')
    deploy_meta = load_deploy_meta(model_store.path)
    if deploy_meta is None:
        from model.model import infer_model_safety_shapes
        inferred = infer_model_safety_shapes(training_status["model_state"])
        deploy_meta = build_deploy_meta(
            train_env=train_env,
            raw_feature_dim=int(inferred["feature_dim"]),
            use_env_net=bool(inferred["use_env_net"]),
            actor_input_dim=int(inferred["embedding_dim"]),
        )
        save_deploy_meta(model_store.path, deploy_meta)
    if deploy_meta is not None and formula == MA_EVAL_FORMULA_DEFAULT:
        formula = deploy_meta.get("ma_eval_formula", formula)
    if deploy_meta is not None and eval_env == EVAL_ENV:
        eval_env = deploy_meta.get("eval_env", eval_env)

    sampler = FixedSampler.partial(formula)
    env = make_env_safety(
        eval_env, sampler, flat=False,
        render_mode='human' if render else None,
    )
    num_agents = getattr(sar_task(env), 'agent_num', 2)
    agent_keys = [f'agent_{i}' for i in range(num_agents)]

    config = model_configs[train_env]
    training_status = model_store.load_training_status(map_location='cpu')
    probe_env = make_env_safety(train_env, sampler, flat=True, sequence=False)
    try:
        model = build_model_safety(
            probe_env, training_status, config, deploy_meta=deploy_meta,
        )
    finally:
        probe_env.close()
    props = get_env_attr(env, 'get_propositions')()
    search = ExhaustiveSearchSafety(env, model, props, num_loops=2)
    agent = MultiAgentSARAgent(env, model, search, props, num_agents, verbose=render)

    num_successes = 0
    num_violations = 0
    num_unreachable = 0
    steps = []
    rets = []

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)

    for i in pbar:
        obs, info = env.reset(seed=seed + i), {}
        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=deterministic)
            except NoPathsException:
                num_unreachable += 1
                rets.append(0)
                break
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                success = ma_episode_success(info, agent_keys)
                violation = ma_episode_violation(info, agent_keys)
                final_reward = int(success)
                if success:
                    num_successes += 1
                    steps.append(num_steps)
                elif violation:
                    num_violations += 1
                rets.append(final_reward * gamma ** (num_steps - 1))
                if not render:
                    pbar.set_postfix({
                        'S': num_successes / (i + 1),
                        'V': num_violations / (i + 1),
                        'ADR': np.mean(rets),
                        'AS': np.mean(steps) if steps else 0,
                    })

    env.close()
    average_steps = np.mean(steps) if steps else float('nan')
    adr = np.mean(rets) if rets else 0.0
    print(
        f'{seed}: {num_successes / num_episodes:.3f},'
        f'{num_violations / num_episodes:.3f},'
        f'{num_unreachable / num_episodes:.3f},'
        f'{adr:.3f},{average_steps:.3f}'
    )
    return num_successes, num_violations, average_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_env', type=str, default=EVAL_ENV)
    parser.add_argument('--train_env', type=str, default=TRAIN_ENV)
    parser.add_argument('--exp', type=str, default='GenZ-LTL')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--formula', type=str, default=MA_EVAL_FORMULA_DEFAULT)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    gamma = 0.998
    simulate_ma_sar(
        args.eval_env, args.train_env, gamma, args.exp, args.seed,
        args.num_episodes, args.formula, args.render, args.deterministic,
    )


if __name__ == '__main__':
    main()
