"""Multi-agent sequence wrappers for SAR (team subgoals, per-agent observations)."""

from __future__ import annotations

from typing import Any, Callable, SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces

from ltl.automata import LDBASequence
from ltl.logic import Assignment
from envs.seq_wrapper import (
    pre_process_obs_sar,
    sar_agent_obs_keys,
    sar_feat_dim,
    sar_task,
)


def _agent_ids(env: gymnasium.Env) -> list[str]:
    agents = getattr(env.unwrapped, "possible_agents", None)
    if agents:
        return list(agents)
    n = getattr(env.unwrapped, "num_agents", 1)
    return [f"agent_{i}" for i in range(n)]


class SequenceWrapperMA(gymnasium.Wrapper):
    """Team reach-avoid subgoals with per-agent feature observations."""

    def __init__(
        self,
        env: gymnasium.Env,
        sample_sequence: Callable[..., LDBASequence],
        partial_reward: bool = False,
    ):
        super().__init__(env)
        self.agent_ids = _agent_ids(env)
        self.num_agents = len(self.agent_ids)
        lidar_bins = sar_task(env).lidar_conf.num_bins
        feat_dim = sar_feat_dim(lidar_bins)
        feat_space = spaces.Box(-np.inf, np.inf, (feat_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            agent: spaces.Dict({"features": feat_space})
            for agent in self.agent_ids
        })
        self.action_space = env.action_space
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.feat_shape = (feat_dim,)
        self.obs = None
        self.info = None

    def _process_subgoal(
        self,
        obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info["propositions"]
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.0
            info["violation"] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info["success"] = True
            reward = 1.0 if terminated else (1.0 if self.partial_reward else 0.0)
        return obs, reward, terminated, truncated, info

    def _build_agent_obs(self, info: dict) -> dict:
        reach, avoid = self.goal_seq[self.num_reached] if self.num_reached < len(self.goal_seq) else self.goal_seq[-1]
        out = {}
        for i, agent in enumerate(self.agent_ids):
            keys = sar_agent_obs_keys(i)
            features = pre_process_obs_sar(
                self.env, keys, reach, avoid, self.feat_shape, agent_idx=i,
            )
            out[agent] = {
                "features": features,
                "goal": self.goal_seq[self.num_reached:],
                "initial_goal": self.goal_seq,
                "propositions": info["propositions"],
            }
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(reward, dict):
            reward = float(np.mean(list(reward.values())))
        obs, reward, terminated, truncated, info = self._process_subgoal(
            obs, reward, terminated, truncated, info,
        )
        if isinstance(terminated, dict):
            terminated = any(terminated.values())
        if isinstance(truncated, dict):
            truncated = any(truncated.values())
        agent_obs = self._build_agent_obs(info)
        self.obs = agent_obs
        self.info = info
        return agent_obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        agent_obs = self._build_agent_obs(info)
        self.obs = agent_obs
        self.info = info
        return agent_obs, info

    def get_propositions(self):
        return sorted(self.propositions)

    def get_possible_assignments(self):
        return self.env.get_possible_assignments()


class SequenceSafetyWrapperMA(SequenceWrapperMA):
    """MA sequence wrapper for RCO: team (reward, cost) per step."""

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info["propositions"]
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()

        reward = 0.0
        cost = -1.0
        terminated_out = False
        if assignment in avoid:
            cost = 1.0
            info["violation"] = True
            terminated_out = True
        elif assignment in reach:
            reward = 1.0
            info["success"] = True
            self.goal_seq = self.sample_sequence(assignment)
            self.num_reached = 0
            reach, avoid = self.goal_seq[self.num_reached]
        elif info.get("cost", 0) > 0:
            cost = 1.0
            terminated_out = True

        if isinstance(truncated, dict):
            truncated = any(truncated.values())

        agent_obs = self._build_agent_obs(info)
        self.obs = agent_obs
        self.info = info
        return agent_obs, (reward, cost), terminated_out, truncated, info
