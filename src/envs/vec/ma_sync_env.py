"""Vectorized multi-agent env runner (list of MA sequence envs)."""

from __future__ import annotations

from typing import Any


class MASyncEnv:
    """Step N multi-agent envs; each env uses dict actions and dict observations."""

    def __init__(self, envs: list):
        assert envs
        self.envs = envs
        self.num_envs = len(envs)
        self.num_procs = len(envs)
        if hasattr(envs[0], "agent_ids"):
            self.agent_ids = list(envs[0].agent_ids)
        elif hasattr(envs[0].unwrapped, "possible_agents"):
            self.agent_ids = list(envs[0].unwrapped.possible_agents)
        else:
            n = len(envs[0].observation_space.spaces)
            self.agent_ids = [f"agent_{i}" for i in range(n)]
        self.num_agents = len(self.agent_ids)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def get_propositions(self) -> list[str]:
        return self.envs[0].get_propositions()

    @staticmethod
    def _reset_obs(env) -> dict[str, Any]:
        out = env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def reset(self) -> list[dict[str, Any]]:
        return [self._reset_obs(env) for env in self.envs]

    def step(self, actions: list[dict[str, Any]]):
        results = [env.step(act) for env, act in zip(self.envs, actions)]
        obs_list = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] if len(r) == 4 else r[4] for r in results]
        return obs_list, rewards, dones, infos


# Alias used by torch_ac MA algos
MASyncVecEnv = MASyncEnv
