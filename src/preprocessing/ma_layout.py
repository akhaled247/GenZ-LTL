"""Lightweight MA obs/action layout helpers (no torch_ac / model imports)."""

from typing import Any

import torch


def flatten_ma_obs(
        obs_list: list[dict[str, dict[str, Any]]],
        agent_keys: list[str],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Expand env-major dict obs into flat per-agent obs list + agent indices."""
    flat_obs = []
    agent_ids = []
    for env_obs in obs_list:
        for i, key in enumerate(agent_keys):
            flat_obs.append(env_obs[key])
            agent_ids.append(i)
    return flat_obs, agent_ids


def unflatten_ma_actions(
        actions: torch.Tensor,
        num_envs: int,
        num_agents: int,
        agent_keys: list[str],
) -> list[dict[str, Any]]:
    actions_np = actions.detach().cpu().numpy()
    out = []
    idx = 0
    for _ in range(num_envs):
        action_dict = {}
        for key in agent_keys:
            action_dict[key] = actions_np[idx]
            idx += 1
        out.append(action_dict)
    return out
