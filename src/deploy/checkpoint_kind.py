"""Detect GenZ checkpoint algorithm (PPO+LTL vs RCO safety)."""
from __future__ import annotations

from typing import Any

PPO_SAR_MODEL_CONFIG = "PointLTL0MASAR1-v0"


def detect_rl_algo(state_dict: dict[str, Any]) -> str:
    """Return ``ppo`` or ``rco`` from a ``model_state`` dict."""
    if state_dict is None:
        raise ValueError("state_dict is None")
    if "cost_critic.0.weight" in state_dict:
        return "rco"
    if any(k.startswith("ltl_net.") for k in state_dict):
        return "ppo"
    raise ValueError(
        "Unrecognized checkpoint: expected cost_critic (RCO) or ltl_net (PPO) keys"
    )


def is_rco_state_dict(state_dict: dict[str, Any]) -> bool:
    return detect_rl_algo(state_dict) == "rco"


def ppo_model_config_key(train_env: str) -> str:
    if "MASAR" in train_env:
        return PPO_SAR_MODEL_CONFIG
    return train_env.split(".")[0]
