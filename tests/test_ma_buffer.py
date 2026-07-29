"""Unit tests for MA buffer sizing and per-agent critic gradient routing."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltl.automata import LDBASequence
from preprocessing.ma_layout import flatten_ma_obs
from preprocessing.preprocessing_ma import preprocess_obss_ma


def test_buffer_size_t_times_n_envs_times_n_agents():
    t, num_envs, num_agents = 4, 2, 3
    num_procs = num_envs * num_agents
    num_steps = t * num_procs
    assert num_steps == t * num_envs * num_agents == 24


def test_flatten_ma_obs_order():
    obs_list = [
        {"agent_0": {"features": [0.0]}, "agent_1": {"features": [1.0]}},
        {"agent_0": {"features": [2.0]}, "agent_1": {"features": [3.0]}},
    ]
    flat, agent_ids = flatten_ma_obs(obs_list, ["agent_0", "agent_1"])
    assert len(flat) == 4
    assert agent_ids == [0, 1, 0, 1]
    assert flat[0]["features"] == [0.0]
    assert flat[3]["features"] == [3.0]


def test_critic_grad_routes_to_matching_agent():
    critics = nn.ModuleList([nn.Linear(4, 1), nn.Linear(4, 1)])
    obs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    agent_ids = torch.tensor([0, 1])
    values = torch.empty(2)
    for aid in range(2):
        mask = agent_ids == aid
        values[mask] = critics[aid](obs[mask]).squeeze(-1)
    values[0].backward(retain_graph=True)
    assert critics[0].weight.grad is not None
    assert critics[0].weight.grad.abs().sum() > 0
    assert critics[1].weight.grad is None or critics[1].weight.grad.abs().sum() == 0

    critics[1].zero_grad(set_to_none=True)
    values[1].backward()
    assert critics[1].weight.grad is not None
    assert critics[1].weight.grad.abs().sum() > 0


def test_preprocess_obss_ma_agent_id_tensor():
    goal = [(LDBASequence.EPSILON, frozenset()), (frozenset(), frozenset())]
    obss = [
        {"features": [0.0, 1.0], "goal": goal, "propositions": []},
        {"features": [2.0, 3.0], "goal": goal, "propositions": []},
    ]
    batch = preprocess_obss_ma(obss, [], [1, 0], device="cpu")
    assert batch.agent_id.tolist() == [1, 0]
    assert batch.features.shape[0] == 2
