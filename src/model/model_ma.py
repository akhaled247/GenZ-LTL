from __future__ import annotations

from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig, ModelSafetyConfig
from model.ltl.ltl_net import LTLNet
from model.model import Model, ModelSafety
from model.policy import ContinuousActor, DiscreteActor
from preprocessing.vocab import VOCAB
from utils import torch_utils


class ModelMA(Model):
    """Shared actor + per-agent reward critics."""

    def __init__(
        self,
        actor: nn.Module,
        critics: nn.ModuleList,
        ltl_net: nn.Module,
        env_net: Optional[nn.Module],
    ):
        super().__init__(actor, critics[0], ltl_net, env_net)
        self.critics = critics
        self.num_agents = len(critics)

    def forward_agent(self, agent_id: int, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critics[agent_id](embedding).squeeze(1)
        return dist, value

    def forward(self, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        values = torch.empty(embedding.shape[0], device=embedding.device, dtype=embedding.dtype)
        for aid in range(self.num_agents):
            mask = obs.agent_id == aid
            if mask.any():
                values[mask] = self.critics[aid](embedding[mask]).squeeze(-1)
        return dist, values


class ModelSafetyMA(ModelSafety):
    """Shared actor + lagrangian + per-agent reward/cost critics."""

    def __init__(
        self,
        actor: nn.Module,
        critics: nn.ModuleList,
        cost_critics: nn.ModuleList,
        lagrangian_net: nn.Module,
        env_net: Optional[nn.Module],
    ):
        super().__init__(actor, critics[0], cost_critics[0], lagrangian_net, env_net)
        self.critics = critics
        self.cost_critics = cost_critics
        self.num_agents = len(critics)

    def forward_agent(self, agent_id: int, obs, collect: bool = True):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critics[agent_id](embedding).squeeze(1)
        cost_value = self.cost_critics[agent_id](embedding).squeeze(1)
        if collect:
            return dist, value, cost_value
        lagrangian = self.lagrangian_net(embedding).squeeze(1)
        return dist, value, cost_value, lagrangian

    def forward(self, obs, collect: bool = True):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        values = torch.empty(embedding.shape[0], device=embedding.device, dtype=embedding.dtype)
        cost_values = torch.empty(embedding.shape[0], device=embedding.device, dtype=embedding.dtype)
        for aid in range(self.num_agents):
            mask = obs.agent_id == aid
            if mask.any():
                values[mask] = self.critics[aid](embedding[mask]).squeeze(-1)
                cost_values[mask] = self.cost_critics[aid](embedding[mask]).squeeze(-1)
        if collect:
            return dist, values, cost_values
        lagrangian = self.lagrangian_net(embedding).squeeze(1)
        return dist, values, cost_values, lagrangian


def _action_dim(env: gymnasium.Env) -> int:
    space = env.action_space
    if callable(space):
        agent = env.unwrapped.possible_agents[0]
        space = space(agent)
    elif hasattr(space, "spaces"):
        first = next(iter(space.spaces.values()))
        space = first
    if hasattr(space, "shape") and space.shape is not None:
        return int(space.shape[0])
    return int(space.n)


def _num_agents(env: gymnasium.Env) -> int:
    if hasattr(env.unwrapped, "possible_agents"):
        return len(env.unwrapped.possible_agents)
    if hasattr(env, "num_agents"):
        return int(env.num_agents)
    return len(env.observation_space.spaces)


def build_model_ma(
    env: gymnasium.Env,
    training_status: dict[str, Any],
    model_config: ModelConfig,
    num_agents: int | None = None,
) -> ModelMA:
    if len(VOCAB) <= 3:
        raise ValueError("VOCAB not initialized")
    if num_agents is None:
        num_agents = _num_agents(env)
    probe = next(iter(env.observation_space.spaces.keys()))
    obs_shape = env.observation_space[probe]["features"].shape
    action_dim = _action_dim(env)
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        env_net = None
        env_embedding_dim = obs_shape[0]

    embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB["PAD"])
    ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)
    embed_dim = env_embedding_dim + ltl_net.embedding_dim

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(
            action_dim=action_dim,
            layers=[embed_dim, *model_config.actor.layers],
            activation=model_config.actor.activation,
        )
    else:
        actor = ContinuousActor(
            action_dim=action_dim,
            layers=[embed_dim, *model_config.actor.layers],
            activation=model_config.actor.activation,
            state_dependent_std=model_config.actor.state_dependent_std,
        )

    critics = nn.ModuleList([
        torch_utils.make_mlp_layers(
            [embed_dim, *model_config.critic.layers, 1],
            activation=model_config.critic.activation,
            final_layer_activation=False,
        )
        for _ in range(num_agents)
    ])
    model = ModelMA(actor, critics, ltl_net, env_net)
    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"], strict=False)
    return model


def build_model_safety_ma(
    env: gymnasium.Env,
    training_status: dict[str, Any],
    model_config: ModelSafetyConfig,
    num_agents: int | None = None,
) -> ModelSafetyMA:
    if num_agents is None:
        num_agents = _num_agents(env)
    probe = next(iter(env.observation_space.spaces.keys()))
    obs_shape = env.observation_space[probe]["features"].shape
    action_dim = _action_dim(env)
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        env_net = None
        env_embedding_dim = obs_shape[0]

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(
            action_dim=action_dim,
            layers=[env_embedding_dim, *model_config.actor.layers],
            activation=model_config.actor.activation,
        )
    else:
        actor = ContinuousActor(
            action_dim=action_dim,
            layers=[env_embedding_dim, *model_config.actor.layers],
            activation=model_config.actor.activation,
            state_dependent_std=model_config.actor.state_dependent_std,
        )

    critics = nn.ModuleList([
        torch_utils.make_mlp_layers(
            [env_embedding_dim, *model_config.critic.layers, 1],
            activation=model_config.critic.activation,
            final_layer_activation=False,
        )
        for _ in range(num_agents)
    ])
    cost_critics = nn.ModuleList([
        torch_utils.make_mlp_layers(
            [env_embedding_dim, *model_config.cost_critic.layers, 1],
            activation=model_config.cost_critic.activation,
            final_layer_activation=False,
        )
        for _ in range(num_agents)
    ])
    lagrangian_net = torch_utils.make_mlp_layers(
        [env_embedding_dim, *model_config.lagrangian.layers, 1],
        activation=model_config.lagrangian.activation,
    )
    model = ModelSafetyMA(actor, critics, cost_critics, lagrangian_net, env_net)
    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"], strict=False)
    return model
