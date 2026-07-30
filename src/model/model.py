from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig, ModelSafetyConfig, StandardNetConfig
from model.ltl.ltl_net import LTLNet
from model.mixed_distribution import MixedDistribution
from preprocessing.vocab import VOCAB
from model.policy import ContinuousActor
from model.policy import DiscreteActor
from utils import torch_utils


def _linear_layers_from_mlp_prefix(
        state_dict: dict[str, Any],
        prefix: str,
) -> list[tuple[int, int]]:
    """Return (out, in) shapes for each Linear in an MLP Sequential keyed by ``prefix``."""
    layers: list[tuple[int, int]] = []
    i = 0
    while f"{prefix}.{i}.weight" in state_dict:
        w = state_dict[f"{prefix}.{i}.weight"]
        layers.append((int(w.shape[0]), int(w.shape[1])))
        i += 2
    return layers


def infer_model_safety_shapes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct safety-model tensor shapes saved in a training checkpoint."""
    actor_in = int(state_dict["actor.enc.0.weight"].shape[1])
    action_dim = int(state_dict["actor.mu.0.weight"].shape[0])
    actor_hidden = [
        int(state_dict[f"actor.enc.{i}.weight"].shape[0])
        for i in range(0, 100, 2)
        if f"actor.enc.{i}.weight" in state_dict
    ]

    env_net_layers: list[int] | None = None
    feat_dim = actor_in
    if "env_net.mlp.0.weight" in state_dict:
        env_linears = _linear_layers_from_mlp_prefix(state_dict, "env_net.mlp")
        feat_dim = env_linears[0][1]
        env_net_layers = [out for out, _in in env_linears]
        embedding_dim = env_linears[-1][0]
    else:
        embedding_dim = actor_in

    if embedding_dim != actor_in:
        raise ValueError(
            f"Checkpoint env_net output {embedding_dim} != actor input {actor_in}"
        )

    return {
        "feat_dim": feat_dim,
        "embedding_dim": embedding_dim,
        "env_net_layers": env_net_layers,
        "action_dim": action_dim,
        "actor_hidden": actor_hidden,
    }


class Model(nn.Module):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 ltl_net: nn.Module,
                 env_net: Optional[nn.Module],
                 ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.ltl_net = ltl_net
        self.env_net = env_net
        self.recurrent = False

    def compute_embedding(self, obs):
        env_embedding = self.env_net(obs.features) if self.env_net is not None else obs.features
        ltl_embedding = self.ltl_net(obs.seq)
        return torch.cat([env_embedding, ltl_embedding], dim=1)

    def forward(self, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critic(embedding).squeeze(1)
        return dist, value


def build_model(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
) -> Model:
    if len(VOCAB) <= 3:
        raise ValueError('VOCAB not initialized')
    obs_shape = env.observation_space['features'].shape
    action_space = env.action_space
    action_dim = action_space.n if isinstance(action_space, gymnasium.spaces.Discrete) else action_space.shape[0]
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        assert len(obs_shape) == 1
        env_net = None
        env_embedding_dim = obs_shape[0]

    embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB['PAD'])
    ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[env_embedding_dim + ltl_net.embedding_dim, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[env_embedding_dim + ltl_net.embedding_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_net.embedding_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    model = Model(actor, critic, ltl_net, env_net)

    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"])
    return model


class ModelSafety(Model):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 cost_critic: nn.Module,
                 lagrangian_net: nn.Module,
                 env_net: Optional[nn.Module],):
        
        super().__init__(actor, critic, None, env_net)
        self.cost_critic = cost_critic
        self.lagrangian_net = lagrangian_net
        
    def forward(self, obs, collect: bool = True):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critic(embedding).squeeze(1)
        cost_value = self.cost_critic(embedding).squeeze(1)
        if collect:
            return dist, value, cost_value 
        lagrangian = self.lagrangian_net(embedding).squeeze(1)
        return dist, value, cost_value, lagrangian

    def compute_embedding(self, obs):
        env_embedding = self.env_net(obs.features) if self.env_net is not None else obs.features
        return env_embedding

def build_model_safety(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelSafetyConfig,
) -> ModelSafety:
    state_dict = training_status.get("model_state")
    inferred = infer_model_safety_shapes(state_dict) if state_dict else None

    if inferred is not None:
        obs_shape = (inferred["feat_dim"],)
        env_embedding_dim = inferred["embedding_dim"]
        action_dim = inferred["action_dim"]
        actor_hidden = inferred["actor_hidden"] or list(model_config.actor.layers)
    else:
        obs_shape = env.observation_space['features'].shape
        env_embedding_dim = int(obs_shape[0])
        action_dim = (
            env.action_space.n
            if isinstance(env.action_space, gymnasium.spaces.Discrete)
            else int(env.action_space.shape[0])
        )
        actor_hidden = list(model_config.actor.layers)

    env_net = None
    if model_config.env_net is not None:
        if inferred is not None and inferred["env_net_layers"] is not None:
            env_net_cfg = StandardNetConfig(
                layers=inferred["env_net_layers"],
                activation=model_config.env_net.activation,
            )
            env_net = env_net_cfg.build(obs_shape)
        else:
            env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    elif inferred is not None:
        env_embedding_dim = inferred["embedding_dim"]

    if inferred is not None:
        use_discrete = "actor.mu.0.weight" not in state_dict
    else:
        use_discrete = isinstance(env.action_space, gymnasium.spaces.Discrete)

    if use_discrete:
        actor = DiscreteActor(
            action_dim=action_dim,
            layers=[env_embedding_dim, *actor_hidden],
            activation=model_config.actor.activation,
        )
    else:
        actor = ContinuousActor(
            action_dim=action_dim,
            layers=[env_embedding_dim, *actor_hidden],
            activation=model_config.actor.activation,
            state_dependent_std=model_config.actor.state_dependent_std,
        )

    critic = torch_utils.make_mlp_layers(
        [env_embedding_dim, *model_config.critic.layers, 1],
        activation=model_config.critic.activation,
    )
    cost_critic = torch_utils.make_mlp_layers(
        [env_embedding_dim, *model_config.cost_critic.layers, 1],
        activation=model_config.cost_critic.activation,
        final_layer_activation=False,
    )
    lagrangian_net = torch_utils.make_mlp_layers(
        [env_embedding_dim, *model_config.lagrangian.layers, 1],
        activation=model_config.lagrangian.activation,
    )

    model_safety = ModelSafety(actor, critic, cost_critic, lagrangian_net, env_net)

    if state_dict is not None:
        model_safety.load_state_dict(state_dict)
    return model_safety
