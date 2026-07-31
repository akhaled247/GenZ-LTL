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


def infer_model_safety_shapes(
        state_dict: dict[str, Any],
        *,
        num_propositions: int | None = None,
        use_subgoal_one_hot: bool = False,
) -> dict[str, Any]:
    """Reconstruct safety-model tensor shapes saved in a training checkpoint."""
    actor_in = int(state_dict["actor.enc.0.weight"].shape[1])
    action_dim = int(state_dict["actor.mu.0.weight"].shape[0])
    actor_hidden = [
        int(state_dict[f"actor.enc.{i}.weight"].shape[0])
        for i in range(0, 100, 2)
        if f"actor.enc.{i}.weight" in state_dict
    ]

    subgoal_dim = 2 * num_propositions if use_subgoal_one_hot and num_propositions else 0

    env_net_layers: list[int] | None = None
    feat_dim = actor_in
    use_env_net = False
    if "env_net.mlp.0.weight" in state_dict:
        env_linears = _linear_layers_from_mlp_prefix(state_dict, "env_net.mlp")
        feat_dim = env_linears[0][1]
        env_net_layers = [out for out, _in in env_linears]
        env_net_out = env_linears[-1][0]
        use_env_net = env_net_out + subgoal_dim == actor_in

    embedding_dim = actor_in
    if use_env_net:
        feature_dim = feat_dim
    else:
        feature_dim = actor_in - subgoal_dim

    return {
        "feat_dim": feat_dim if use_env_net else feature_dim,
        "feature_dim": feature_dim,
        "embedding_dim": embedding_dim,
        "env_net_layers": env_net_layers if use_env_net else None,
        "use_env_net": use_env_net,
        "action_dim": action_dim,
        "actor_hidden": actor_hidden,
        "subgoal_dim": subgoal_dim,
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
                 env_net: Optional[nn.Module],
                 use_subgoal_one_hot: bool = False,):
        
        super().__init__(actor, critic, None, env_net)
        self.cost_critic = cost_critic
        self.lagrangian_net = lagrangian_net
        self.use_subgoal_one_hot = use_subgoal_one_hot
        
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
        if self.use_subgoal_one_hot:
            env_embedding = torch.cat([env_embedding, obs.current_subgoal], dim=1)
        return env_embedding

def build_model_safety(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelSafetyConfig,
        deploy_meta: dict[str, Any] | None = None,
        use_subgoal_one_hot: bool = False,
        num_propositions: int | None = None,
) -> ModelSafety:
    state_dict = training_status.get("model_state")
    if deploy_meta is not None:
        use_subgoal_one_hot = bool(deploy_meta.get("use_subgoal_one_hot", use_subgoal_one_hot))
        if deploy_meta.get("num_propositions") is not None:
            num_propositions = int(deploy_meta["num_propositions"])

    if num_propositions is None and hasattr(env, "get_propositions"):
        num_propositions = len(env.get_propositions())

    subgoal_dim = 2 * num_propositions if use_subgoal_one_hot and num_propositions else 0

    inferred = (
        infer_model_safety_shapes(
            state_dict,
            num_propositions=num_propositions,
            use_subgoal_one_hot=use_subgoal_one_hot,
        )
        if state_dict
        else None
    )

    if deploy_meta is not None:
        raw_feature_dim = int(deploy_meta["raw_feature_dim"])
        use_env_net_meta = bool(deploy_meta["use_env_net"])
        actor_input_dim = int(deploy_meta["actor_input_dim"])
        if inferred is not None:
            inferred = dict(inferred)
            inferred["feature_dim"] = raw_feature_dim
            inferred["use_env_net"] = use_env_net_meta
            inferred["embedding_dim"] = actor_input_dim
    elif inferred is not None:
        raw_feature_dim = int(inferred["feature_dim"])
        use_env_net_meta = inferred["use_env_net"]
        actor_input_dim = int(inferred["embedding_dim"])
    else:
        raw_feature_dim = int(env.observation_space['features'].shape[0])
        use_env_net_meta = model_config.env_net is not None
        actor_input_dim = (
            (model_config.env_net.build((raw_feature_dim,)).embedding_size + subgoal_dim)
            if model_config.env_net is not None and use_env_net_meta
            else raw_feature_dim + subgoal_dim
        )

    if inferred is not None:
        obs_shape = (inferred["feat_dim"],) if inferred.get("use_env_net") else (raw_feature_dim,)
        env_embedding_dim = inferred["embedding_dim"] - subgoal_dim
        action_dim = inferred["action_dim"]
        actor_hidden = inferred["actor_hidden"] or list(model_config.actor.layers)
    else:
        obs_shape = (raw_feature_dim,)
        if model_config.env_net is not None and use_env_net_meta:
            env_embedding_dim = model_config.env_net.build(obs_shape).embedding_size
        else:
            env_embedding_dim = int(obs_shape[0])
        action_dim = (
            env.action_space.n
            if isinstance(env.action_space, gymnasium.spaces.Discrete)
            else int(env.action_space.shape[0])
        )
        actor_hidden = list(model_config.actor.layers)

    actor_input_dim = env_embedding_dim + subgoal_dim

    env_net = None
    use_env_net = inferred.get("use_env_net", True) if inferred is not None else use_env_net_meta
    if model_config.env_net is not None and use_env_net:
        if inferred is not None and inferred["env_net_layers"] is not None:
            env_net_cfg = StandardNetConfig(
                layers=inferred["env_net_layers"],
                activation=model_config.env_net.activation,
            )
            env_net = env_net_cfg.build(obs_shape)
        else:
            env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
        actor_input_dim = env_embedding_dim + subgoal_dim
    elif inferred is not None:
        env_embedding_dim = inferred["embedding_dim"] - subgoal_dim

    if inferred is not None:
        use_discrete = "actor.mu.0.weight" not in state_dict
    else:
        use_discrete = isinstance(env.action_space, gymnasium.spaces.Discrete)

    if use_discrete:
        actor = DiscreteActor(
            action_dim=action_dim,
            layers=[actor_input_dim, *actor_hidden],
            activation=model_config.actor.activation,
        )
    else:
        actor = ContinuousActor(
            action_dim=action_dim,
            layers=[actor_input_dim, *actor_hidden],
            activation=model_config.actor.activation,
            state_dependent_std=model_config.actor.state_dependent_std,
        )

    critic = torch_utils.make_mlp_layers(
        [actor_input_dim, *model_config.critic.layers, 1],
        activation=model_config.critic.activation,
    )
    cost_critic = torch_utils.make_mlp_layers(
        [actor_input_dim, *model_config.cost_critic.layers, 1],
        activation=model_config.cost_critic.activation,
        final_layer_activation=False,
    )
    lagrangian_net = torch_utils.make_mlp_layers(
        [actor_input_dim, *model_config.lagrangian.layers, 1],
        activation=model_config.lagrangian.activation,
    )

    model_safety = ModelSafety(
        actor, critic, cost_critic, lagrangian_net, env_net,
        use_subgoal_one_hot=use_subgoal_one_hot,
    )

    if state_dict is not None:
        model_safety.load_state_dict(state_dict, strict=use_env_net)
    model_safety.raw_feature_dim = raw_feature_dim
    model_safety.input_feat_dim = raw_feature_dim  # legacy alias for MA eval scripts
    model_safety.use_subgoal_one_hot = use_subgoal_one_hot
    from deploy.feature_recipe import infer_feat_recipe
    from envs.seq_wrapper import sar_task

    lidar_bins = sar_task(env).lidar_conf.num_bins
    if deploy_meta is not None:
        model_safety.feat_recipe = deploy_meta.get("feat_recipe", "sar_v1")
    else:
        model_safety.feat_recipe = infer_feat_recipe(raw_feature_dim, lidar_bins)
    return model_safety
