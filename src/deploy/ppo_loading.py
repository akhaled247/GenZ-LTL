"""Load PPO+LTL checkpoints for SAR deploy eval."""
from __future__ import annotations

from typing import Any

from config import model_configs
from deploy.checkpoint_kind import detect_rl_algo, ppo_model_config_key
from envs import make_env
from envs.seq_wrapper import sar_feat_dim, sar_task
from ltl import FixedSampler
from model.model import build_model
from utils.deploy_meta import FEAT_RECIPE_SAR_V1
from utils.model_store import ModelStore


def attach_ppo_sar_fields(model: Any, env: Any) -> None:
    lidar_bins = sar_task(env).lidar_conf.num_bins
    model.raw_feature_dim = int(sar_feat_dim(lidar_bins))
    model.input_feat_dim = model.raw_feature_dim
    model.feat_recipe = FEAT_RECIPE_SAR_V1


def load_ppo_model_for_deploy(
    train_env: str,
    exp: str,
    seed: int,
    formula: str,
):
    model_store = ModelStore(train_env, exp, seed, None)
    training_status = model_store.load_training_status(map_location="cpu")
    if detect_rl_algo(training_status["model_state"]) != "ppo":
        raise ValueError(f"Experiment {exp} is not a PPO checkpoint (use RCO deploy path)")
    model_store.load_vocab()

    sampler = FixedSampler.partial(formula)
    probe_env = make_env(train_env, sampler, flat=True, sequence=True, max_steps=2500)
    try:
        config_key = ppo_model_config_key(train_env)
        model = build_model(probe_env, training_status, model_configs[config_key])
        attach_ppo_sar_fields(model, probe_env)
    finally:
        probe_env.close()
    return model, model_store
