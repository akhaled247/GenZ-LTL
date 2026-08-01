"""Load checkpoint + model for SAR MA deploy."""
from __future__ import annotations

from typing import Any

import gymnasium

from config import model_configs
from deploy.feature_recipe import attach_model_deploy_fields, ensure_deploy_meta
from envs import make_env_safety
from envs.seq_wrapper import sar_task
from ltl import FixedSampler
from model.model import build_model_safety
from utils.model_store import ModelStore
from utils.train_device import resolve_training_device

def load_model_for_deploy(
    train_env: str,
    exp: str,
    seed: int,
    formula: str,
    device: str = "cpu"
) -> tuple[Any, dict[str, Any], ModelStore]:
    if device != "cpu":
          device = resolve_training_device(device)
    model_store = ModelStore(train_env, exp, seed, None)
    training_status = model_store.load_training_status(map_location=device)

    probe_env = make_env_safety(
        train_env, FixedSampler.partial(formula), flat=True, sequence=False,
    )
    try:
        lidar_bins = sar_task(probe_env).lidar_conf.num_bins
    finally:
        probe_env.close()

    deploy_meta = ensure_deploy_meta(
        model_store.path, training_status, train_env, lidar_bins=lidar_bins,
    )
    config = model_configs[train_env]
    probe_env = make_env_safety(
        train_env, FixedSampler.partial(formula), flat=True, sequence=False,
    )
    try:
        model = build_model_safety(
            probe_env, training_status, config, deploy_meta=deploy_meta,
        )
        model.to(device)
        attach_model_deploy_fields(model, deploy_meta)
    finally:
        probe_env.close()
    return model, deploy_meta, model_store
