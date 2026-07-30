"""Backfill deploy_meta.json for an existing RCO experiment (no retrain)."""
from __future__ import annotations

import argparse

import torch

from model.model import infer_model_safety_shapes
from utils.deploy_meta import build_deploy_meta, save_deploy_meta
from utils.model_store import ModelStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PointLTL0MASAR1WC-v0")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    store = ModelStore(args.env, args.name, args.seed, None)
    status = store.load_training_status(map_location="cpu")
    state = status["model_state"]
    inferred = infer_model_safety_shapes(state)
    raw_feature_dim = int(inferred["feature_dim"])
    use_env_net = bool(inferred["use_env_net"])
    actor_input_dim = int(inferred["embedding_dim"])
    meta = build_deploy_meta(
        train_env=args.env,
        raw_feature_dim=raw_feature_dim,
        use_env_net=use_env_net,
        actor_input_dim=actor_input_dim,
    )
    path = save_deploy_meta(store.path, meta)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
