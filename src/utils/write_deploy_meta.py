"""Backfill deploy_meta.json for an existing RCO experiment (no retrain)."""
from __future__ import annotations

import argparse

from envs.sar_features import ensure_deploy_meta
from envs import make_env_safety
from envs.seq_wrapper import sar_task
from ltl import FixedSampler
from utils.model_store import ModelStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PointLTL0MASAR1WC-v0")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--formula",
        default="(!surface_0 U entrapped_0) & F surface_0",
    )
    args = parser.parse_args()

    store = ModelStore(args.env, args.name, args.seed, None)
    status = store.load_training_status(map_location="cpu")
    probe = make_env_safety(args.env, FixedSampler.partial(args.formula), flat=True, sequence=False)
    try:
        lidar_bins = sar_task(probe).lidar_conf.num_bins
    finally:
        probe.close()
    meta = ensure_deploy_meta(store.path, status, args.env, lidar_bins=lidar_bins)
    print(f"feat_recipe={meta.get('feat_recipe')}")
    print(f"Wrote {store.path}/deploy_meta.json")


if __name__ == "__main__":
    main()
