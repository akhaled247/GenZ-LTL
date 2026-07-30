"""Deploy metadata for SA train → MA eval parity (persisted next to status.pth)."""
from __future__ import annotations

import json
import os
from typing import Any

DEPLOY_META_FILENAME = "deploy_meta.json"
MA_EVAL_ENV_DEFAULT = "PointLTL0MASAR2WC-v0"
MA_EVAL_FORMULA_DEFAULT = (
    "((!surface_0 U entrapped_0) & F surface_0) "
    "& ((!surface_1 U entrapped_1) & F surface_1)"
)
FEAT_RECIPE_SAR_V1 = "sar_v1"


def deploy_meta_path(experiment_dir: str) -> str:
    return os.path.join(experiment_dir, DEPLOY_META_FILENAME)


def load_deploy_meta(experiment_dir: str) -> dict[str, Any] | None:
    path = deploy_meta_path(experiment_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_deploy_meta(experiment_dir: str, meta: dict[str, Any]) -> str:
    os.makedirs(experiment_dir, exist_ok=True)
    path = deploy_meta_path(experiment_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def build_deploy_meta(
    *,
    train_env: str,
    raw_feature_dim: int,
    use_env_net: bool,
    actor_input_dim: int,
    eval_env: str = MA_EVAL_ENV_DEFAULT,
    ma_eval_formula: str = MA_EVAL_FORMULA_DEFAULT,
    feat_recipe: str = FEAT_RECIPE_SAR_V1,
) -> dict[str, Any]:
    return {
        "train_env": train_env,
        "eval_env": eval_env,
        "raw_feature_dim": int(raw_feature_dim),
        "actor_input_dim": int(actor_input_dim),
        "use_env_net": bool(use_env_net),
        "feat_recipe": feat_recipe,
        "ma_eval_formula": ma_eval_formula,
    }
