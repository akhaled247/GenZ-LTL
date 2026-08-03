"""Deploy metadata for SA train → MA eval parity (persisted next to status.pth)."""
from __future__ import annotations

import json
import os
from typing import Any

DEPLOY_META_FILENAME = "deploy_meta.json"
MA_EVAL_ENV_DEFAULT = "PointLTL0MASAR2WC-v0"
MA_EVAL_FORMULA_DEFAULT = (
    "((!surface_0 & !surface_1) U all_entrapped) & (F surface_0 & F surface_1)"
)
# Per-agent Until on pulse-only entrapped_i is fragile (no sticky props) — #54.
_FRAGILE_MA_UNTIL_MARKERS = (
    "U entrapped_0",
    "U entrapped_1",
)
FEAT_RECIPE_SAR_V1 = "sar_v1"
FEAT_RECIPE_LEGACY_V0 = "legacy_v0"


def warn_if_fragile_ma_formula(formula: str) -> None:
    """Print warning when MA formula uses per-agent entrapped Until (false violations)."""
    compact = " ".join(formula.split())
    if any(marker in compact for marker in _FRAGILE_MA_UNTIL_MARKERS):
        print(
            "WARNING: MA formula uses per-agent 'U entrapped_i' — entrapped props are "
            "one-shot pulses, so Until often never releases (false violations). "
            f"Prefer: {MA_EVAL_FORMULA_DEFAULT}"
        )


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
    use_subgoal_one_hot: bool = False,
    entr_bldg_obs: bool = False,
    num_propositions: int | None = None,
) -> dict[str, Any]:
    meta = {
        "train_env": train_env,
        "eval_env": eval_env,
        "raw_feature_dim": int(raw_feature_dim),
        "actor_input_dim": int(actor_input_dim),
        "use_env_net": bool(use_env_net),
        "feat_recipe": feat_recipe,
        "ma_eval_formula": ma_eval_formula,
        "use_subgoal_one_hot": bool(use_subgoal_one_hot),
        "entr_bldg_obs": bool(entr_bldg_obs),
    }
    if num_propositions is not None:
        meta["num_propositions"] = int(num_propositions)
    return meta
