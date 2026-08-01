"""SAR feature recipe for deploy (canonical 48-dim vs legacy padded checkpoints)."""
from __future__ import annotations

from typing import Any

from envs.seq_wrapper import sar_feat_dim
from utils.deploy_meta import FEAT_RECIPE_SAR_V1, build_deploy_meta, load_deploy_meta, save_deploy_meta

FEAT_RECIPE_LEGACY_V0 = "legacy_v0"


def canonical_raw_dim(lidar_bins: int, *, include_walls_lidar: bool = False) -> int:
    return sar_feat_dim(lidar_bins, include_walls_lidar=include_walls_lidar)


def infer_feat_recipe(raw_feature_dim: int, lidar_bins: int) -> str:
    canonical = {
        canonical_raw_dim(lidar_bins, include_walls_lidar=False),
        canonical_raw_dim(lidar_bins, include_walls_lidar=True),
    }
    if int(raw_feature_dim) in canonical:
        return FEAT_RECIPE_SAR_V1
    return FEAT_RECIPE_LEGACY_V0


def resolve_feat_shape(model: Any, lidar_bins: int) -> tuple[int, ...]:
    raw = getattr(model, "raw_feature_dim", None)
    if raw is not None:
        return (int(raw),)
    legacy = getattr(model, "input_feat_dim", None)
    if legacy is not None:
        return (int(legacy),)
    return (canonical_raw_dim(lidar_bins),)


def allow_legacy_padding(model: Any) -> bool:
    return getattr(model, "feat_recipe", FEAT_RECIPE_SAR_V1) == FEAT_RECIPE_LEGACY_V0


def attach_model_deploy_fields(model: Any, deploy_meta: dict[str, Any]) -> None:
    model.raw_feature_dim = int(deploy_meta["raw_feature_dim"])
    model.input_feat_dim = int(deploy_meta["raw_feature_dim"])
    model.feat_recipe = deploy_meta.get("feat_recipe", FEAT_RECIPE_SAR_V1)
    model.use_env_net_deploy = bool(deploy_meta.get("use_env_net", True))
    model.use_subgoal_one_hot = bool(deploy_meta.get("use_subgoal_one_hot", False))


def sar_preprocess_for_deploy(
    env: Any,
    model: Any,
    reach: Any,
    avoid: Any,
    *,
    agent_idx: int = 0,
) -> Any:
    """Build goal-conditioned SAR features using deploy_meta / model recipe."""
    from envs.seq_wrapper import sar_task

    lidar_bins = sar_task(env).lidar_conf.num_bins
    feat_shape = resolve_feat_shape(model, lidar_bins)
    legacy = allow_legacy_padding(model)
    if hasattr(env, "pre_process_obs_sar"):
        return env.pre_process_obs_sar(
            reach,
            avoid,
            agent_idx=agent_idx,
            feat_shape=feat_shape,
            allow_legacy_padding=legacy,
        )
    if hasattr(env, "pre_process_obs_zones"):
        return env.pre_process_obs_zones(reach, avoid)
    return env.pre_process_obs_letter(reach, avoid)


def ensure_deploy_meta(
    experiment_dir: str,
    training_status: dict[str, Any],
    train_env: str,
    *,
    lidar_bins: int = 16,
) -> dict[str, Any]:
    """Load or infer deploy_meta.json next to status.pth."""
    existing = load_deploy_meta(experiment_dir)
    if existing is not None:
        return existing

    from model.model import infer_model_safety_shapes

    inferred = infer_model_safety_shapes(training_status["model_state"])
    raw = int(inferred["feature_dim"])
    meta = build_deploy_meta(
        train_env=train_env,
        raw_feature_dim=raw,
        use_env_net=bool(inferred["use_env_net"]),
        actor_input_dim=int(inferred["embedding_dim"]),
        feat_recipe=infer_feat_recipe(raw, lidar_bins),
    )
    save_deploy_meta(experiment_dir, meta)
    return meta
