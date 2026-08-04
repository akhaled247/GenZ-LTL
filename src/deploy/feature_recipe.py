"""SAR feature recipe for deploy (canonical 48-dim vs legacy padded checkpoints)."""
from __future__ import annotations

from typing import Any

from envs.seq_wrapper import sar_feat_dim
from utils.deploy_meta import (
    FEAT_RECIPE_LEGACY_V0,
    FEAT_RECIPE_SAR_V1,
    FEAT_RECIPE_ZONE_COMPAT,
    build_deploy_meta,
    load_deploy_meta,
    save_deploy_meta,
)


def is_zone_safety_train_env(train_env: str | None) -> bool:
    """True for PointLtlSafety* zone pretrain (48-d agent|reach|avoid)."""
    return bool(train_env) and "PointLtlSafety" in str(train_env)


def resolve_zone_compat(
    train_env: str | None,
    zone_compat_requested: bool = False,
    *,
    feat_recipe: str | None = None,
) -> bool:
    """Zone-compat packing only for Zone→SAR transfer.

    SAR-trained ckpts (MASAR*) always keep proposition-independent buildings +
    walls lidar slices (``sar_v1``), even if CLI/meta asks for zone_compat.
    """
    if not is_zone_safety_train_env(train_env):
        return False
    if feat_recipe == FEAT_RECIPE_ZONE_COMPAT:
        return True
    return bool(zone_compat_requested)


def canonical_raw_dim(
    lidar_bins: int,
    *,
    include_walls_lidar: bool = False,
    zone_compat: bool = False,
) -> int:
    return sar_feat_dim(
        lidar_bins,
        include_walls_lidar=include_walls_lidar,
        zone_compat=zone_compat,
    )


def infer_feat_recipe(raw_feature_dim: int, lidar_bins: int) -> str:
    if int(raw_feature_dim) == canonical_raw_dim(lidar_bins, zone_compat=True):
        return FEAT_RECIPE_ZONE_COMPAT
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
    zone_compat = getattr(model, "feat_recipe", None) == FEAT_RECIPE_ZONE_COMPAT
    return (canonical_raw_dim(lidar_bins, zone_compat=zone_compat),)


def allow_legacy_padding(model: Any) -> bool:
    return getattr(model, "feat_recipe", FEAT_RECIPE_SAR_V1) == FEAT_RECIPE_LEGACY_V0


def model_uses_zone_compat(model: Any) -> bool:
    return getattr(model, "feat_recipe", None) == FEAT_RECIPE_ZONE_COMPAT


def attach_model_deploy_fields(model: Any, deploy_meta: dict[str, Any]) -> None:
    model.raw_feature_dim = int(deploy_meta["raw_feature_dim"])
    model.input_feat_dim = int(deploy_meta["raw_feature_dim"])
    model.feat_recipe = deploy_meta.get("feat_recipe", FEAT_RECIPE_SAR_V1)
    model.use_env_net_deploy = bool(deploy_meta.get("use_env_net", True))
    model.use_subgoal_one_hot = bool(deploy_meta.get("use_subgoal_one_hot", False))
    model.train_env = deploy_meta.get("train_env")


def apply_zone_compat_deploy_meta(deploy_meta: dict[str, Any], lidar_bins: int = 16) -> dict[str, Any]:
    """Force zone_compat recipe fields for Zone→SAR eval (copy).

    No-op (returns copy unchanged recipe-wise) when ``train_env`` is SAR — those
    ckpts keep independent buildings/walls slices.
    """
    meta = dict(deploy_meta)
    train_env = meta.get("train_env")
    if not is_zone_safety_train_env(train_env):
        # Keep / restore sar_v1 dims from raw_feature_dim when possible.
        raw = int(meta.get("raw_feature_dim", canonical_raw_dim(lidar_bins, include_walls_lidar=True)))
        meta["feat_recipe"] = infer_feat_recipe(raw, lidar_bins)
        if meta["feat_recipe"] == FEAT_RECIPE_ZONE_COMPAT:
            # Ambiguous 48-d on a SAR train_env — prefer walls-on sar_v1 layout.
            meta["feat_recipe"] = FEAT_RECIPE_SAR_V1
            meta["raw_feature_dim"] = canonical_raw_dim(lidar_bins, include_walls_lidar=True)
        return meta
    meta["feat_recipe"] = FEAT_RECIPE_ZONE_COMPAT
    meta["raw_feature_dim"] = canonical_raw_dim(lidar_bins, zone_compat=True)
    return meta


def ensure_sar_v1_indep_lidars(
    deploy_meta: dict[str, Any],
    *,
    lidar_bins: int = 16,
    include_walls_lidar: bool | None = None,
) -> dict[str, Any]:
    """For SAR-trained runs, force proposition-independent buildings (+ walls)."""
    meta = dict(deploy_meta)
    train_env = meta.get("train_env")
    if is_zone_safety_train_env(train_env):
        return meta

    raw = int(meta.get("raw_feature_dim", 0))
    dim_no_walls = canonical_raw_dim(lidar_bins, include_walls_lidar=False)
    dim_walls = canonical_raw_dim(lidar_bins, include_walls_lidar=True)
    dim_zone = canonical_raw_dim(lidar_bins, zone_compat=True)

    if include_walls_lidar is None:
        if raw == dim_walls:
            include_walls = True
        elif raw == dim_no_walls:
            include_walls = False
        elif raw == dim_zone or meta.get("feat_recipe") == FEAT_RECIPE_ZONE_COMPAT:
            # Stale zone_compat meta on a SAR ckpt — restore L1-style walls channel.
            include_walls = True
            meta["raw_feature_dim"] = dim_walls
        else:
            # Prefer walls for L1+/named wall levels; else keep raw and recipe infer.
            include_walls = any(
                tag in str(train_env)
                for tag in ("MASAR1", "LTL1", "LTL2", "MASAR2")
            ) and "LTL0" not in str(train_env)
            if include_walls and raw not in (dim_no_walls, dim_walls):
                meta["raw_feature_dim"] = dim_walls
    else:
        include_walls = bool(include_walls_lidar)
        meta["raw_feature_dim"] = canonical_raw_dim(
            lidar_bins, include_walls_lidar=include_walls,
        )

    recipe = infer_feat_recipe(int(meta["raw_feature_dim"]), lidar_bins)
    if recipe == FEAT_RECIPE_ZONE_COMPAT:
        recipe = FEAT_RECIPE_SAR_V1
        meta["raw_feature_dim"] = canonical_raw_dim(
            lidar_bins, include_walls_lidar=include_walls,
        )
    meta["feat_recipe"] = recipe
    meta["entr_bldg_obs"] = bool(meta.get("entr_bldg_obs", False))
    return meta


def _env_spec_id(env: Any) -> str:
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None) if spec is not None else None
    return env_id if isinstance(env_id, str) else ""


def sar_preprocess_for_deploy(
    env: Any,
    model: Any,
    reach: Any,
    avoid: Any,
    *,
    agent_idx: int = 0,
) -> Any:
    """Build goal-conditioned features for SAR or zone safety deploy.

    ``LDBAWrapper`` exposes both ``pre_process_obs_sar`` and
    ``pre_process_obs_zones``. Prefer zones for ``PointLtlSafety*`` so color
    props (``blue``, ``green``, …) are not parsed as SAR casualty keys.

    Packing follows ``model.feat_recipe`` (not env CLI alone): SAR-trained
    models keep independent buildings/walls lidars.
    """
    env_id = _env_spec_id(env)
    if "PointLtlSafety" in env_id and hasattr(env, "pre_process_obs_zones"):
        return env.pre_process_obs_zones(reach, avoid)
    if "LetterSafety" in env_id and hasattr(env, "pre_process_obs_letter"):
        return env.pre_process_obs_letter(reach, avoid)

    from envs.seq_wrapper import sar_task

    lidar_bins = sar_task(env).lidar_conf.num_bins
    feat_shape = resolve_feat_shape(model, lidar_bins)
    legacy = allow_legacy_padding(model)
    train_env = getattr(model, "train_env", None)
    zone_compat = resolve_zone_compat(
        train_env,
        model_uses_zone_compat(model),
        feat_recipe=getattr(model, "feat_recipe", None),
    )
    if hasattr(env, "pre_process_obs_sar"):
        return env.pre_process_obs_sar(
            reach,
            avoid,
            agent_idx=agent_idx,
            feat_shape=feat_shape,
            allow_legacy_padding=legacy,
            zone_compat=zone_compat,
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
        meta = dict(existing)
        meta.setdefault("train_env", train_env)
        return ensure_sar_v1_indep_lidars(meta, lidar_bins=lidar_bins)

    from model.model import infer_model_safety_shapes

    inferred = infer_model_safety_shapes(training_status["model_state"])
    raw = int(inferred["feature_dim"])
    recipe = infer_feat_recipe(raw, lidar_bins)
    if not is_zone_safety_train_env(train_env) and recipe == FEAT_RECIPE_ZONE_COMPAT:
        recipe = FEAT_RECIPE_SAR_V1
    meta = build_deploy_meta(
        train_env=train_env,
        raw_feature_dim=raw,
        use_env_net=bool(inferred["use_env_net"]),
        actor_input_dim=int(inferred["embedding_dim"]),
        feat_recipe=recipe,
    )
    meta = ensure_sar_v1_indep_lidars(meta, lidar_bins=lidar_bins)
    save_deploy_meta(experiment_dir, meta)
    return meta
