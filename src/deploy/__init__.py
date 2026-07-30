"""Canonical SAR SA→MA deploy package (paper §5.3)."""
from deploy.coordinator import MultiAgentSARCoordinator
from deploy.ma_rollout import simulate_ma_sar
from deploy.feature_recipe import (
    FEAT_RECIPE_LEGACY_V0,
    allow_legacy_padding,
    ensure_deploy_meta,
    resolve_feat_shape,
)

__all__ = [
    "MultiAgentSARCoordinator",
    "simulate_ma_sar",
    "FEAT_RECIPE_LEGACY_V0",
    "allow_legacy_padding",
    "ensure_deploy_meta",
    "resolve_feat_shape",
]
