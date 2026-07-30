"""Canonical SAR SA→MA deploy package (paper §5.3).

Import submodules directly to avoid circular imports, e.g.:
  from deploy.feature_recipe import ensure_deploy_meta
  from deploy.coordinator import MultiAgentSARCoordinator
  from deploy.ma_rollout import simulate_ma_sar
"""

__all__ = [
    "feature_recipe",
    "coordinator",
    "ma_rollout",
    "loading",
    "env_check",
]
