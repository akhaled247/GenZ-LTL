"""Smoke tests for SAR LTL eval routing (no checkpoint / Rabinizer on reset)."""
import numpy as np

from envs.env_utils import is_safety_model_env, make_safety_gym_env
from envs.ldba_wrapper import LDBAWrapper
from envs.ltl_wrapper import LTLWrapper
from envs.seq_wrapper import pre_process_obs_sar, sar_agent_obs_keys, sar_feat_dim
from ltl import FixedSampler
from ltl.logic import FrozenAssignment


def test_is_safety_model_env_routes_sar_wc():
    assert is_safety_model_env('PointLTL0MASAR1WC-v0')
    assert is_safety_model_env('PointLTL0MASAR2WC-v0')
    assert not is_safety_model_env('PointLTL0MASAR1-v0')
    assert is_safety_model_env('PointLtlSafety2-v0')


def test_masar1wc_ldba_wrapper_has_sar_preprocess():
    formula = '(!surface_0 U entrapped_0) & F surface_0'
    base = make_safety_gym_env('PointLTL0MASAR1WC-v0', flat=True)
    try:
        base.reset(seed=0)
        props = base.get_propositions()
        inner = LTLWrapper(base, FixedSampler.partial(formula)(props))
        env = LDBAWrapper(inner)
        assert hasattr(env, 'pre_process_obs_sar')
        reach = frozenset([FrozenAssignment({'entrapped_0': True})])
        avoid = frozenset()
        feat = env.pre_process_obs_sar(reach, avoid)
        feat_dim = sar_feat_dim(base.unwrapped.task.lidar_conf.num_bins)
        assert feat.shape == (feat_dim,)
        assert feat.dtype == np.float32
    finally:
        base.close()


def test_pre_process_obs_sar_shape_matches_feat_dim():
    formula = '(!surface_0 U entrapped_0) & F surface_0'
    base = make_safety_gym_env('PointLTL0MASAR1WC-v0', flat=True)
    try:
        base.reset(seed=1)
        props = base.get_propositions()
        inner = LTLWrapper(base, FixedSampler.partial(formula)(props))
        env = LDBAWrapper(inner)
        lidar_bins = base.unwrapped.task.lidar_conf.num_bins
        feat_dim = sar_feat_dim(lidar_bins)
        keys = sar_agent_obs_keys(0)
        reach = frozenset([FrozenAssignment({'entrapped_0': True})])
        avoid = frozenset()
        feat = pre_process_obs_sar(env, keys, reach, avoid, (feat_dim,), agent_idx=0)
        assert feat.shape == (feat_dim,)
    finally:
        base.close()
