"""Smoke tests for MASAR2 coordinator eval wiring (no checkpoint / Rabinizer on reset)."""
from envs.env_utils import make_safety_gym_env
from envs.ldba_wrapper import LDBAWrapper
from envs.ltl_wrapper import LTLWrapper
from envs.seq_wrapper import sar_feat_dim
from ltl import FixedSampler


def test_masar2wc_non_flat_env_builds():
    base = make_safety_gym_env('PointLTL0MASAR2WC-v0', flat=False)
    try:
        obs, _ = base.reset(seed=0)
        assert getattr(base.unwrapped.task, 'agent_num', 1) == 2
        assert 'agent_0' in obs and 'agent_1' in obs
    finally:
        base.close()


def test_masar2_per_agent_preprocess_shapes():
    formula = '(!surface_1 U entrapped_1) & F surface_1'
    base = make_safety_gym_env('PointLTL0MASAR2WC-v0', flat=False)
    try:
        base.reset(seed=2)
        props = base.get_propositions()
        env = LDBAWrapper(LTLWrapper(base, FixedSampler.partial(formula)(props)))
        reach = frozenset()
        avoid = frozenset()
        for agent_idx in (0, 1):
            feat = env.pre_process_obs_sar(reach, avoid, agent_idx=agent_idx)
            feat_dim = sar_feat_dim(base.unwrapped.task.lidar_conf.num_bins)
            assert feat.shape == (feat_dim,)
    finally:
        base.close()
