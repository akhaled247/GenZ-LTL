"""Tests for PPO vs RCO checkpoint detection."""
import pytest

from deploy.checkpoint_kind import detect_rl_algo, ppo_model_config_key


def test_detect_ppo_from_ltl_net():
    state = {"ltl_net.rnn.weight_ih_l0": None, "actor.0.weight": None}
    assert detect_rl_algo(state) == "ppo"


def test_detect_rco_from_cost_critic():
    state = {"cost_critic.0.weight": None, "actor.0.weight": None}
    assert detect_rl_algo(state) == "rco"


def test_ppo_model_config_key_masar():
    assert ppo_model_config_key("PointLTL0MASAR1WC-v0") == "PointLTL0MASAR1-v0"


def test_detect_unknown_raises():
    with pytest.raises(ValueError):
        detect_rl_algo({"actor.0.weight": None})
