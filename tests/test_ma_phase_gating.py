"""Tests for MA deploy phase gating of reach/avoid features."""
from unittest.mock import MagicMock

from deploy.ma_phase_gating import (
    ENTRAPPED_TEAM,
    SURFACE_TEAM,
    gated_reach_avoid_for_features,
    should_use_ma_phase_gating,
)
from ltl.logic import Assignment


PROPS = {
    "surface_0", "surface_1", "entrapped_0", "entrapped_1",
    ENTRAPPED_TEAM, SURFACE_TEAM,
}


def _mock_env(*, all_entrapped: bool):
    task = MagicMock()
    task.entrapped_casualtys = MagicMock()
    task.entrapped_casualtys.num = 2
    task.entrapped_casualtys.rescued = [True, True] if all_entrapped else [False, False]
    env = MagicMock()
    env.task = task
    return env


def test_should_use_ma_phase_gating():
    assert should_use_ma_phase_gating(PROPS)
    assert not should_use_ma_phase_gating({"surface_0", "entrapped_0"})


def test_gated_entrapped_phase():
    env = _mock_env(all_entrapped=False)
    reach, avoid = gated_reach_avoid_for_features(env, frozenset(), frozenset(), PROPS)
    assert len(reach) == 1
    assert next(iter(reach)).get_true_propositions() == {ENTRAPPED_TEAM}
    assert {next(iter(a.get_true_propositions())) for a in avoid} == {
        "surface_0", "surface_1", SURFACE_TEAM,
    }


def test_gated_surface_phase():
    env = _mock_env(all_entrapped=True)
    reach, avoid = gated_reach_avoid_for_features(env, frozenset(), frozenset(), PROPS)
    assert len(reach) == 1
    assert next(iter(reach)).get_true_propositions() == {SURFACE_TEAM}
    assert avoid == frozenset()


def test_passthrough_without_team_props():
    env = _mock_env(all_entrapped=False)
    raw_reach = frozenset([Assignment.single_proposition("entrapped_0", PROPS).to_frozen()])
    raw_avoid = frozenset()
    props = {"surface_0", "entrapped_0"}
    reach, avoid = gated_reach_avoid_for_features(env, raw_reach, raw_avoid, props)
    assert reach == raw_reach
    assert avoid == raw_avoid
