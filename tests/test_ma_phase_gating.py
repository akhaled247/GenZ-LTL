"""Tests for MA deploy hierarchical per-agent reach/avoid features."""
from unittest.mock import MagicMock

from deploy.ma_phase_gating import (
    ANY_WALLS,
    ENTRAPPED_TEAM,
    SURFACE_TEAM,
    WALLS_PROP,
    gated_reach_avoid_for_features,
    hierarchical_reach_avoid_for_agent,
    should_use_ma_hierarchy,
    should_use_ma_phase_gating,
)
from ltl.logic import Assignment


PROPS = {
    "surface_0", "surface_1", "entrapped_0", "entrapped_1",
    ENTRAPPED_TEAM, SURFACE_TEAM,
}

PROPS_WITH_WALLS = PROPS | {WALLS_PROP, ANY_WALLS}


def _mock_env(*, all_entrapped: bool):
    task = MagicMock()
    task.entrapped_casualtys = MagicMock()
    task.entrapped_casualtys.num = 2
    task.entrapped_casualtys.rescued = [True, True] if all_entrapped else [False, False]
    env = MagicMock()
    env.task = task
    return env


def _true_names(assignments):
    return {next(iter(a.get_true_propositions())) for a in assignments}


def test_should_use_ma_hierarchy():
    assert should_use_ma_hierarchy(PROPS, num_agents=2)
    assert not should_use_ma_hierarchy(PROPS, num_agents=1)
    assert not should_use_ma_hierarchy({"surface_0", "entrapped_0"}, num_agents=2)
    assert should_use_ma_phase_gating(PROPS)


def test_hierarchical_entrapped_phase_per_agent():
    env = _mock_env(all_entrapped=False)
    reach0, avoid0 = hierarchical_reach_avoid_for_agent(env, 0, PROPS)
    reach1, avoid1 = hierarchical_reach_avoid_for_agent(env, 1, PROPS)
    assert _true_names(reach0) == {"entrapped_0"}
    assert _true_names(reach1) == {"entrapped_1"}
    assert _true_names(avoid0) == {"surface_0"}
    assert _true_names(avoid1) == {"surface_1"}


def test_hierarchical_entrapped_phase_includes_any_walls():
    env = _mock_env(all_entrapped=False)
    reach, avoid = hierarchical_reach_avoid_for_agent(env, 0, PROPS_WITH_WALLS)
    assert _true_names(reach) == {"entrapped_0"}
    assert _true_names(avoid) == {"surface_0", ANY_WALLS}


def test_hierarchical_surface_phase():
    env = _mock_env(all_entrapped=True)
    reach0, avoid0 = hierarchical_reach_avoid_for_agent(env, 0, PROPS)
    reach1, avoid1 = hierarchical_reach_avoid_for_agent(env, 1, PROPS)
    assert _true_names(reach0) == {"surface_0"}
    assert _true_names(reach1) == {"surface_1"}
    assert avoid0 == frozenset()
    assert avoid1 == frozenset()


def test_hierarchical_surface_phase_keeps_walls_avoid():
    env = _mock_env(all_entrapped=True)
    _, avoid = hierarchical_reach_avoid_for_agent(env, 1, PROPS_WITH_WALLS)
    assert _true_names(avoid) == {ANY_WALLS}


def test_gated_uses_hierarchy_with_agent_idx():
    env = _mock_env(all_entrapped=False)
    reach, avoid = gated_reach_avoid_for_features(
        env, frozenset(), frozenset(), PROPS, agent_idx=1, num_agents=2,
    )
    assert _true_names(reach) == {"entrapped_1"}
    assert _true_names(avoid) == {"surface_1"}


def test_passthrough_without_team_props():
    env = _mock_env(all_entrapped=False)
    raw_reach = frozenset([Assignment.single_proposition("entrapped_0", PROPS).to_frozen()])
    raw_avoid = frozenset()
    props = {"surface_0", "entrapped_0"}
    reach, avoid = gated_reach_avoid_for_features(
        env, raw_reach, raw_avoid, props, agent_idx=0, num_agents=2,
    )
    assert reach is raw_reach
    assert avoid is raw_avoid
