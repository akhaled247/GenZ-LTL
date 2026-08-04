"""walls must never appear in Büchi reach sets (SAR)."""
from __future__ import annotations

from ltl.logic import Assignment
from sequence.search.exhaustive_search import sanitize_reach_avoid_walls, WALLS_PROP


def test_sanitize_strips_walls_from_reach_and_forces_avoid():
    props = {"entrapped_0", "surface_0", WALLS_PROP}
    # Compound LDBA assignment: entrapped + walls co-active
    reach = Assignment.where("entrapped_0", WALLS_PROP, propositions=props).to_frozen()
    avoid = frozenset([
        Assignment.single_proposition("surface_0", props).to_frozen(),
    ])
    out = sanitize_reach_avoid_walls(reach, avoid, props)
    assert out is not None
    new_reach, new_avoid = out
    reach_props = set()
    for a in new_reach:
        reach_props |= set(a.get_true_propositions())
    assert reach_props == {"entrapped_0"}
    assert WALLS_PROP not in reach_props
    avoid_props = set()
    for a in new_avoid:
        avoid_props |= set(a.get_true_propositions())
    assert WALLS_PROP in avoid_props
    assert "surface_0" in avoid_props


def test_strip_walls_from_expanded_reach_set():
    props = {"all_entrapped", "surface_0", WALLS_PROP}
    # What buggy ExhaustiveSearchSafety emitted: walls as its own reach atom
    reach = frozenset([
        Assignment.single_proposition("all_entrapped", props).to_frozen(),
        Assignment.single_proposition(WALLS_PROP, props).to_frozen(),
    ])
    avoid = frozenset([Assignment.single_proposition("surface_0", props).to_frozen()])
    from sequence.search.exhaustive_search import strip_walls_from_reach_set
    out = strip_walls_from_reach_set(reach, avoid, props)
    assert out is not None
    new_reach, new_avoid = out
    reach_props = {p for a in new_reach for p in a.get_true_propositions()}
    assert reach_props == {"all_entrapped"}
    assert WALLS_PROP in {p for a in new_avoid for p in a.get_true_propositions()}


def test_sanitize_rejects_walls_only_reach():
    props = {"entrapped_0", WALLS_PROP}
    reach = Assignment.single_proposition(WALLS_PROP, props).to_frozen()
    avoid = frozenset()
    assert sanitize_reach_avoid_walls(reach, avoid, props) is None


def test_sanitize_noop_when_no_walls_in_alphabet():
    props = {"entrapped_0", "surface_0"}
    reach = Assignment.single_proposition("entrapped_0", props).to_frozen()
    avoid = frozenset([Assignment.single_proposition("surface_0", props).to_frozen()])
    out = sanitize_reach_avoid_walls(reach, avoid, props)
    assert out is not None
    new_reach, new_avoid = out
    assert len(new_reach) == 1
    assert WALLS_PROP not in {p for a in new_avoid for p in a.get_true_propositions()}
