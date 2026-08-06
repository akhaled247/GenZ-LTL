"""walls must never appear in Büchi reach sets (SAR)."""
from __future__ import annotations

from ltl.automata import LDBASequence
from ltl.logic import Assignment
from sequence.search.exhaustive_search import (
    WALLS_PROP,
    sanitize_ldba_sequence_walls,
    sanitize_reach_avoid_walls,
    sequence_has_surface_before_entrapped,
    strip_walls_from_reach_set,
)


def test_sanitize_strips_walls_from_reach_and_forces_avoid():
    props = {"entrapped_0", "surface_0", WALLS_PROP}
    reach = Assignment.where("entrapped_0", WALLS_PROP, propositions=props).to_frozen()
    avoid = frozenset([
        Assignment.single_proposition("surface_0", props).to_frozen(),
    ])
    out = sanitize_reach_avoid_walls(reach, avoid, props)
    assert out is not None
    new_reach, new_avoid = out
    reach_props = {p for a in new_reach for p in a.get_true_propositions()}
    assert reach_props == {"entrapped_0"}
    avoid_props = {p for a in new_avoid for p in a.get_true_propositions()}
    assert WALLS_PROP in avoid_props and "surface_0" in avoid_props


def test_sanitize_drops_reach_prop_that_is_also_avoided():
    props = {"entrapped_0", "surface_0", WALLS_PROP}
    reach = Assignment.single_proposition("surface_0", props).to_frozen()
    avoid = frozenset([
        Assignment.single_proposition("surface_0", props).to_frozen(),
        Assignment.single_proposition(WALLS_PROP, props).to_frozen(),
    ])
    assert sanitize_reach_avoid_walls(reach, avoid, props) is None


def test_strip_walls_from_expanded_reach_set():
    props = {"all_entrapped", "surface_0", WALLS_PROP}
    reach = frozenset([
        Assignment.single_proposition("all_entrapped", props).to_frozen(),
        Assignment.single_proposition(WALLS_PROP, props).to_frozen(),
    ])
    avoid = frozenset([Assignment.single_proposition("surface_0", props).to_frozen()])
    out = strip_walls_from_reach_set(reach, avoid, props)
    assert out is not None
    new_reach, new_avoid = out
    assert {p for a in new_reach for p in a.get_true_propositions()} == {"all_entrapped"}
    assert WALLS_PROP in {p for a in new_avoid for p in a.get_true_propositions()}


def test_reject_surface_before_entrapped_order():
    props = {"entrapped_0", "surface_0", WALLS_PROP}
    bad = LDBASequence([
        (
            frozenset([Assignment.single_proposition("surface_0", props).to_frozen()]),
            frozenset([Assignment.single_proposition(WALLS_PROP, props).to_frozen()]),
        ),
        (
            frozenset([Assignment.single_proposition("entrapped_0", props).to_frozen()]),
            frozenset([Assignment.single_proposition(WALLS_PROP, props).to_frozen()]),
        ),
    ])
    assert sequence_has_surface_before_entrapped(bad)
    assert sanitize_ldba_sequence_walls(bad, props) is None

    good = LDBASequence([
        (
            frozenset([Assignment.single_proposition("entrapped_0", props).to_frozen()]),
            frozenset([
                Assignment.single_proposition("surface_0", props).to_frozen(),
                Assignment.single_proposition(WALLS_PROP, props).to_frozen(),
            ]),
        ),
        (
            frozenset([Assignment.single_proposition("surface_0", props).to_frozen()]),
            frozenset([Assignment.single_proposition(WALLS_PROP, props).to_frozen()]),
        ),
    ])
    assert not sequence_has_surface_before_entrapped(good)
    assert sanitize_ldba_sequence_walls(good, props) is not None


def test_allow_surface_only_after_until_done():
    """Mid-episode F surface — sequence has no entrapped stage."""
    props = {"entrapped_0", "surface_0", WALLS_PROP}
    surface_only = LDBASequence([
        (
            frozenset([Assignment.single_proposition("surface_0", props).to_frozen()]),
            frozenset([Assignment.single_proposition(WALLS_PROP, props).to_frozen()]),
        ),
    ])
    assert not sequence_has_surface_before_entrapped(surface_only)
    assert sanitize_ldba_sequence_walls(surface_only, props) is not None
