"""Curriculum avoid matching: walls single-prop vs dual walls/any_walls emission."""
from envs.seq_wrapper import assignment_hits_labels, avoid_forbids_walls
from ltl.logic import Assignment


PROPS = {"entrapped_0", "surface_0", "walls", "any_walls"}


def _single(p: str):
    return Assignment.single_proposition(p, PROPS).to_frozen()


def test_assignment_hits_labels_exact():
    live = _single("walls")
    avoid = frozenset({_single("walls"), _single("surface_0")})
    assert assignment_hits_labels(live, avoid)


def test_assignment_hits_labels_walls_dual_emission():
    """Live walls+any_walls must still hit curriculum avoid {walls}."""
    live = Assignment.where("walls", "any_walls", propositions=PROPS).to_frozen()
    avoid = frozenset({_single("walls")})
    assert assignment_hits_labels(live, avoid)
    assert avoid_forbids_walls(avoid)


def test_assignment_hits_labels_any_walls_avoid():
    live = Assignment.where("walls", "any_walls", propositions=PROPS).to_frozen()
    avoid = frozenset({_single("any_walls")})
    assert assignment_hits_labels(live, avoid)


def test_assignment_misses_unrelated_avoid():
    live = _single("walls")
    avoid = frozenset({_single("surface_0")})
    assert not assignment_hits_labels(live, avoid)
    assert not avoid_forbids_walls(avoid)
