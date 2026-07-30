"""MA SAR deploy: gate Büchi subgoals to clean entrapped → surface feature phases."""
from __future__ import annotations

from typing import Any, FrozenSet

from ltl.logic import Assignment, FrozenAssignment

SURFACE_PROPS = ("surface_0", "surface_1", "all_surface")
ENTRAPPED_TEAM = "all_entrapped"
SURFACE_TEAM = "all_surface"


def should_use_ma_phase_gating(propositions: set[str] | frozenset[str]) -> bool:
    return ENTRAPPED_TEAM in propositions and SURFACE_TEAM in propositions


def _single_prop_assignment(prop: str, propositions: set[str]) -> FrozenAssignment:
    return Assignment.single_proposition(prop, propositions).to_frozen()


def _find_task(env: Any) -> Any:
    cur: Any = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "task"):
            return cur.task
        cur = getattr(cur, "env", None)
    raise RuntimeError("SAR task not found on env wrapper chain")


def _all_entrapped_rescued(env: Any) -> bool:
    task = _find_task(env)
    geom = getattr(task, "entrapped_casualtys", None)
    if geom is None or getattr(geom, "num", 0) <= 0:
        return False
    return all(geom.rescued)


def gated_reach_avoid_for_features(
    env: Any,
    reach: FrozenSet[FrozenAssignment],
    avoid: FrozenSet[FrozenAssignment],
    propositions: set[str],
) -> tuple[FrozenSet[FrozenAssignment], FrozenSet[FrozenAssignment]]:
    """Replace messy Büchi reach/avoid frozensets with two-phase SAR deploy features.

    Phase A (until all entrapped rescued): reach ``all_entrapped``; avoid all surface props.
    Phase B: reach ``all_surface`` only; avoid empty.

    Büchi search / LTL tracking still uses the original ``sequence``; only policy features
    are gated so they match SA training (single reach prop, no LDBA label compounds).
    """
    if not should_use_ma_phase_gating(propositions):
        return reach, avoid

    props = set(propositions)
    surface_avoid = frozenset(
        _single_prop_assignment(p, props)
        for p in SURFACE_PROPS
        if p in props
    )

    if not _all_entrapped_rescued(env):
        return (
            frozenset({_single_prop_assignment(ENTRAPPED_TEAM, props)}),
            surface_avoid,
        )

    return (
        frozenset({_single_prop_assignment(SURFACE_TEAM, props)}),
        frozenset(),
    )
