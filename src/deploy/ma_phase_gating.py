"""MA SAR deploy: hierarchical per-agent reach/avoid feature phases."""
from __future__ import annotations

from typing import Any, FrozenSet

from ltl.logic import Assignment, FrozenAssignment

SURFACE_TEAM = "all_surface"
ENTRAPPED_TEAM = "all_entrapped"
WALLS_PROP = "walls"
ANY_WALLS = "any_walls"


def should_use_ma_hierarchy(
    propositions: set[str] | frozenset[str],
    num_agents: int = 2,
) -> bool:
    """True for symmetric team MA missions (v1 hierarchical feature recipe)."""
    return (
        num_agents >= 2
        and ENTRAPPED_TEAM in propositions
        and SURFACE_TEAM in propositions
    )


# Backwards-compatible alias used by older tests / imports.
def should_use_ma_phase_gating(propositions: set[str] | frozenset[str]) -> bool:
    return should_use_ma_hierarchy(propositions, num_agents=2)


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


def _walls_avoid(propositions: set[str]) -> FrozenSet[FrozenAssignment]:
    """Prefer ``any_walls`` when present; else ``walls``."""
    if ANY_WALLS in propositions:
        return frozenset({_single_prop_assignment(ANY_WALLS, propositions)})
    if WALLS_PROP in propositions:
        return frozenset({_single_prop_assignment(WALLS_PROP, propositions)})
    return frozenset()


def hierarchical_reach_avoid_for_agent(
    env: Any,
    agent_idx: int,
    propositions: set[str],
) -> tuple[FrozenSet[FrozenAssignment], FrozenSet[FrozenAssignment]]:
    """Per-agent two-phase SAR features for symmetric team formulas.

    Phase A (until all entrapped rescued): reach ``entrapped_i``; avoid ``surface_i`` + walls.
    Phase B: reach ``surface_i``; avoid walls / ``any_walls``.

    Büchi search / LTL tracking stay on the shared team formula; only policy features
    are hierarchical so they match SA training (single local reach prop).
    """
    props = set(propositions)
    walls_avoid = _walls_avoid(props)
    entrapped = f"entrapped_{agent_idx}"
    surface = f"surface_{agent_idx}"
    if entrapped not in props or surface not in props:
        raise ValueError(
            f"Hierarchical MA features need {entrapped!r} and {surface!r} in alphabet; "
            f"got {sorted(props)}"
        )

    if not _all_entrapped_rescued(env):
        return (
            frozenset({_single_prop_assignment(entrapped, props)}),
            frozenset({_single_prop_assignment(surface, props)}) | walls_avoid,
        )

    return (
        frozenset({_single_prop_assignment(surface, props)}),
        walls_avoid,
    )


def gated_reach_avoid_for_features(
    env: Any,
    reach: FrozenSet[FrozenAssignment],
    avoid: FrozenSet[FrozenAssignment],
    propositions: set[str],
    *,
    agent_idx: int = 0,
    num_agents: int = 1,
) -> tuple[FrozenSet[FrozenAssignment], FrozenSet[FrozenAssignment]]:
    """Feature reach/avoid: hierarchical per-agent when team MA; else Büchi passthrough."""
    if should_use_ma_hierarchy(propositions, num_agents=num_agents):
        return hierarchical_reach_avoid_for_agent(env, agent_idx, set(propositions))
    return reach, avoid
