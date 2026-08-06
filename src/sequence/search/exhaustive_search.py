from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
import numpy as np
from torch import nn

from ltl.automata import LDBA, LDBATransition, LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from sequence.search import SequenceSearch


class NoPathsException(Exception):
    """
    Raised when no paths are found during search.
    """
    pass


WALLS_PROP = "walls"
ANY_WALLS = "any_walls"
_WALLS_PROPS = frozenset({WALLS_PROP, ANY_WALLS})


def _is_walls_prop(name: str) -> bool:
    return name in _WALLS_PROPS


def _is_surface_prop(name: str) -> bool:
    return name == "all_surface" or name == "any_surface" or name.startswith("surface_")


def _is_entrapped_prop(name: str) -> bool:
    return name == "all_entrapped" or name.startswith("entrapped_")


def _assignment_true_props(assignment: FrozenAssignment) -> set[str]:
    return set(assignment.get_true_propositions())


def _reach_true_props(reach) -> set[str]:
    if reach == LDBASequence.EPSILON:
        return set()
    props: set[str] = set()
    for assignment in reach:
        props |= _assignment_true_props(assignment)
    return props


def _avoid_true_props(avoid: frozenset[FrozenAssignment]) -> set[str]:
    props: set[str] = set()
    for assignment in avoid:
        props |= _assignment_true_props(assignment)
    return props


def _expand_avoid_singles(
    avoid: frozenset[FrozenAssignment],
    propositions,
) -> frozenset[FrozenAssignment]:
    props = set(propositions) if not isinstance(propositions, set) else propositions
    cleaned: list[FrozenAssignment] = []
    seen: set[str] = set()
    for a in avoid:
        for p in _assignment_true_props(a):
            if _is_walls_prop(p):
                continue
            if p not in seen:
                seen.add(p)
                cleaned.append(Assignment.single_proposition(p, props).to_frozen())
    if ANY_WALLS in props:
        cleaned.append(Assignment.single_proposition(ANY_WALLS, props).to_frozen())
    elif WALLS_PROP in props:
        cleaned.append(Assignment.single_proposition(WALLS_PROP, props).to_frozen())
    return frozenset(cleaned)


def sanitize_reach_avoid_walls(
    reach_assignment: FrozenAssignment,
    avoid: frozenset[FrozenAssignment],
    propositions,
) -> tuple[frozenset[FrozenAssignment], frozenset[FrozenAssignment]] | None:
    """Strip ``walls`` from reach; always place it in avoid when in the alphabet.

    Also drops any reach prop that appears in avoid (e.g. surface in both).
    """
    props = set(propositions) if not isinstance(propositions, set) else propositions
    new_avoid = avoid # _expand_avoid_singles(avoid, props)
    avoid_props = _avoid_true_props(new_avoid)
    reach_props = [
        name for name, truth in reach_assignment
        if truth and not _is_walls_prop(name) and name not in avoid_props
    ]
    if not reach_props:
        return None
    new_reach = frozenset(
        Assignment.single_proposition(p, props).to_frozen() for p in reach_props
    )
    return new_reach, new_avoid


def strip_walls_from_reach_set(
    reach: frozenset[FrozenAssignment],
    avoid: frozenset[FrozenAssignment],
    propositions,
) -> tuple[frozenset[FrozenAssignment], frozenset[FrozenAssignment]] | None:
    """Sanitize a frozenset of reach assignments (one Büchi stage)."""
    props = set(propositions) if not isinstance(propositions, set) else propositions
    new_avoid = avoid #_expand_avoid_singles(avoid, props)
    avoid_props = _avoid_true_props(new_avoid)
    reach_props: list[str] = []
    for assignment in reach:
        for name, truth in assignment:
            if (
                truth
                and not _is_walls_prop(name)
                and name not in avoid_props
                and name not in reach_props
            ):
                reach_props.append(name)
    if not reach_props:
        return None
    new_reach = frozenset(
        Assignment.single_proposition(p, props).to_frozen() for p in reach_props
    )
    return new_reach, new_avoid


def sequence_has_reach_avoid_conflict(seq: LDBASequence) -> bool:
    """True if any stage puts the same prop in both reach and avoid."""
    for reach, avoid in seq:
        if reach == LDBASequence.EPSILON:
            continue
        if _reach_true_props(reach) & _avoid_true_props(avoid):
            return True
    return False


def sequence_has_surface_before_entrapped(seq: LDBASequence) -> bool:
    """True if a surface reach stage appears before an entrapped reach stage.

    Matches ``!surface U entrapped``. Allows surface-only sequences (valid after
    Until is already satisfied in the LDBA state).
    """
    surface_idxs: list[int] = []
    entrapped_idxs: list[int] = []
    for i, (reach, _avoid) in enumerate(seq):
        props = _reach_true_props(reach)
        if any(_is_surface_prop(p) for p in props):
            surface_idxs.append(i)
        if any(_is_entrapped_prop(p) for p in props):
            entrapped_idxs.append(i)
    if surface_idxs and entrapped_idxs and min(surface_idxs) < min(entrapped_idxs):
        return True
    return False


def sanitize_ldba_sequence_walls(seq: LDBASequence, propositions) -> LDBASequence | None:
    """Apply walls/reach sanitization to every stage; drop invalid Until order."""
    stages: list[tuple[frozenset[FrozenAssignment], frozenset[FrozenAssignment]]] = []
    for reach, avoid in seq:
        if reach == LDBASequence.EPSILON:
            stages.append((reach, avoid))
            continue
        sanitized = strip_walls_from_reach_set(reach, avoid, propositions)
        if sanitized is None:
            return None
        stages.append(sanitized)
    if not stages:
        return None
    out = LDBASequence(stages)
    if sequence_has_reach_avoid_conflict(out):
        return None
    if sequence_has_surface_before_entrapped(out):
        return None
    return out


@dataclass
class Path:
    reach_avoid: list[tuple[LDBATransition, set[LDBATransition]]]
    loop_index: int

    def __len__(self):
        return len(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Path(self.reach_avoid[item], self.loop_index)
        return self.reach_avoid[item]

    def __str__(self):
        p = [(reach.source, {a.target for a in avoid}) for reach, avoid in self[:self.loop_index]]
        loop = [(reach.source, {a.target for a in avoid}) for reach, avoid in self[self.loop_index:]]
        return str(p + loop * 3)

    def prepend(self, reach: LDBATransition, avoid: set[LDBATransition]) -> 'Path':
        return Path([(reach, avoid)] + self.reach_avoid, self.loop_index)

    def to_sequence(self, num_loops: int) -> LDBASequence:
        seq = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[:self.loop_index] if not r.is_epsilon()]
        loop = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[self.loop_index:] if not r.is_epsilon()]
        # seq = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[:self.loop_index]]
        # loop = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[self.loop_index:]]
        seq = seq + loop * num_loops
        return LDBASequence(seq)

    @staticmethod
    def reach_avoid_to_assignments(reach: LDBATransition, avoid: set[LDBATransition]) -> tuple[frozenset, frozenset]:
        avoid = [a.valid_assignments for a in avoid]
        avoid = set() if not avoid else set.union(*avoid)
        if reach.is_epsilon():
            new_reach = LDBASequence.EPSILON
        else:
            new_reach = frozenset(reach.feasible_assignments)
            assert new_reach  # at least one feasible assignment
        return new_reach, frozenset(avoid)


class ExhaustiveSearch(SequenceSearch):
    def __init__(self, model: nn.Module, propositions, num_loops: int, value_threshold: float = 0.4):
        super().__init__(model, propositions)
        self.num_loops = num_loops
        self.value_threshold = value_threshold

    def __call__(self, ldba: LDBA, ldba_state: int, obs=None) -> LDBASequence:
        seqs = self.all_sequences(ldba, ldba_state, obs, self.num_loops)
        return max(seqs, key=lambda s: self.get_value(s, obs))

    def all_sequences(self, ldba: LDBA, ldba_state: int, obs=None, num_loops=1) -> list[LDBASequence]:
        num_loops = 0 if ldba.is_finite_specification() else num_loops
        paths = self.dfs(ldba, ldba_state, [], {}, None, obs, num_loops)
        return [path.to_sequence(num_loops) for path in paths]

    def dfs(self, ldba: LDBA, state: int, current_path: list[LDBATransition], state_to_path_index: dict[int, int],
            accepting_transition: Optional[LDBATransition], obs=None, num_loops=1) -> list[Path]:
        """
        Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
        """
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        transition_to_max_value = {}
        for transition in ldba.state_to_transitions[state]:
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if updated_accepting_transition in current_path[state_to_path_index[transition.target]:]:
                        # found accepting cycle
                        path = Path(reach_avoid=[], loop_index=state_to_path_index[transition.target])
                        future_paths = [path]
                    else:
                        # found non-accepting cycle
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = self.dfs(ldba, transition.target, current_path, state_to_path_index,
                                            updated_accepting_transition, obs)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                    else:
                        if obs is not None:
                            future_seqs = [fp.to_sequence(num_loops) for fp in future_paths]
                            max_value = max([self.get_value(s, obs) for s in future_seqs])
                            transition_to_max_value[transition] = max_value
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = ExhaustiveSearch.prune_paths(paths)
        # for path in paths:
        #     path[0][1].update(neg_transitions)  # now we update the negative transitions
        for path in paths:
            path[0][1].update(neg_transitions)  # now we update the negative transitions
            if obs is None:
                continue
            chosen_seq = path.to_sequence(num_loops)
            chosen_value = self.get_value(chosen_seq, obs)
            for transition in ldba.state_to_transitions[state]:
                if transition in neg_transitions or transition.source == transition.target or transition == path[0][0]:
                    continue
                if transition not in transition_to_max_value:
                    continue
                alternative_value = transition_to_max_value[transition]
                if chosen_value - alternative_value > self.value_threshold:
                    path[0][1].add(transition)
        return paths

    @staticmethod
    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        paths = [paths[i] for i in range(len(paths)) if i not in to_remove]
        return paths

    @staticmethod
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        path1 = [t[0].valid_assignments for t in path1]
        path2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        found = False
        for p in path1:
            if p.issubset(path2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(path2):
                    found = True
                    break
        return found


class ExhaustiveSearchSafety(SequenceSearch):
    def __init__(self, env, model: nn.Module, propositions, num_loops: int, device=None):
        super().__init__(model, propositions, device=device)
        self.env = env
        self.num_loops = num_loops
        from envs.seq_wrapper import sar_task
        self.num_agents = getattr(sar_task(env), "agent_num", 1)

    def __call__(self, ldba: LDBA, ldba_states: List[int], obs=None) -> LDBASequence:
        seqs = self.all_sequences(ldba, ldba_states, obs, self.num_loops)

        processed_seqs = []
        for seq in seqs:
            reach_list, avoid = seq[0]
            suffix = seq[1:]

            # Eliminate avoid assignments that are subsets of another
            # E.g., if we have green, we can eliminate green & magenta
            avoid_assignments = []
            avoid_sets = []
            # Sort by length of true propositions: start from smallest (most general) assignments
            for a, s in sorted([(a, a.get_true_propositions()) for a in avoid], key=lambda x: len(x[1])):
                if not any(other <= s for other in avoid_sets):
                    avoid_sets.append(s)
                    avoid_assignments.append(a)
            new_avoid = frozenset(avoid_assignments)

            for reach in reach_list:
                true_props = reach.get_true_propositions()
                # Check conflicts with avoid set (ignore walls co-activation on reach)
                true_wo_walls = true_props - _WALLS_PROPS
                if not true_wo_walls:
                    continue
                if any(avoid_set <= true_wo_walls for avoid_set in avoid_sets):
                    continue
                sanitized = sanitize_reach_avoid_walls(reach, new_avoid, self.propositions)
                if sanitized is None:
                    continue
                new_reach, walls_avoid = sanitized
                # Keep suffix but strip walls from every later reach stage too.
                head = LDBASequence([(new_reach, walls_avoid)] + list(suffix))
                full = sanitize_ldba_sequence_walls(head, self.propositions)
                if full is None:
                    continue
                processed_seqs.append(full)

        if not processed_seqs:
            raise NoPathsException()

        return max(processed_seqs, key=lambda s: self._score_subgoal([s[0]], obs))

    def _score_subgoal(self, subgoal, obs) -> float:
        if hasattr(self.model, "cost_critic"):
            return self.get_value_safety(subgoal, obs)
        return self.get_value_sar(subgoal, obs)

    def all_sequences(self, ldba: LDBA, ldba_states: List[int], obs=None, num_loops=1) -> List[LDBASequence]:
        num_loops = 0 if ldba.is_finite_specification() else num_loops
        return [
            path.to_sequence(num_loops)
            for ldba_state in ldba_states
            for path in self.dfs(ldba, ldba_state, [], {}, None, obs, num_loops)
        ]

    def dfs(self, ldba: LDBA, state: int, current_path: list[LDBATransition], state_to_path_index: dict[int, int],
            accepting_transition: Optional[LDBATransition], obs=None, num_loops=1) -> list[Path]:
        """
        Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
        """
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        # transition_to_max_value = {}
        for transition in ldba.state_to_transitions[state]:
            if not transition.is_feasible():
                # If all valid assignments are marked as unfeasible, skip this transition.
                continue
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if updated_accepting_transition in current_path[state_to_path_index[transition.target]:]:
                        # found accepting cycle
                        path = Path(reach_avoid=[], loop_index=state_to_path_index[transition.target])
                        future_paths = [path]
                    else:
                        # found non-accepting cycle
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = self.dfs(ldba, transition.target, current_path, state_to_path_index,
                                            updated_accepting_transition, obs)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = ExhaustiveSearch.prune_paths(paths)
        for path in paths:
            path[0][1].update(neg_transitions)  # now we update the negative transitions
        return paths

    @staticmethod
    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        paths = [paths[i] for i in range(len(paths)) if i not in to_remove]
        return paths

    @staticmethod
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        path1 = [t[0].valid_assignments for t in path1]
        path2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        found = False
        for p in path1:
            if p.issubset(path2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(path2):
                    found = True
                    break
        return found
