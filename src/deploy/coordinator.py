"""Shared-policy MA coordinator: one Büchi search, per-agent SAR features."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from deploy.feature_recipe import sar_preprocess_for_deploy
from deploy.ma_phase_gating import gated_reach_avoid_for_features
from ltl.logic import Assignment
from sequence.search import SequenceSearch
from sequence.search.exhaustive_search import strip_walls_from_reach_set


class MultiAgentSARCoordinator:
    """One Büchi coordinator + per-agent goal-conditioned features (paper §5.3)."""

    def __init__(
        self,
        env: Any,
        model: Any,
        search: SequenceSearch,
        propositions: set[str],
        num_agents: int,
        verbose: bool = False,
        device = None,
    ):
        from model.agent import Agent

        self.env = env
        self.model = model
        self.search = search
        self.propositions = propositions
        self.num_agents = num_agents
        self.verbose = verbose
        self.sequence = None
        self.current_goal_steps = 0
        self.timeout = float('inf')
        self.last_reach = None
        self.last_avoid = None
        dev = device if device is not None else next(model.parameters()).device
        self._forward_agent = Agent(env, model, search, propositions, verbose=verbose, device=dev)

    def reset(self) -> None:
        self.sequence = None
        self.current_goal_steps = 0
        self.last_reach = None
        self.last_avoid = None
        self._forward_agent.reset()

    def get_action(self, obs, info, deterministic: bool = False) -> dict[str, np.ndarray]:
        if "ldba_state_changed" in info or self.sequence is None:
            prev_seq = self.sequence
            self.sequence = self.search(obs["ldba"], obs["ldba_states"], obs)
            if self.sequence != prev_seq:
                self.current_goal_steps = 0
            if self.verbose:
                print(f"Selected sequence: {self.sequence}")
        else:
            self.current_goal_steps += 1
            if self.current_goal_steps >= self.timeout:
                unfeasible_states = [
                    s
                    for s, accepting in zip(obs["ldba_states"], obs["ldba_states_accepting"])
                    if not accepting
                ]
                if unfeasible_states:
                    true_props = set()
                    for a in self.sequence[0][0]:
                        true_props = true_props.union(a.get_true_propositions())
                    reach_assignment = Assignment.where(
                        *true_props, propositions=obs["ldba"].propositions,
                    ).to_frozen()
                    obs["ldba"].mark_unfeasible(unfeasible_states, reach_assignment)
                    prev_seq = self.sequence
                    self.sequence = self.search(obs["ldba"], obs["ldba_states"], obs)
                    assert self.sequence != prev_seq
                self.current_goal_steps = 0

        assert self.sequence is not None
        reach, avoid = self.sequence[0]
        reach, avoid = gated_reach_avoid_for_features(
            self.env, reach, avoid, self.propositions,
        )
        # Belt-and-suspenders: never feed walls as a reach lidar target.
        stripped = strip_walls_from_reach_set(reach, avoid, self.propositions)
        if stripped is not None:
            reach, avoid = stripped
        self.last_reach = reach
        self.last_avoid = avoid
        if self.verbose:
            print(f"Feature reach/avoid: {reach} | {avoid}")
            reach_names = sorted({p for a in reach for p in a.get_true_propositions()})
            if "walls" in reach_names:
                print("ERROR: walls still in feature reach after sanitize — bug")
        actions: dict[str, np.ndarray] = {}
        for agent_idx in range(self.num_agents):
            obs_i = copy.deepcopy(obs)
            obs_i["goal"] = self.sequence
            obs_i["features"] = sar_preprocess_for_deploy(
                self.env, self.model, reach, avoid, agent_idx=agent_idx,
            )
            self._forward_agent.agent_idx = agent_idx
            action = self._forward_agent.forward(obs_i, deterministic).flatten()
            actions[f"agent_{agent_idx}"] = action
        return actions


# Backwards-compatible alias
MultiAgentSARAgent = MultiAgentSARCoordinator
