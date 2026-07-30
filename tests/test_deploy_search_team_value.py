"""Tests for Büchi search team-value scoring."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from sequence.search.sequence_search import SequenceSearch


class _DummySearch(SequenceSearch):
    def __call__(self, ldba, ldba_states, obs=None):
        raise NotImplementedError

    def __init__(self):
        super().__init__(MagicMock(), set())
        self.env = MagicMock()
        self.model = MagicMock()
        self.num_agents = 2

    def _value_safety_for_agent(self, seq, obs, agent_idx: int) -> float:
        return float(agent_idx + 1)


def test_get_value_safety_uses_min_across_agents():
    search = _DummySearch()
    val = search.get_value_safety(("seq",), {})
    assert val == 1.0
