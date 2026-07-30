"""Preflight checks for SAR paper-protocol env stacks."""
from __future__ import annotations

import gymnasium

from specbench.envs.zones.safety_gym_wrapper_sar_ltl import SafetyGymWrapperMASARLTL
from specbench.envs.zones.safety_gym_wrapper_sar_wc import SafetyGymWrapperMASARWC


def find_sar_wrapper(env: gymnasium.Env):
    cur: gymnasium.Env | None = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (SafetyGymWrapperMASARWC, SafetyGymWrapperMASARLTL)):
            return cur
        cur = getattr(cur, "env", None)
    return None


def assert_sar_wc_paper_protocol(env: gymnasium.Env) -> None:
    """Paper §5.3: SAR WC wrapper (collision ignored for cost/termination)."""
    wrapper = find_sar_wrapper(env)
    if wrapper is None:
        raise RuntimeError(
            "SAR deploy env missing SafetyGymWrapperMASARWC — "
            "use PointLTL*MASAR*WC-v0 with sar_ltl_ordering=False"
        )
    if isinstance(wrapper, SafetyGymWrapperMASARLTL):
        raise RuntimeError(
            "SAR deploy expected WC wrapper (sar_ltl_ordering=False); "
            "got SafetyGymWrapperMASARLTL"
        )
