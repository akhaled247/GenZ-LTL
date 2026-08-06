"""Isolated SAR base-env factory (SpecRL fast-reset path vs legacy GenZ fork)."""

from __future__ import annotations

import gymnasium
from gymnasium.wrappers import FlattenObservation


def make_sar_base_env(
    name: str,
    render_mode: str | None = None,
    backend: str = "specrl",
    flat: bool = True,
    sar_ltl_ordering: bool = False,
) -> gymnasium.Env:
    """Build SAR underlying env before GenZ sequence / safety wrappers."""
    if backend == "specrl":
        import specbench  # noqa: F401 — registers SAR env IDs
        from specbench.envs.zones.zone_env import make_zone_env

        return make_zone_env(
            name,
            render_mode=render_mode,
            flat=flat,
            sar_ltl_ordering=sar_ltl_ordering,
        )

    if backend == "genz_local":
        import safety_gymnasium
        from envs.zones.safety_gym_wrapper import SafetyGymWrapper

        env = safety_gymnasium.make(name, render_mode=render_mode)
        env = SafetyGymWrapper(env)
        return FlattenObservation(env)

    raise ValueError(f"Unknown sar_env_backend: {backend}")
