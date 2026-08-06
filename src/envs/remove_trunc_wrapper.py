from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType


class RemoveTruncWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        done = False
        if isinstance(terminated, dict):
            done = done or any(terminated.values())
        else:
            done = done or terminated
        if isinstance(truncated, dict):
            done = done or any(truncated.values())
        else:
            done = done or truncated
        return obs, reward, done, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> WrapperObsType:
        obs, _ = super().reset(seed=seed, options=options)
        return obs
