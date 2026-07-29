from .ppo import PPO
from .rco import RCO


def __getattr__(name: str):
    if name == "MAIPPO":
        from .ma_ippo import MAIPPO
        return MAIPPO
    if name == "MARCO":
        from .ma_rco import MARCO
        return MARCO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PPO", "RCO", "MAIPPO", "MARCO"]
