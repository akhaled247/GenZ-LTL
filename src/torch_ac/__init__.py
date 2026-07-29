from .utils import DictList, ParallelEnv
from .algos import PPO, RCO


def __getattr__(name: str):
    if name in ("MAIPPO", "MARCO"):
        from . import algos
        return getattr(algos, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DictList", "ParallelEnv", "PPO", "RCO", "MAIPPO", "MARCO"]
