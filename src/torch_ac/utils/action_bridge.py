"""GPU tensor → env action numpy bridge (optional pinned fast path)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class ActionBridge(Protocol):
    def to_env_actions(self, action: torch.Tensor) -> np.ndarray:
        ...


class DefaultActionBridge:
    """Current behavior: synchronous detach + cpu + numpy."""

    def to_env_actions(self, action: torch.Tensor) -> np.ndarray:
        return action.detach().cpu().numpy()


class FastActionBridge:
    """Pinned CPU buffer + non_blocking copy when CUDA is available."""

    def __init__(self, action_shape: tuple[int, ...], device: str):
        self._use_fast = device.startswith("cuda") and torch.cuda.is_available()
        if self._use_fast:
            self._buf = torch.empty(action_shape, dtype=torch.float32, pin_memory=True)
        else:
            self._buf = None

    def to_env_actions(self, action: torch.Tensor) -> np.ndarray:
        if self._buf is None:
            return action.detach().cpu().numpy()
        self._buf.copy_(action, non_blocking=True)
        return self._buf.numpy()


def make_action_bridge(
    device: str,
    action_shape: tuple[int, ...],
    fast: bool,
) -> ActionBridge:
    if fast:
        return FastActionBridge(action_shape, device)
    return DefaultActionBridge()
