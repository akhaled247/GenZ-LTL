"""Resolve training device strings (gpu, cuda:N) for GenZ trainers."""

from __future__ import annotations

import torch


def resolve_training_device(device: str) -> str:
    """Normalize device string and validate CUDA availability / index."""
    if device == "gpu":
        device = "cuda:0"

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA is not available."
        if ":" in device:
            index = int(device.split(":")[1])
            assert 0 <= index < torch.cuda.device_count(), (
                f"Invalid CUDA device {device}; "
                f"only {torch.cuda.device_count()} device(s) available."
            )
        else:
            device = "cuda:0"

    return device
