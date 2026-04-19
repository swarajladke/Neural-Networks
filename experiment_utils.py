from __future__ import annotations

import torch


def metric_to_float(value) -> float:
    """Convert scalars or tensors to a stable Python float for reporting."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return float(value.mean().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def metric_to_str(value, precision: int = 4) -> str:
    return f"{metric_to_float(value):.{precision}f}"
