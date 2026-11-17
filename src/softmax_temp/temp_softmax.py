"""The one place where softmax temperature is applied.

Every model / dataset goes through this module so the temperature is handled
identically everywhere. Convention: p = softmax(logits / T), i.e. T < 1 sharpens
the distribution and T > 1 flattens it (inverse temperature beta = 1 / T).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureSoftmax(nn.Module):
    """Softmax with a fixed temperature. T = 1 is exactly the plain softmax."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def forward(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.softmax(self.scale(logits), dim=dim)

    def log_probs(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.log_softmax(self.scale(logits), dim=dim)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on temperature-scaled logits (training-time temperature)."""
        return F.cross_entropy(self.scale(logits), targets)


def scaled_lr(base_lr: float, temperature: float, lr_mode: str, optimizer: str = "sgd") -> float:
    """Learning rate for a given temperature.

    Dividing logits by T rescales the gradient w.r.t. the logits by ~1/T, so with
    plain SGD, changing T also changes the effective step size. To separate the
    effect of temperature from the effect of step size we run two LR conditions:

      fixed        -> lr = base_lr for every T
      compensated  -> lr = base_lr * T   (first-order cancellation of the 1/T
                      gradient scale; SGD only)

    Adaptive optimizers (Adam/AdamW) normalise gradient magnitude, so the
    1/T scale largely cancels on its own and `compensated` falls back to
    `fixed` for them. `compensated` is a first-order heuristic, not a tuned
    optimum: for a true tuned-LR condition, sweep `optim.lr` per temperature
    (see docs/decisions.md).
    """
    if lr_mode == "fixed":
        return base_lr
    if lr_mode == "compensated":
        return base_lr * temperature if optimizer.lower() == "sgd" else base_lr
    raise ValueError(f"unknown lr_mode '{lr_mode}' (expected 'fixed' or 'compensated')")
