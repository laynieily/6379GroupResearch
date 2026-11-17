"""Evaluation metrics: accuracy, NLL, and ECE.

NLL and ECE are reported at two temperatures:
  *_trainT : the temperature the model was trained with
  *_T1     : the raw logits (T = 1)
Accuracy is invariant to temperature (argmax does not change).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = torch.zeros(())
    n = targets.numel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.float().sum() / n * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(total)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> dict:
    out = {"acc": float(logits.argmax(1).eq(targets).float().mean())}
    for tag, temp in (("trainT", temperature), ("T1", 1.0)):
        log_p = F.log_softmax(logits / temp, dim=-1)
        out[f"nll_{tag}"] = float(F.nll_loss(log_p, targets))
        out[f"ece_{tag}"] = ece(log_p.exp(), targets)
    return out


@torch.no_grad()
def evaluate(model, loader, device, temperature: float) -> dict:
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_targets.append(y)
    return compute_metrics(torch.cat(all_logits), torch.cat(all_targets), temperature)
