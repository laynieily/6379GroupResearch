"""Tiny learnable synthetic image dataset for smoke tests (no download needed)."""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def _make(n: int, means: torch.Tensor, gen: torch.Generator):
    k, shape = means.shape[0], means.shape[1:]
    y = torch.randint(0, k, (n,), generator=gen)
    x = 0.5 * torch.randn(n, *shape, generator=gen) + means[y]
    return TensorDataset(x, y)


def build(cfg: dict):
    d = cfg["data"]
    gen = torch.Generator().manual_seed(d.get("split_seed", 0))
    shape, k = (3, 32, 32), 10
    means = torch.randn(k, *shape, generator=gen)  # class prototypes shared by all splits
    train = _make(d.get("n_train", 512), means, gen)
    val = _make(d.get("n_val", 128), means, gen)
    test = _make(d.get("n_test", 128), means, gen)
    return train, val, test, {"num_classes": k, "task": "image"}
