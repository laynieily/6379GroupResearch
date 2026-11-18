"""Dataset registry. Each builder returns (train_ds, val_ds, test_ds, info)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def split_indices(n: int, val_fraction: float, split_seed: int):
    """Fixed train/val split, independent of the run seed so all runs share it."""
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_fraction)
    return perm[n_val:], perm[:n_val]


def build_dataloaders(cfg: dict, seed: int):
    name = cfg["data"]["name"]
    if name == "cifar10":
        from .cifar10 import build
    elif name == "imagenet":
        from .imagenet import build
    elif name == "imdb":
        from .imdb import build
    elif name == "synthetic":
        from .synthetic import build
    else:
        raise ValueError(f"unknown dataset '{name}'")

    train_ds, val_ds, test_ds, info = build(cfg)

    tcfg = cfg["train"]
    g = torch.Generator().manual_seed(seed)  # seeds the training shuffle order
    kw = dict(
        batch_size=tcfg["batch_size"],
        num_workers=tcfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=info.get("collate_fn"),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return (train_loader, val_loader, test_loader), info
