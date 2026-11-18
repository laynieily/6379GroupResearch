"""ImageNet-1k (ILSVRC2012). IMPLEMENTED BUT UNTESTED.

Expects the standard layout (not downloaded by this repo):
    <data.root>/train/<wnid>/*.JPEG
    <data.root>/val/<wnid>/*.JPEG      (val images sorted into class folders)

Because the official ImageNet val set is the usual "test" set, we hold out a
fraction of the training set for model selection / curves and report final
numbers on the official val folder.
"""
from __future__ import annotations

from torch.utils.data import Subset

from . import split_indices


def build(cfg: dict):
    from torchvision import datasets, transforms

    d = cfg["data"]
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm]
        if d.get("augment", True)
        else [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
    )

    full_train = datasets.ImageFolder(f"{d['root']}/train", transform=train_tf)
    full_val = datasets.ImageFolder(f"{d['root']}/train", transform=eval_tf)
    test = datasets.ImageFolder(f"{d['root']}/val", transform=eval_tf)

    train_idx, val_idx = split_indices(len(full_train), d.get("val_fraction", 0.02), d.get("split_seed", 0))
    info = {"num_classes": 1000, "task": "image"}
    return Subset(full_train, train_idx), Subset(full_val, val_idx), test, info
