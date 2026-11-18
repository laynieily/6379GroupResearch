from __future__ import annotations

from torch.utils.data import Subset

from . import split_indices

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build(cfg: dict):
    from torchvision import datasets, transforms

    d = cfg["data"]
    norm = transforms.Normalize(MEAN, STD)
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    if d.get("augment", True):
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
    else:
        train_tf = eval_tf

    # Two views of the same training set so the val split gets the eval transform.
    full_train = datasets.CIFAR10(d["root"], train=True, download=d.get("download", True), transform=train_tf)
    full_val = datasets.CIFAR10(d["root"], train=True, download=False, transform=eval_tf)
    test = datasets.CIFAR10(d["root"], train=False, download=d.get("download", True), transform=eval_tf)

    train_idx, val_idx = split_indices(len(full_train), d.get("val_fraction", 0.1), d.get("split_seed", 0))
    info = {"num_classes": 10, "task": "image"}
    return Subset(full_train, train_idx), Subset(full_val, val_idx), test, info
