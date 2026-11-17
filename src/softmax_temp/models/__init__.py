from __future__ import annotations


def build_model(cfg: dict, info: dict):
    name = cfg["model"]["name"]
    if name == "resnet18_cifar":
        from .resnet import resnet18_cifar

        return resnet18_cifar(info["num_classes"])
    if name == "resnet50":
        from .resnet import resnet50_imagenet

        return resnet50_imagenet(info["num_classes"])
    if name == "gru":
        from .gru import GRUClassifier

        m = cfg["model"]
        return GRUClassifier(
            vocab_size=info["vocab_size"],
            num_classes=info["num_classes"],
            embed_dim=m.get("embed_dim", 128),
            hidden_dim=m.get("hidden_dim", 256),
            num_layers=m.get("num_layers", 1),
            dropout=m.get("dropout", 0.3),
            pad_idx=info.get("pad_idx", 0),
        )
    raise ValueError(f"unknown model '{name}'")
