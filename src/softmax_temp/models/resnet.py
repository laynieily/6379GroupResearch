"""ResNets. Models output raw logits; temperature is applied outside (temp_softmax.py)."""
from __future__ import annotations

import torch.nn as nn
import torchvision


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 with the usual CIFAR stem (3x3 conv, no max-pool)."""
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def resnet50_imagenet(num_classes: int = 1000) -> nn.Module:
    return torchvision.models.resnet50(num_classes=num_classes)
