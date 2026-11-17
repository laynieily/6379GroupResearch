"""YAML config loading with single-level inheritance and dotted CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A `base: other.yaml` key (relative path) is merged underneath.

    `overrides` are strings like "optim.lr=0.05" (values parsed as YAML).
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(load_config(path.parent / base), cfg)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got '{item}'")
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg
