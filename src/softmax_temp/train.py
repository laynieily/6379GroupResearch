"""Single entry point: one config in, one results folder out.

    python -m softmax_temp.train --config configs/cifar10_resnet.yaml \
        --temp 0.5 --seed 1 --lr-mode compensated

Outputs to  <output.dir>/<dataset>/<run_id>/ :
    config.json    resolved config (incl. effective lr)
    history.csv    per-epoch train/val metrics
    metrics.json   final summary (test metrics, epochs-to-target, git hash, ...)
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from .config import load_config
from .data import build_dataloaders
from .evaluate import evaluate
from .models import build_model
from .temp_softmax import TemperatureSoftmax, scaled_lr
from .utils import get_device, git_hash, make_run_id, set_seed, write_json


def build_optimizer(model, cfg: dict, lr: float):
    o = cfg["optim"]
    name = o["name"].lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=o.get("momentum", 0.9),
            nesterov=o.get("nesterov", True),
            weight_decay=o.get("weight_decay", 5e-4),
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=o.get("weight_decay", 1e-2))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=o.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer '{name}'")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Linear warmup then cosine (or constant) decay, stepped once per iteration."""
    t = cfg["train"]
    total = t["epochs"] * steps_per_epoch
    warmup = int(t.get("warmup_epochs", 0) * steps_per_epoch)
    kind = cfg.get("schedule", "cosine")

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run(cfg: dict) -> dict | None:
    seed = cfg["seed"]
    temperature = float(cfg["temperature"])
    lr_mode = cfg.get("lr_mode", "fixed")
    base_lr = float(cfg["optim"]["lr"])
    lr = scaled_lr(base_lr, temperature, lr_mode, cfg["optim"]["name"])
    dataset = cfg["data"]["name"]

    run_id = make_run_id(temperature, seed, lr_mode, base_lr)
    run_dir = Path(cfg["output"]["dir"]) / dataset / run_id
    if (run_dir / "metrics.json").exists() and not cfg["output"].get("overwrite", False):
        print(f"[skip] {run_dir} already has metrics.json")
        return None

    set_seed(seed, deterministic=cfg.get("deterministic", False))
    device = get_device()
    (train_loader, val_loader, test_loader), info = build_dataloaders(cfg, seed)
    model = build_model(cfg, info).to(device)
    tsm = TemperatureSoftmax(temperature)
    optimizer = build_optimizer(model, cfg, lr)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    resolved = {**cfg, "effective_lr": lr, "run_id": run_id, "device": str(device)}
    write_json(resolved, run_dir / "config.json")
    print(f"[run] {dataset} {run_id} | T={temperature:g} lr={lr:g} ({lr_mode}) on {device}")

    tcfg = cfg["train"]
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    target = cfg.get("target_acc")
    history, epochs_to_target, diverged = [], None, False
    t0 = time.time()

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss = tsm.loss(logits.float(), y)
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        if diverged:
            print(f"[diverged] non-finite loss at epoch {epoch}")
            break

        val = evaluate(model, val_loader, device, temperature)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(1, seen),
            "train_acc": correct / max(1, seen),
            **{f"val_{k}": v for k, v in val.items()},
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        if epochs_to_target is None and target is not None and val["acc"] >= target:
            epochs_to_target = epoch
        print(
            f"  epoch {epoch:>3}/{tcfg['epochs']} loss {row['train_loss']:.4f} "
            f"train_acc {row['train_acc']:.3f} val_acc {val['acc']:.3f} "
            f"val_nll(T1) {val['nll_T1']:.3f} val_ece(T1) {val['ece_T1']:.3f}"
        )

    if history:
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    metrics = {
        "run_id": run_id,
        "dataset": dataset,
        "model": cfg["model"]["name"],
        "temperature": temperature,
        "seed": seed,
        "lr_mode": lr_mode,
        "base_lr": base_lr,
        "effective_lr": lr,
        "optimizer": cfg["optim"]["name"],
        "epochs_run": len(history),
        "diverged": diverged,
        "target_acc": target,
        "epochs_to_target": epochs_to_target,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "best_val_acc": max((h["val_acc"] for h in history), default=None),
        "runtime_s": time.time() - t0,
        "git_hash": git_hash(),
    }
    if not diverged:
        # Test set is touched exactly once, at the end of training.
        test = evaluate(model, test_loader, device, temperature)
        metrics.update({f"test_{k}": v for k, v in test.items()})
        print(f"[done] test_acc {test['acc']:.4f} nll(T1) {test['nll_T1']:.4f} ece(T1) {test['ece_T1']:.4f}")
    write_json(metrics, run_dir / "metrics.json")
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--temp", type=float, help="softmax temperature T (overrides config)")
    p.add_argument("--seed", type=int)
    p.add_argument("--lr-mode", choices=["fixed", "compensated"])
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                   help="dotted override, e.g. --set optim.lr=0.05 --set train.epochs=2")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = list(args.overrides)
    if args.temp is not None:
        overrides.append(f"temperature={args.temp}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.lr_mode is not None:
        overrides.append(f"lr_mode={args.lr_mode}")
    run(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
