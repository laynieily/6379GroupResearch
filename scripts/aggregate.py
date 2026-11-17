#!/usr/bin/env python
"""Collect results/*/*/metrics.json into one summary table (mean/std over seeds).

    python scripts/aggregate.py                      # writes results/summary/summary.csv
    python scripts/aggregate.py --plot               # also plots test accuracy vs T
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["test_acc", "test_nll_T1", "test_ece_T1", "test_nll_trainT", "test_ece_trainT"]


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = [json.load(open(p)) for p in sorted(results_dir.glob("*/*/metrics.json"))]
    if not rows:
        raise SystemExit(f"no metrics.json found under {results_dir}/<dataset>/<run_id>/")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "lr_mode", "base_lr", "temperature"]
    g = df.groupby(keys)
    out = g.agg(n_runs=("seed", "count"), n_diverged=("diverged", "sum"),
                frac_reached_target=("epochs_to_target", lambda s: s.notna().mean()),
                epochs_to_target_mean=("epochs_to_target", "mean"))
    for m in METRICS:
        if m in df:
            out[f"{m}_mean"] = g[m].mean()
            out[f"{m}_std"] = g[m].std()
    return out.reset_index()


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset, d in summary.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for lr_mode, s in d.groupby("lr_mode"):
            s = s.sort_values("temperature")
            ax.errorbar(s["temperature"], s["test_acc_mean"], yerr=s["test_acc_std"],
                        marker="o", capsize=3, label=lr_mode)
        ax.set_xscale("log")
        ax.set_xlabel("temperature T")
        ax.set_ylabel("test accuracy")
        ax.set_title(dataset)
        ax.legend(title="LR condition")
        fig.tight_layout()
        fig.savefig(out_dir / f"acc_vs_temp_{dataset}.png", dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    results = Path(args.results)
    out_dir = results / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_runs(results))
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    if args.plot:
        plot(summary, out_dir)


if __name__ == "__main__":
    main()
