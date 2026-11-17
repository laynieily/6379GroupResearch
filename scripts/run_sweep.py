#!/usr/bin/env python
"""Expand the sweep grid (temperatures x seeds x lr_modes) into runs.

Each run is the exact same CLI call as a manual run:
    python -m softmax_temp.train --config ... --temp T --seed S --lr-mode M

Examples
    python scripts/run_sweep.py --config configs/cifar10_resnet.yaml --dry-run
    python scripts/run_sweep.py --config configs/cifar10_resnet.yaml
    # split across 4 cluster jobs (job k of 4):
    python scripts/run_sweep.py --config configs/cifar10_resnet.yaml --shard 0/4

Finished runs (metrics.json present) are skipped, so re-running resumes the sweep.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys

from softmax_temp.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    p.add_argument("--dry-run", action="store_true", help="print commands without running them")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                   help="dotted override passed to every run, e.g. --set train.epochs=2")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    sw = cfg["sweep"]
    jobs = list(itertools.product(sw["lr_modes"], sw["temperatures"], sw["seeds"]))
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} runs in this shard ({args.shard}) of grid "
          f"{len(sw['temperatures'])} temps x {len(sw['seeds'])} seeds x {len(sw['lr_modes'])} lr modes")

    for lr_mode, temp, seed in jobs:
        cmd = [sys.executable, "-m", "softmax_temp.train", "--config", args.config,
               "--temp", str(temp), "--seed", str(seed), "--lr-mode", lr_mode]
        for o in args.overrides:
            cmd += ["--set", o]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=False)  # keep going if one run fails


if __name__ == "__main__":
    main()
