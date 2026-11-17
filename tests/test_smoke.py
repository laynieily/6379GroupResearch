"""End-to-end check on synthetic data: train -> outputs on disk -> skip on rerun."""
import json
from pathlib import Path

from softmax_temp.config import load_config
from softmax_temp.train import run

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_smoke_run_writes_outputs_and_resumes(tmp_path):
    cfg = load_config(CONFIG, [f"output.dir={tmp_path}", "temperature=2.0", "lr_mode=compensated", "seed=1"])
    metrics = run(cfg)
    assert metrics is not None and not metrics["diverged"]
    assert metrics["effective_lr"] == cfg["optim"]["lr"] * 2.0

    run_dirs = list((tmp_path / "synthetic").iterdir())
    assert len(run_dirs) == 1
    for name in ("config.json", "history.csv", "metrics.json"):
        assert (run_dirs[0] / name).exists()
    assert 0.0 <= json.load(open(run_dirs[0] / "metrics.json"))["test_acc"] <= 1.0

    assert run(cfg) is None  # finished runs are skipped
