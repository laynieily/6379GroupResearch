# 6379GroupResearch
Studying how softmax temperature affects learning efficiency, accuracy, and calibration in ResNets and GRUs on CIFAR-10, ImageNet, and IMDB.

A controlled, reproducible study of how softmax temperature affects learning efficiency,
accuracy, and calibration in neural networks. We train ResNets on CIFAR-10 and ImageNet and
GRUs on IMDB across a range of temperatures, with `p = softmax(logits / T)`.

Because dividing the logits by `T` rescales gradients (it can act like a learning-rate change),
every sweep runs under two LR conditions, `fixed` and `compensated`, to separate the effect of
temperature from the effect of step size. See [`docs/decisions.md`](docs/decisions.md).

## Status

| Piece | State |
|---|---|
| Temperature layer, metrics (acc / NLL / ECE), config system | Done, unit-tested |
| Training loop, sweep runner, aggregation + plot | Done, smoke-tested on synthetic data |
| CIFAR-10 + ResNet-18 | Implemented; **run it once with a real download to confirm** |
| ImageNet + ResNet-50 | Implemented, **untested** (needs a local ImageNet copy) |
| IMDB + GRU | Model done; **data loader is a stub** (`src/softmax_temp/data/imdb.py`) |
| Team decisions (grid, optimizer, epochs, ...) | **Placeholders**: see `docs/decisions.md` |

## Setup

Requires **Python 3.10+** and git. All commands are run from the repo root.

### 1. Clone

```bash
git clone <repo-url>
cd 6379GroupResearch
```

### 2. Create the virtual environment (first time only)

```bash
python -m venv .venv
```

(On macOS/Linux you may need `python3` instead of `python`.)

### 3. Activate it (every new terminal)

| Shell | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| macOS / Linux | `source .venv/bin/activate` |

Your prompt should now start with `(.venv)`. Run `deactivate` to leave it.

> **PowerShell says "running scripts is disabled"?** Run this once, then activate again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

> **VS Code:** `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv`. New terminals
> will then activate it automatically.

### 4. Install the project and dependencies (first time only)

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs PyTorch, torchvision, numpy, pandas, pyyaml, matplotlib, and pytest, and installs
this repo in editable mode so code changes take effect without reinstalling.

> **GPU:** the default `pip install torch` may give you a CPU-only build on Windows. For CUDA,
> install torch first using the command from <https://pytorch.org/get-started/locally/>, then
> run `pip install -e ".[dev]"`. Check with
> `python -c "import torch; print(torch.cuda.is_available())"`.

### 5. Verify

```bash
pytest -q                      # ~15 s, no downloads needed
```

Datasets are downloaded into `data/` (gitignored) the first time a real run needs them.

## Run

Make sure the virtual environment is activated first (step 3 above).

```bash
# 0) sanity check the whole pipeline in seconds (synthetic data, no download)
python -m softmax_temp.train --config configs/smoke.yaml --temp 1.0 --seed 0

# 1) one real run
python -m softmax_temp.train --config configs/cifar10_resnet.yaml \
    --temp 0.5 --seed 1 --lr-mode compensated

# 2) the full sweep (temperatures x seeds x LR conditions, from the config's `sweep:` block)
python scripts/run_sweep.py --config configs/cifar10_resnet.yaml --dry-run   # preview
python scripts/run_sweep.py --config configs/cifar10_resnet.yaml
python scripts/run_sweep.py --config configs/cifar10_resnet.yaml --shard 0/4 # job 0 of 4 (cluster)

# 3) aggregate
python scripts/aggregate.py --plot      # -> results/summary/summary.csv + plots
```

Any config value can be overridden from the command line: `--set optim.lr=0.05 --set train.epochs=50`.
Finished runs are skipped, so re-running a sweep resumes where it stopped.

## Layout

```
configs/        base.yaml + one config per dataset (+ smoke.yaml)
src/softmax_temp/
  temp_softmax.py   the ONE place temperature is applied (+ the LR conditions)
  train.py          single entry point: config in, results folder out
  evaluate.py       accuracy, NLL, ECE (at train temperature and at T=1)
  data/  models/    one module per dataset / architecture
scripts/        run_sweep.py (grid -> runs), aggregate.py (runs -> summary table)
results/        raw per-run outputs (gitignored); results/summary/ is committed
docs/           decisions.md: the choices the team still has to settle
tests/          unit tests + an end-to-end smoke test
```

Each run writes `results/<dataset>/<run_id>/` containing `config.json` (resolved config),
`history.csv` (per-epoch curves), and `metrics.json` (final test metrics, epochs-to-target,
divergence flag, git commit).

## Metrics

- **Accuracy**: invariant to temperature (argmax does not change), so any accuracy effect is
  an effect on *training*.
- **Epochs to target**: first epoch at which validation accuracy reaches `target_acc`.
- **NLL / ECE**: reported at the training temperature (`*_trainT`) and at raw logits (`*_T1`).
- The test set is evaluated once per run, at the end.

## Related work

Agarwala et al., *Temperature Check*; Masarczyk et al., *Unpacking Softmax*; Guo et al.,
*On Calibration of Modern Neural Networks* (post-hoc temperature scaling).
