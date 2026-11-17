# Design decisions

Fill this in as the team settles each item so the repo matches what you told people.
"Current default" is the placeholder in `configs/`, **not** an agreed choice.

| Item | Current default (placeholder) | Decision | Owner / date |
|---|---|---|---|
| Temperature set (the "grid") | `[0.1, 0.25, 0.5, 1, 2, 4, 10]`, log-spaced, includes T=1 | TBD | |
| Seeds | `[0, 1, 2]` | TBD | |
| LR conditions | `fixed` and `compensated` (lr x T, SGD only) | TBD | |
| Architecture | ResNet-18 (CIFAR stem), ResNet-50 (ImageNet), 1-layer GRU (IMDB) | TBD | |
| Optimizer | SGD + Nesterov momentum 0.9, wd 5e-4 (AdamW for GRU) | TBD | |
| Epochs | 30 CIFAR-10 / 90 ImageNet / 10 IMDB | TBD | |
| Data augmentation | Random crop (pad 4) + flip for CIFAR-10; random-resized-crop for ImageNet; none for IMDB | TBD | |
| LR schedule | 1 warmup epoch, then cosine | TBD | |

## Things worth deciding deliberately

- **LR confound.** Dividing logits by T rescales the gradient by about 1/T, so with SGD,
  changing T also changes the effective step size. We run `fixed` (same LR for every T)
  and `compensated` (LR x T) to separate the two effects. `compensated` is a first-order
  heuristic. For a properly *tuned* condition, sweep `optim.lr` per temperature on the
  validation set and pick the best, e.g.
  `python scripts/run_sweep.py --config ... --set optim.lr=0.03`, repeated per LR.
- **Adaptive optimizers.** Adam/AdamW normalise gradient magnitude, so the 1/T scaling
  mostly cancels on its own and `compensated` equals `fixed` for them (handled in
  `scaled_lr`). For the GRU, consider whether you want SGD there too, so the LR
  conditions are comparable across datasets.
- **Training-time vs post-hoc temperature.** This repo trains with temperature T and
  reports NLL/ECE both at T (`*_trainT`) and at raw logits (`*_T1`). Post-hoc temperature
  scaling (Guo et al.) is a separate baseline worth adding: fit T on the validation set
  after training at T=1 and compare calibration.
- **Large T with `compensated`.** LR x T can get large (T=10 -> 10x the base LR) and
  diverge. Diverged runs are recorded (`diverged: true`) rather than crashing the sweep;
  decide whether to cap the LR or report divergence as a result.
- **Compute.** CIFAR-10 / ResNet-18 sweeps are cheap. ImageNet is not: consider fewer
  temperatures, one seed, fewer epochs, or a subset.
- **Validation vs test.** Curves and epochs-to-target use a fixed held-out validation
  split; the test set is evaluated once at the end of each run.
