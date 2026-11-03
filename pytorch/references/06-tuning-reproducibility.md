# Hyperparameter tuning and reproducibility

## Overview

Tuning is a search layered on training: pick which knobs to turn, choose a strategy
to explore them, and make every run reproducible and logged so the comparison is
real. Order of operations: get learning rate and batch size roughly right, control
randomness so two runs are comparable, then automate the search. An untracked,
unseeded sweep produces numbers you cannot trust.

## Recommended approach

Seed Python, NumPy, torch, **and** the DataLoader workers; optimize against a fixed
validation split, never train loss and never test.

```python
import random, numpy as np, torch
from torch.utils.data import DataLoader

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)                 # seeds CPU and all CUDA devices
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):                 # workers reseed themselves - pin them
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)

g = torch.Generator(); g.manual_seed(42)
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=4, worker_init_fn=seed_worker, generator=g)
```

`torch.manual_seed` alone is not enough. Multi-process DataLoader workers get their
own RNG and reseed differently unless you pass `worker_init_fn` and a seeded
`generator`. Augmentation randomness lives in the workers - exactly where silent
irreproducibility creeps in.

## API reference

| Purpose | Form | Notes |
|---------|------|-------|
| Seed torch | `torch.manual_seed(s)` | CPU plus all CUDA devices |
| Seed workers | `worker_init_fn=` and `generator=` on `DataLoader` | pins augmentation RNG |
| Force determinism | `torch.use_deterministic_algorithms(True)` | errors on nondeterministic ops; slower |
| Deterministic fill | `torch.utils.deterministic.fill_uninitialized_memory` | catches uninitialized reads |
| cuDNN autotune | `torch.backends.cudnn.benchmark = False` | `True` is faster but nondeterministic |
| Env for CUBLAS | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | required for full CUDA determinism |
| Optuna study | `optuna.create_study(direction="minimize")` | TPE Bayesian by default |
| Suggest, log scale | `trial.suggest_float("lr", 1e-5, 1e-1, log=True)` | for LR and weight decay |
| Suggest, categorical | `trial.suggest_categorical("batch_size", [...])` | discrete set |
| Run search | `study.optimize(objective, n_trials=50)` | |
| Prune | `trial.report(v, step)` plus `trial.should_prune()` | kill losers early |
| Track | `torch.utils.tensorboard.SummaryWriter` | needs the `tensorboard` package |

Knobs in priority order - spend budget where the payoff is
(LR >> batch ~ schedule > weight decay > width/depth):

| Knob | Typical range | Notes |
|------|---------------|-------|
| learning rate | `1e-5` to `1e-1`, log | the single highest-impact knob |
| batch size | 16 to 512, powers of 2 | coupled to LR |
| weight decay | `1e-4` to `0.1`, log (try `0` as a separate arm) | AdamW's default is `0.01`, not `0` |
| LR schedule | cosine / one-cycle | nearly always helps |
| dropout | `0` to `0.5` | regularization |
| warmup steps | 0 to a few percent of total | important for transformers |
| augmentation strength | see `08-transforms-augmentation.md` | a regularization knob |
| hidden size / depth | task-dependent | expensive; sweep last |

## Walkthrough

Prove that seeding is what makes two runs comparable, then compute the LR-batch
coupling you will act on. Output executed on torch 2.15.0.

**1. Same seed, identical draws.**

```python
import torch
torch.manual_seed(42); a = torch.rand(3)
torch.manual_seed(42); b = torch.rand(3)
print("a:", [round(v, 4) for v in a.tolist()])
print("b:", [round(v, 4) for v in b.tolist()], "| equal:", torch.equal(a, b))
```
```
a: [0.8823, 0.915, 0.3829]
b: [0.8823, 0.915, 0.3829] | equal: True
```

**2. No reseed, a different stream.** This is what an unseeded A/B comparison is
actually measuring - noise, not your change.

```python
c = torch.rand(3)          # continues the stream
print("c:", [round(v, 4) for v in c.tolist()], "| equal to a:", torch.equal(a, c))
```
```
c: [0.9593, 0.3904, 0.6009] | equal to a: False
```

**3. Act on the LR-batch coupling.** Gradient noise scales as `1/sqrt(batch)`, so
64 -> 256 halves it. The scaling rule depends on the optimizer: linear for SGD (Goyal et
al., 2017), square root for Adam and AdamW (Malladi et al., NeurIPS 2022, whose full rule
also adjusts Adam's other hyperparameters).

```python
bs0, bs1, lr0 = 64, 256, 3e-4
k = bs1 / bs0
print("noise ratio:", (bs0 / bs1) ** 0.5)
print("AdamW, square-root rule:", round(lr0 * k ** 0.5, 6))
print("SGD, linear rule:       ", round(lr0 * k, 6))
```
```
noise ratio: 0.5
AdamW, square-root rule: 0.0006
SGD, linear rule:        0.0012
```

So for this AdamW run, quadrupling the batch points the LR toward `6e-4`, not `1.2e-3`.
Treat either rule as a starting point to retune, with warmup.

## Gotchas

- **Unseeded runs are not comparable.** Seed Python, NumPy, torch *and* DataLoader
  workers before trusting any A/B result.
- **Tune on validation, report on test.** Every decision made by looking at a set
  uses up its independence.
- **Optimize a validation metric, never training loss** - training loss is minimized
  by the most overfit configuration.
- **Sample LR and weight decay log-uniformly**, not uniformly - ratios matter more
  than differences.
- **LR and batch size are coupled** - retune LR whenever you change batch size.
- **Log per-epoch curves, not just finals** - the shape tells you to train longer,
  lower the LR, or add regularization; a single scalar cannot.
- **Full determinism is slow** and some ops have no deterministic kernel at all
  (`use_deterministic_algorithms(True)` raises on those). Reserve it for debugging
  and publication; seed for routine sweeps.
- **`cudnn.benchmark = True` is nondeterministic** and also re-autotunes on every new
  input shape - a loss, not a win, for variable-shape workloads.
- **Changing the number of DataLoader workers changes augmentation randomness, not
  sample order.** With a seeded `generator`, a map-style dataset yields the same order
  for 0, 2 or 4 workers, but each worker's RNG seed differs, so random augmentations
  differ. It is a variable in any comparison that augments.

## Alternatives

Search strategy escalates with trial count: grid search -> random search -> Bayesian
(Optuna TPE) -> Hyperband/ASHA with early stopping.

Random beats grid at equal budget. If only 2 of 5 knobs matter, a grid re-tests the
same few values of the important ones while varying the irrelevant ones; random draws
a fresh value for every knob each trial, sampling the high-impact axes far more
densely (Bergstra and Bengio, 2012). Bayesian beats random past a handful of trials;
add pruning once trials are cheap.

The full Optuna objective:

```python
import optuna

def objective(trial):
    lr   = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd   = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    bs   = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    drop = trial.suggest_float("dropout", 0.0, 0.5)

    seed_all(42)                                    # same seed every trial
    model = make_model(dropout=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    for epoch in range(EPOCHS):
        train_one_epoch(model, loader, loss_fn, opt, device)
        val_loss, _ = evaluate(model, val_loader, loss_fn, device)
        trial.report(val_loss, epoch)               # enables pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print(study.best_params, study.best_value)
```

**Finding the LR without a full sweep:** an LR range test - ramp the LR
exponentially over a few hundred batches and plot loss against LR - locates the
usable band in one run, and the knee is roughly the max for `OneCycleLR`.

**Tracking.** TensorBoard is built in. Log every hyperparameter, train and val loss
per epoch, the selection metric, the LR trajectory, and the git commit plus seed -
the last two are what make a result re-runnable months later. Weights & Biases and
MLflow add hosted dashboards and run comparison.

## Related

- `05-data-training-loop.md` - each trial is one full run of this loop.
- `04-losses-optimizers.md` - the `lr`, `weight_decay`, and schedule being searched.
- `07-fine-tuning.md` - fine-tuning drops LR to `1e-5`-`5e-5`, a different range.
- `10-debugging.md` - controlled one-change-at-a-time diagnosis mirrors tuning.

## Sources

- docs.pytorch.org/docs/2.13 - `notes/randomness.html`, `deterministic.html`,
  `generated/torch.use_deterministic_algorithms.html`, `data.html`
  (`worker_init_fn`, `generator`).
- Walkthrough output executed on torch 2.15.0.
- Optuna documentation - `study.optimize`, `suggest_float(log=True)`, pruners.
- Bergstra and Bengio (2012), "Random Search for Hyper-Parameter Optimization," JMLR.
- Goyal et al. (2017), "Accurate, Large Minibatch SGD," arXiv:1706.02677 - linear rule for SGD.
- Malladi et al. (2022), "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms,"
  NeurIPS, arXiv:2205.10287 - square-root rule for Adam.
