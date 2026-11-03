# Data and the training loop

`Dataset`/`DataLoader`, the train and eval loop, AMP in-loop, accumulation, checkpoints.

## Overview

Training is one loop: feed batches from a `DataLoader`, compute loss, backprop, step.
Evaluation is the same loop with gradients off, `model.eval()` on, and a metric
instead of a backward pass. AMP, gradient accumulation, `torch.compile`, and
checkpointing bolt onto that five-line core without changing it.

## Recommended approach

Each DataLoader iteration yields one **batch**; the fixed order inside it is
`zero_grad` -> `forward` -> `loss` -> `backward` -> `step`.

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device, log_every=100):
    model.train()
    running = 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()                     # set_to_none=True by default
        logits = model(x)                         # forward, builds the graph
        loss = loss_fn(logits, y)                 # scalar
        loss.backward()                           # fill .grad
        optimizer.step()                          # update params
        running += loss.item() * x.size(0)        # weight by batch size
        if (i + 1) % log_every == 0:
            print(f"  batch {i+1}/{len(loader)}  loss {loss.item():.4f}")
    return running / len(loader.dataset)          # mean over samples, not batches
```

The full epoch loop wraps it with validation, the scheduler, and checkpointing:

```python
device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()

best_val = float("inf")
for epoch in range(EPOCHS):
    tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    scheduler.step()                                   # per-epoch scheduler
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best.pth")
```

`assets/train_template.py` is this loop, complete and runnable, with AMP,
accumulation, resume, and early stopping already wired in.

## API reference

| API | Form | Notes |
|-----|------|-------|
| Map-style dataset | subclass with `__len__` and `__getitem__(i) -> (x, y)` | augment inside `__getitem__` |
| Iterable dataset | subclass `IterableDataset` with `__iter__` | streams; shard by worker yourself |
| `DataLoader` | `(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='', in_order=True)` | unchanged from 2.13 through 2.15 |
| Train mode | `model.train()` | dropout on, BN uses batch stats |
| Eval mode | `model.eval()` plus `@torch.inference_mode()` | dropout off, BN uses running stats, no graph |
| Detach metric | `loss.item()` | never store the tensor |
| AMP | `torch.autocast(device.type, dtype=...)`; fp16 adds `torch.amp.GradScaler` | see `01-devices-amp.md` |
| Accumulate | divide loss by `K`, step every `K` batches | effective batch = `batch_size * K` |
| Compile | `torch.compile(model)` | see `11-compile-export.md` |
| Save weights | `torch.save(model.state_dict(), path)` | not the model object |
| Load | `torch.load(path, weights_only=True)` | `weights_only` defaults to `True` since 2.6 |
| Load safetensors | `torch.load("model.safetensors")` | native since 2.13, no extra library (`tests/test_claims.py`) |

DataLoader knobs:

| Knob | Effect | Default |
|------|--------|---------|
| `num_workers` | parallel loader processes; start at 4 | `0` (main thread) |
| `pin_memory` | enables fast async host-to-device copy | `False` |
| `persistent_workers` | avoid respawning workers each epoch | `False` |
| `prefetch_factor` | batches prefetched per worker | `2` when workers > 0 |
| `drop_last` | drop the ragged final batch (stabilizes BN) | `False` |
| `in_order` | keep batch order deterministic across workers; `False` can raise throughput with uneven workers | `True` |

`model.train()` / `model.eval()` flip Dropout and BatchNorm only - they do **not**
touch gradients:

| Mode | Dropout | BatchNorm | Gradients |
|------|---------|-----------|-----------|
| `train()` | drops units | batch stats, updates running stats | tracked |
| `eval()` | identity | stored running stats | still tracked unless disabled |

## Walkthrough

The eval metric - the deterministic half of the loop. Output executed on torch 2.15.0.

**1. A batch of logits and integer targets.** `B=4`, `C=3`.

```python
import torch
logits = torch.tensor([[2.1, 0.3, -1.0],
                       [0.5, 1.8,  0.2],
                       [0.1, 0.0,  2.5],
                       [1.2, 1.1,  0.0]])   # (4, 3), from model(x)
y = torch.tensor([0, 1, 2, 1])              # (4,) int64
```

**2. Predictions are the argmax over classes.**

```python
preds = logits.argmax(1)
print("preds:", preds.tolist(), "| targets:", y.tolist())
```
```
preds: [0, 1, 2, 0] | targets: [0, 1, 2, 1]
```

**3. Accuracy is the mean of correct predictions.** Row 3 predicts class 0 but the
target is 1, so 3 of 4 are right.

```python
print("correct:", int((preds == y).sum()), "of", len(y),
      "| acc:", (preds == y).float().mean().item())
```
```
correct: 3 of 4 | acc: 0.75
```

In the loop this is `correct += (logits.argmax(1) == y).sum().item()`, accumulated
over batches and divided by `n` at the end:

```python
@torch.inference_mode()                    # faster than no_grad
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n       # (avg loss, accuracy)
```

## Gotchas

- **`zero_grad` before `backward`, `backward` before `step`** - grads accumulate.
- **`model.eval()` for validation, `model.train()` back after.** Forgetting `eval()`
  leaves dropout active and BN on batch stats (noisy, pessimistic val, wildly wrong
  on tiny batches); forgetting to switch back silently disables dropout for the rest
  of training.
- **Never shuffle val/test** - metrics need a stable order.
- **Accumulate `loss.item()`, not the tensor** - storing the tensor keeps its whole
  graph alive. `.item()` forces a device sync, so call it once per logging interval,
  not per micro-op.
- **Gradient accumulation needs `loss / K`** or the effective gradient is `K` times
  too large. Handle the final partial group so the last batches still step.
- **Save `state_dict`, not the model object** (pickling the class breaks on
  refactor). To resume, also save optimizer, scheduler, and epoch - without optimizer
  state, Adam's moments reset and training lurches.
- **`num_workers > 0` needs a picklable dataset and an `if __name__ == "__main__":`
  guard** on Windows and macOS, and since Python 3.14 on Linux too, where the default
  start method changed from `fork` to `forkserver` (CPython gh-84559). Notebooks may hang or raise
  `BrokenPipeError`: use `num_workers=0` there, or pass `multiprocessing_context`.
- **Do heavy per-sample work in `__getitem__`, not `__init__`** - `__getitem__` runs
  across workers; `__init__` runs once on the main process and is pickled to each.
- **`len(loader)` is the number of batches** (`ceil(len(dataset) / batch_size)`);
  `len(loader.dataset)` is the number of samples. Dividing the running loss by the
  wrong one is a silent scale error.

## Alternatives

The three accelerators layer onto the same five lines.

**Mixed precision** - wrap forward and loss in `autocast`; fp16 adds a `GradScaler`,
bf16 does not.

```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
# optional: scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update()
```

**Gradient accumulation** - simulate a bigger batch than fits in memory. Works
precisely because grads accumulate by default.

```python
for i, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y) / K
    loss.backward()
    if (i + 1) % K == 0 or (i + 1) == len(loader):
        optimizer.step(); optimizer.zero_grad()
```

**`torch.compile`** - `model = torch.compile(model)` after moving to device; same
loop, fused kernels. See `11-compile-export.md`.

**Checkpointing - inference versus resume:**

```python
# Inference only
torch.save(model.state_dict(), "weights.pth")
model.load_state_dict(torch.load("weights.pth", weights_only=True)); model.eval()

# Resume training: capture all mutable state
torch.save({"epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "best_val": best_val}, "ckpt.pth")
```

Since 2.6, `torch.load` defaults to `weights_only=True` when `pickle_module` is
unset - it refuses to execute arbitrary pickled code, restricting the unpickler to
tensors and primitives. Loading an old checkpoint containing custom classes needs
`torch.serialization.add_safe_globals([...])` or `weights_only=False` (trusted
sources only). Since 2.15 nested (jagged or strided) tensors are no longer on the
`weights_only` allowlist; allow `torch.nested._internal.nested_tensor._rebuild_njt` (the global named in the error) with
`torch.serialization.safe_globals([...])`.
For sharded/distributed checkpoints use
`torch.distributed.checkpoint` instead - see `12-distributed.md`.

**Logging.** `torch.utils.tensorboard.SummaryWriter` is built in but needs the
`tensorboard` package installed. Convert the batch index into a global step so curves
stay continuous across epochs: `step = epoch * len(loader) + i`. Average the loss
over the logging window rather than logging a single noisy batch.

## Related

- `01-devices-amp.md` - device placement, `pin_memory`, AMP dtype rules.
- `04-losses-optimizers.md` - the loss/optimizer/scheduler driving the step.
- `06-tuning-reproducibility.md` - seeding the loop and the DataLoader workers.
- `13-performance-profiling.md` - when the loop is data-bound rather than compute-bound.

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) (`DataLoader`, `torch.load`, `torch.save`);
  walkthrough output executed on the same build.
- docs.pytorch.org/docs/2.13 - `data.html`, `notes/serialization.html`,
  `generated/torch.optim.Optimizer.zero_grad.html`, `notes/amp_examples.html`.
