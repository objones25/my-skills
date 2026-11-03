# Debugging PyTorch

Shapes, devices, dtypes, NaNs, "it runs but won't learn", OOM, slow training.

## Overview

PyTorch failures cluster into a handful of recurring shapes: a tensor on the wrong
device, a dimension that does not line up, a loss that goes NaN, or the worst one -
training that runs cleanly and learns nothing. This file is symptom-first: find the
error in the table, jump to the section. The single most powerful technique is the
recommended approach below - reach for it first.

## Recommended approach

**Overfit one batch before touching anything else.** Take a single batch and train on
it for many steps. A correct model plus loss plus optimizer, with enough capacity and
hard targets, drives one batch's loss to near zero (to a known floor with label
smoothing or soft targets; `scripts/diagnose.py overfit` computes the label-smoothing
floor and also checks the model beats a constant prediction). If it cannot, the bug is in your model, loss, or step - not your data
or hyperparameters - and the search space just collapsed.

```python
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(200):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        print(step, loss.item())
# Expect loss -> ~0. If it plateaus high, the bug is in model/loss/step.
```

Memorizing one batch does not depend on generalization, data quality, or LR
schedules, so a failure eliminates all of them. What remains is short: gradients not
flowing (frozen params, detached graph, optimizer over the wrong params), a broken
loss (wrong targets, double activation), a shape or broadcast bug, or `model.eval()`
left on. If one batch *does* overfit but the full run will not, the bug moved
downstream - data and labels, LR for the real data scale, or too-strong augmentation.

## Symptom index

| Symptom / error text | Likely cause | Section |
|----------------------|--------------|---------|
| `mat1 and mat2 shapes cannot be multiplied` | wrong `in_features`, missing transpose | Shapes |
| `Expected all tensors to be on the same device` | model and batch on different devices | Devices |
| `expected scalar type Long but found Float` | wrong target dtype for the loss | dtypes |
| `Expected target size (B, C), got (B,)` | one-hot target passed to CrossEntropy | dtypes |
| `Boolean value of Tensor is ambiguous` | tensor used in a Python `if` | dtypes |
| `cannot assign module before Module.__init__() call` | layer assigned before `super().__init__()` | Shapes |
| `Trying to backward through the graph a second time` | graph freed, or a loss kept across iterations | Gradients |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | in-place op on a tensor autograd needs | Gradients |
| `view size is not compatible with input tensor's size and stride` | `.view()` after a transpose | Shapes |
| loss is `nan` or `inf` after a few steps | LR too high, `log(0)`, fp16 overflow | NaN |
| loss flat, accuracy at chance | the "runs but won't learn" cluster | Won't learn |
| train accuracy high, val accuracy low | overfitting | Fit |
| `CUDA out of memory` | batch or activations too big, graph leak | OOM |
| GPU near-idle, training slow | data-loading bound, sync points | Slow |
| `Tensor.names` / named tensor `AttributeError` | named tensors were removed in 2.13 | - |
| `FutureWarning: all_gather_into_tensor` (2.13, 2.14 only) | renamed to `all_gather_single` in 2.13; on 2.15 the old name is a silent alias | `12-distributed.md` |
| `FutureWarning: optimizer parameters need to be organized in ordered collections` | params passed as a `set` (2.15) | pass a list |
| `ValueError: Muon only supports 2D parameters` | a bias or norm weight given to `Muon` | split params: 2-D to Muon, the rest to AdamW (`04-losses-optimizers.md`) |
| `UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly` | `checkpoint()` without `use_reentrant` | pass `use_reentrant=False` |
| `UserWarning: flex_attention called without torch.compile()` | eager FlexAttention materializes the full score matrix | `torch.compile(flex_attention)` |
| `TypeError: Input tensor should be a float tensor. Got torch.uint8.` | `v2.Normalize` before `ToDtype` | `08-transforms-augmentation.md` |
| `ImportError` on a second `import torch` in one process (2.15) | `importlib.reload(torch)` or deleting it from `sys.modules` | restart the interpreter |
| `UnpicklingError: Weights only load failed` | `torch.load` (default `weights_only=True` since 2.6) met a custom class | `torch.serialization.safe_globals([...])`, or `weights_only=False` for trusted files |

Diagnostic tools:

| Tool | Call | Use |
|------|------|-----|
| Find the first NaN | `torch.autograd.set_detect_anomaly(True)` | debug only, slow |
| Guard a step | `assert torch.isfinite(loss)` | catch a non-finite loss early |
| Check inputs | `torch.isnan(x).any()` | rule out bad data |
| Inspect device | `next(model.parameters()).device`, `x.device` | device mismatch |
| Inspect grads | `p.grad.norm()` per named param | dead, huge, or `None` |
| Peak memory | `torch.accelerator.max_memory_allocated()` | size the batch |
| Live vs cached | `memory_allocated()` versus `memory_reserved()` | leak versus allocator cache |
| Save memory | `torch.utils.checkpoint.checkpoint` | recompute activations |
| Compile diagnostics | `TORCH_LOGS="graph_breaks,recompiles"` | why compile is slow |
| Scripted triage | `python scripts/diagnose.py overfit\|grads\|nan --model file.py:factory` | overfit test, per-parameter gradient table, first module to emit a non-finite value |
| Static check | `python scripts/lint_pytorch.py src/` | dropped `.to()`, softmax before CrossEntropy, plain list of layers, and 20 more |

### dtypes

| Error | Cause | Fix |
|-------|-------|-----|
| `expected scalar type Long but found Float` | CrossEntropy got float targets | `y = y.long()` (class indices) |
| `result type Float can't be cast to Long` | BCE/MSE got int targets | `y = y.float()` |
| `Expected target size (B, C), got (B,)` | CrossEntropy target is one-hot | pass indices `(B,)` |
| `Boolean value of Tensor is ambiguous` | tensor in a Python `if` | `.item()`, or `.any()` / `.all()` |

### NaN and Inf losses

| Cause | Tell | Fix |
|-------|------|-----|
| LR too high | spikes then NaN within tens of steps | lower LR, add warmup |
| Exploding gradients | grad norm blows up | `clip_grad_norm_(params, 1.0)` |
| `log(0)` or `sqrt(0)` | NaN at a boundary input | add `eps`, use the fused loss |
| Manual softmax/sigmoid then log | overflow | pass raw logits to the fused loss |
| fp16 overflow | NaN only under AMP fp16 | switch to bf16, or use an active `GradScaler` |
| Bad input data | NaN from step 0 | `torch.isnan(x).any()` |
| Zero-length sequence or empty batch | NaN mean over 0 elements | `drop_last=True`, or guard |

### It runs but won't learn

Each item is a real, common cause. Walk the list.

| Check | Symptom if wrong | Fix |
|-------|------------------|-----|
| `zero_grad`, `backward`, `step` all present and ordered | no updates | restore the call |
| optimizer holds the *final* `model.parameters()` | params never update | rebuild after moving the model or swapping layers |
| params require grad | `p.grad is None` after backward | you froze too much |
| no stray `detach()` / `no_grad()` | grad does not reach early layers | remove the cut |
| layers in `nn.ModuleList`, not a plain list | those layers never train | use `nn.ModuleList` |
| LR sane | flat (too small) or diverging (too large) | LR range sweep |
| raw logits plus the right loss | distorted gradients | no final softmax/sigmoid |
| `model.train()` set | dropout and BN misbehave | call it each epoch |
| inputs normalized | activations stall | see `08-transforms-augmentation.md` |
| labels aligned with inputs | learns nothing | verify the Dataset returns matching `(x, y)` |
| loss is not already near its floor | "flat" loss is actually converged | check what chance-level loss is (`ln(C)` for C balanced classes) |

## Walkthrough

The silent broadcasting bug - the shape error that *does not* raise. Output executed
on torch 2.15.0.

**1. Prediction and target with mismatched shapes.** `(4, 1)` versus `(4,)`, the
common off-by-one from a `Linear(..., 1)` head.

```python
import torch, torch.nn as nn
pred   = torch.tensor([1., 2., 3., 4.]).reshape(4, 1)   # (4, 1)
target = torch.tensor([1., 2., 3., 4.])                 # (4,)
print("shapes:", tuple(pred.shape), tuple(target.shape))
```
```
shapes: (4, 1) (4,)
```

**2. Subtraction broadcasts to `(4, 4)` instead of erroring.** Every prediction is
compared to every target - a meaningless grid. The predictions are *exactly* right,
and the loss still reads 2.5.

```python
diff = pred - target
print("diff shape:", tuple(diff.shape), "| MSE:", (diff ** 2).mean().item())
print("MSELoss   :", nn.MSELoss()(pred, target).item())
```
```
diff shape: (4, 4) | MSE: 2.5
MSELoss   : 2.5
```

`nn.MSELoss` emits a `UserWarning` here ("Using a target size that is different to
the input size") but still returns a number and still backpropagates. A warning in
the log is the only signal.

**3. The fix: make the shapes identical before the loss.**

```python
fixed = pred.squeeze(-1) - target
print("fixed shape:", tuple(fixed.shape), "| MSE:", (fixed ** 2).mean().item())
```
```
fixed shape: (4,) | MSE: 0.0
```

When a loss looks wrong but nothing raises, suspect broadcasting first:
`pred.squeeze(-1)` or `target.unsqueeze(-1)` before the loss.

## Gotchas

- **`mat1 and mat2` errors**: check axis order (transpose, `batch_first`) before
  resizing layers; the first Linear after a conv stack needs `in_features = C*H*W`.
- **Broadcasting hides shape bugs.** Make prediction and target shapes identical.
- **Tensors created in `forward` default to CPU** - pass `device=x.device`.
- **CrossEntropy wants int64 indices; BCE and MSE want float** matching-shape targets.
- **Most NaNs are LR-too-high or a manual log/exp** - pass raw logits to fused losses,
  clip gradients, use bf16 or an active `GradScaler` for fp16.
- **`set_detect_anomaly(True)` is debug-only** - it makes backward much slower.
- **OOM during backward means activations, not weights** - smaller batch,
  accumulation, AMP, or checkpointing. `empty_cache()` does not free live tensors.
- **Low GPU utilization means data-bound or sync-bound** - more workers, `pin_memory`,
  fewer `.item()` calls, AMP, `torch.compile`.
- **A `UserWarning` during training is worth reading.** Broadcasting, deprecated
  collectives, and silent SDPA fallbacks all announce themselves that way.

## Alternatives

When the checklist does not localize it, look at the gradients directly.

```python
loss.backward()
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"{name}: grad is None (frozen / detached / unused)")
    else:
        print(f"{name}: grad norm {p.grad.norm():.3e}")
```

`None` means the parameter never reached the loss. Norm `0.0` means connected but the
gradient vanished (dead ReLU, saturated activation, multiplied by zero). Norm `1e3`
or more means exploding - clip it.

**Overfitting versus underfitting**, from the train/val gap:

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| train improving, val improving then degrading | overfitting | augment, dropout, weight decay, early stop, more data |
| train low, val low | underfitting or a bug | more capacity, higher LR, longer; first overfit one batch |
| train roughly equal to val, both mediocre | underfit or data ceiling | bigger model, better features, check label noise |
| val much better than train | leak or bug | dropout left on at eval, easier val set, data leak |

**CUDA OOM**, in order of effort to payoff: smaller batch plus gradient accumulation
-> AMP (`autocast` roughly halves activation memory) -> log `loss.item()` not `loss`
-> `@torch.inference_mode()` at eval -> gradient checkpointing. OOM during `backward`
rather than `forward` points at activations: every layer's forward output is held for
backward, and peak memory hits on the first backward when activations, gradients, and
optimizer state coexist. Measure with `torch.accelerator.max_memory_allocated()`
after one step and size the batch to leave headroom.

**Slow training.** Low GPU utilization usually means data-bound, not compute-bound:
raise `num_workers`, set `pin_memory=True` and `persistent_workers=True`, cut
per-step `.item()`/`.cpu()`/`print` syncs (each forces a device sync), enable AMP,
and `torch.compile`. Confirm rather than guess - see `13-performance-profiling.md`.

**`torch.compile` specifically.** If compiled code is slower or wrong, set
`TORCH_LOGS="graph_breaks,recompiles"` and check for a recompilation loop from
changing input shapes. `torch.compiler.disable` on the offending function isolates
it. See `11-compile-export.md`.

## Related

- `03-models-layers.md` - shape bugs at the flatten boundary, the ModuleList trap.
- `01-devices-amp.md` - device mismatch, buffers, the caching allocator.
- `04-losses-optimizers.md` - loss target conventions behind dtype and NaN errors.
- `05-data-training-loop.md` - the step order the won't-learn checklist verifies.
- `13-performance-profiling.md` - measuring instead of guessing at slowness.

## Sources

- `inspect` against torch 2.15.0 (re-checked on 2.14.1); walkthrough output (including the `UserWarning`)
  executed on the same build.
- docs.pytorch.org/docs/2.13 - `notes/faq.html` (OOM), `notes/randomness.html`,
  `amp.html`, `torch_environment_variables.html`.
