# Losses, optimizers, and LR schedulers

## Overview

Three objects drive learning: the **loss** turns predictions plus targets into one
scalar, autograd fills `.grad`, and the **optimizer** reads `.grad` and updates
parameters by its rule. A **scheduler** reshapes the learning rate over time. The
whole game is picking a loss that matches your output convention, defaulting to
AdamW, and almost always adding a schedule.

## Recommended approach

Raw logits out of the model; the loss owns the softmax or sigmoid.

```python
import torch, torch.nn as nn

loss_fn   = nn.CrossEntropyLoss()                        # logits + int64 indices
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)   # wd already 0.01
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

**`AdamW`'s `weight_decay` defaults to `0.01` (unchanged from 2.13 through 2.15)** - you do not need to
pass it to get decay, and older guides claiming the default is `0` are describing
`Adam`, not `AdamW`. Set it explicitly only to change the value. `lr` runs 1e-3 to
3e-4 for training from scratch, 1e-5 to 5e-5 for fine-tuning.

## API reference

Loss selection - the target convention is the part people get wrong.

| Task | Loss | Model output | Target |
|------|------|--------------|--------|
| Multiclass | `nn.CrossEntropyLoss()` | raw logits `(B, C)` float | class indices `(B,)` **int64** |
| Binary | `nn.BCEWithLogitsLoss()` | raw logit `(B,)` or `(B, 1)` | **float** 0/1, same shape |
| Multilabel | `nn.BCEWithLogitsLoss()` | logits `(B, C)` | float 0/1 `(B, C)` |
| Regression | `nn.MSELoss` / `nn.L1Loss` / `nn.SmoothL1Loss` / `nn.HuberLoss` | raw `(B, *)` | same shape |
| Sequence LM | `nn.CrossEntropyLoss()` over flattened tokens | `(B*T, V)` | `(B*T,)` |
| Large-vocab LM | `nn.LinearCrossEntropyLoss` (2.13+) | hidden `(B, D)` - the layer owns the projection | `(B,)` int64 |

Signatures introspected from torch 2.15.0 (identical on 2.14.1):

| API | Signature (key defaults) |
|-----|--------------------------|
| `nn.CrossEntropyLoss` | `(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)` |
| `nn.BCEWithLogitsLoss` | `(weight=None, reduction='mean', pos_weight=None)` |
| `nn.SmoothL1Loss` | `(reduction='mean', beta=1.0)` |
| `nn.HuberLoss` | `(reduction='mean', delta=1.0)` |
| `nn.LinearCrossEntropyLoss` | `(in_features, num_classes, *, out_features=(), bias=False, reduction='mean', weight=None, ignore_index=None, label_smoothing=0.0, options=None)` |
| `F.linear_cross_entropy` | `(input, linear_weight, target, *, linear_bias=None, weight=None, reduction='mean', ignore_index=None, label_smoothing=0.0, options=None)` |
| `torch.optim.AdamW` | `(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)` |
| `torch.optim.Adam` | same but `weight_decay=0`, plus `decoupled_weight_decay=False` |
| `torch.optim.SGD` | `(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)` |
| `torch.optim.Muon` | `(params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None)` |
| `torch.optim.Adafactor` | `(params, lr=0.01, beta2_decay=-0.8, eps=(None, 0.001), d=1.0, weight_decay=0.0)` |
| Param groups | `AdamW([{"params": a, "lr": 1e-5}, {"params": b}], lr=1e-3)` |
| Grad clip | `nn.utils.clip_grad_norm_(params, max_norm)` - after `backward`, before `step` |

Schedulers (all in `torch.optim.lr_scheduler`):

| Scheduler | Step cadence | Needs total steps? | Best for |
|-----------|--------------|:---:|----------|
| `CosineAnnealingLR(opt, T_max, eta_min=0)` | per epoch | no | general-purpose smooth decay |
| `OneCycleLR(opt, max_lr, steps_per_epoch, epochs)` | per **batch** | yes | fast convergence, image classification |
| `ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)` | per epoch, `step(metric)` | no | unknown epoch count |
| `LinearLR` + `CosineAnnealingLR` via `SequentialLR` | per step | yes | transformers (warmup then decay) |
| `CosineAnnealingWarmRestarts`, `PolynomialLR`, `CyclicLR`, `StepLR`, `MultiStepLR`, `ExponentialLR`, `LambdaLR`, `ChainedScheduler` | varies | | specialized |

`CrossEntropyLoss` options worth knowing: `label_smoothing=0.1` (regularizes),
`weight=class_weights` (imbalance), `ignore_index=-100` (skip padding tokens). For
`BCEWithLogitsLoss`: `pos_weight=torch.tensor([w])` up-weights positives.

## Walkthrough

A single 3-class example, computing exactly what `CrossEntropyLoss` returns via the
fused log-sum-exp path it uses internally. Output executed on torch 2.15.0.

**1. Raw logits and an integer target.** No softmax on the model; the target is a
class index, not one-hot.

```python
import torch, torch.nn as nn
logits = torch.tensor([[2.0, 1.0, 0.1]])   # (1, 3)
target = torch.tensor([0])                 # (1,) int64 class index
ce = nn.CrossEntropyLoss()(logits, target)
print("CrossEntropyLoss:", round(ce.item(), 4))
```
```
CrossEntropyLoss: 0.417
```

**2. What it computed.** The fused form is `-z_y + log(sum_j exp(z_j))`, evaluated
stably by subtracting the max first.

```python
p = logits.softmax(-1)
print("softmax:", [round(v, 4) for v in p[0].tolist()])
print("-log p_y:", round(-torch.log(p[0, 0]).item(), 4))
```
```
softmax: [0.659, 0.2424, 0.0986]
-log p_y: 0.417
```

Same number, two routes - that is what "fused" buys: no explicit softmax to overflow.

**3. Optimizer defaults, read off the installed library** rather than remembered.

```python
import inspect
print("AdamW wd:", inspect.signature(torch.optim.AdamW).parameters["weight_decay"].default)
print("Adam  wd:", inspect.signature(torch.optim.Adam).parameters["weight_decay"].default)
```
```
AdamW wd: 0.01
Adam  wd: 0
```

**4. The fused loss (added in 2.13) is numerically equivalent** to projecting then applying
cross-entropy - it just never materializes the `(B, V)` logits matrix.

```python
import torch.nn.functional as F
B, D, V = 8, 32, 1000
torch.manual_seed(0)
h = torch.randn(B, D); W = torch.randn(V, D) * 0.02; t = torch.randint(0, V, (B,))
fused   = F.linear_cross_entropy(h, W, t)
unfused = F.cross_entropy(h @ W.T, t)
print("fused:", round(fused.item(), 6), "| unfused:", round(unfused.item(), 6),
      "| allclose:", torch.allclose(fused, unfused, atol=1e-5))
```
```
fused: 6.888349 | unfused: 6.888349 | allclose: True
```

At `V = 1000` the saving is irrelevant. At `V = 128000` with a long sequence, the
`(B*T, V)` logits tensor is tens of gigabytes, and this is where the ~4x peak-memory
reduction comes from. The release blog labels it API Unstable, and 2.14 already removed
its `balanced` accuracy policy, and 2.15 added soft-label targets with `reduction="none"` on the
chunked path, so pin the torch version if you depend on its options.

## Gotchas

- **`CrossEntropyLoss` wants logits and int64 class indices**, not softmax outputs
  and not one-hot targets. A float target with the logits' shape `(B, C)` silently switches to the soft-target path (a float `(B,)` target raises)
  instead of erroring. For sequences, flatten to `(B*T, V)` and `(B*T,)`.
- **`BCEWithLogitsLoss` wants logits**, not sigmoid outputs. Threshold at logit `0`
  (equivalently sigmoid `0.5`).
- **Build the optimizer over the final `parameters()`** - after moving the model to
  the device and after swapping any layers. Stale references optimize dead tensors.
- **Scheduler cadence**: per epoch for `CosineAnnealingLR` and `ReduceLROnPlateau`,
  per **batch** for `OneCycleLR`. Stepping `OneCycleLR` per epoch burns the whole
  cycle in one epoch.
- **`scheduler.step()` before the first `optimizer.step()`** warns and skips the
  first LR.
- **`Adam(weight_decay=...)` is coupled L2 - use `AdamW`** (or
  `Adam(decoupled_weight_decay=True)`, available since 2.4).
- **Clip gradients after `backward`, before `step`.** Under fp16 AMP call
  `scaler.unscale_(optimizer)` first, then clip, then `scaler.step(optimizer)`.
- **Excluding norm gains and biases from weight decay is convention, not a tested
  result.** nanoGPT excludes them; OLMoE decays every parameter. The ablated case is
  embeddings: OLMo 2 found decaying them raised its gradient spike score (0.16 vs
  0.092). If you split param groups, say it is convention.
- **Pass parameters as a list, not a set.** A set has no stable order, so optimizer
  state cannot be matched back to parameters on reload; 2.15 emits a `FutureWarning`.
- **`nn.LinearCrossEntropyLoss` owns the projection**, so you delete the final
  `nn.Linear` from the model rather than stacking it in front.

## Alternatives

**AdamW versus SGD.** AdamW is the safe default - forgiving, standard for
transformers and fine-tuning. SGD with momentum
(`SGD(params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)`) often
generalizes slightly better on well-tuned CNN image classification, but needs a good
schedule and more tuning. SGD's `weight_decay` *is* plain L2, and that is correct -
the coupling problem is specific to adaptive methods.

**Muon** (`torch.optim.Muon`; present in 2.14.1 and 2.15.0) orthogonalizes the momentum
update with a Newton-Schulz iteration (a few matrix multiplies that push the update's
singular values toward 1) and has been effective on 2-D weight matrices in LLM
pretraining. It accepts 2-D parameters only: a 1-D parameter raises `ValueError: Muon only
supports 2D parameters` (reproduced on 2.15.0). So embeddings, the LM head, and all 1-D
parameters (norms, biases) go to a separate `AdamW` optimizer, not a param group of the
same one. Its default `weight_decay` is `0.1`, ten times AdamW's. `adjust_lr_fn` picks how
the LR is rescaled per matrix shape: `None`/`"original"` scales by `sqrt(max(1, A/B))`
for an `A x B` matrix; `"match_rms_adamw"` scales by `0.2 * sqrt(max(A, B))`, which per the
docs lets you reuse an LR and weight decay tuned for AdamW; `"spectral_unclamped"` scales by
`sqrt(A/B)` with no clamp. Before 2.15, Muon overwrote its
momentum buffer with bf16 gradients (fixed in 2.15 per the release notes).
**Adafactor** trades optimizer-state memory for a factored second moment - useful
when Adam's two full-size moment buffers do not fit.

Why AdamW beats Adam-with-weight-decay: plain `Adam(weight_decay=lam)` adds
`lam * theta` to the *gradient*, so decay passes through Adam's `1/sqrt(v_hat)`
scaling and parameters with large gradient history get *less* effective decay. AdamW
decays the weight directly after the adaptive step, so every parameter decays at the
same rate.

**Regression loss character:**

| Loss | Per-element | Character |
|------|-------------|-----------|
| `nn.MSELoss` | `(pred - y)^2` | smooth; outliers dominate |
| `nn.L1Loss` | `abs(pred - y)` | robust to outliers, non-smooth at 0 |
| `nn.SmoothL1Loss` / `nn.HuberLoss` | quadratic near 0, linear tail | robust and smooth; default for bbox regression |

`SmoothL1Loss(beta=b)` and `HuberLoss(delta=d)` describe the same curve but differ by
a factor of `delta` in scale - Huber's output is `delta` times SmoothL1's for the same
transition point.

**Warmup then cosine**, the transformer standard:

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cos  = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warm, cos], milestones=[warmup_steps])
```

## Related

- `03-models-layers.md` - the raw-logits head is what keeps these losses stable.
- `05-data-training-loop.md` - where `optimizer.step()` and `scheduler.step()` sit.
- `06-tuning-reproducibility.md` - searching `lr`, `weight_decay`, and the schedule.
- `07-fine-tuning.md` - param groups for discriminative learning rates.

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) for every default in this file; the
  walkthrough output was executed on the same build.
- PyTorch 2.13 release blog - `nn.LinearCrossEntropyLoss` memory claim.
- docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html via Context7 - Muon algorithm,
  `adjust_lr_fn` options; the 1-D `ValueError` reproduced on 2.15.0.
- PyTorch 2.15 release notes - Muon bf16 momentum fix, set-of-params deprecation,
  `torch.optim.functional`, soft labels in `linear_cross_entropy`.
- docs.pytorch.org/docs/2.13 - `optim.html`,
  `generated/torch.nn.CrossEntropyLoss.html`.
