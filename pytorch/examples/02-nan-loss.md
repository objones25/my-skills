# Loss is NaN from step 0

*Illustrative scenario; the script output is real (torch 2.15.0, CPU).*

**Question.** "The loss prints `nan` on the very first step. I lowered the LR to 1e-6 and
it is still `nan`." The model is `examples/models/nan_log.py`.

## Verdict

Not an LR problem: a NaN on step 0 exists before any update happens. Module `1`
(`LogFeatures`) takes `torch.log` of values that can be negative. Clamp or shift its input
(`torch.log(x.clamp_min(1e-6))`, or `log1p` of a non-negative quantity), or remove it.

## Evidence

```
$ python scripts/diagnose.py nan examples/models/nan_log.py
== nan
input non-finite: False
first module with non-finite output: 1 (LogFeatures)
loss: nan
parameters with non-finite grads: not checked (the loss is already non-finite)
```

The script hooks every module's forward and reports the first one whose output contains
NaN or Inf, so it separates "the data is bad" (`input non-finite: True`) from "a layer made
it". Here the input is finite and the first NaN appears in `LogFeatures`.

## Why the LR change did nothing

The NaN-cause table in `references/10-debugging.md` splits by *when* NaN appears.
Spikes-then-NaN after tens of steps points at the LR or exploding gradients; NaN at step 0
points at the data or a numerically unsafe op (`log(0)`, `sqrt` of a negative, a manual
softmax followed by `log`). An LR cannot matter before the first `optimizer.step()`.

## What would change the answer

`input non-finite: True` would move the search to the data pipeline. A NaN that appears
only under fp16 autocast and not in fp32 would point at overflow (switch to bf16 or use an
active `GradScaler`). For NaN that only shows up in the backward pass,
`torch.autograd.set_detect_anomaly(True)` names the backward op, at a large speed cost.
