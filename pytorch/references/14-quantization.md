# Quantization

## Overview

Quantization stores weights (and sometimes activations) in fewer bits than fp32 -
usually int8, int4, or fp8 - to cut model size and memory bandwidth. On modern
hardware inference is nearly always bandwidth-bound, so moving a quarter of the bytes
is most of the speedup; the arithmetic was never the bottleneck.

The important structural fact since 2.13: **quantization development has moved out of
`torch.ao.quantization` into the separate `torchao` package.** The in-core APIs are
still importable but are on a deletion path. Write new code against torchao.

## Recommended approach

Install torchao, quantize in place, then compile.

```python
import torch
from torchao.quantization import quantize_, Int4WeightOnlyConfig

model = model.eval().cuda()        # int4 kernels need CUDA (or XPU); on CPU use Int8WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=32))   # mutates model IN PLACE
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

`quantize_` walks the module tree and swaps eligible layers for quantized versions
based on the config you pass. It returns nothing useful - the mutation is the point.
Compile after quantizing so Inductor sees the quantized ops and can fuse the
dequant-matmul path.

torchao is not bundled with torch. `pip install torchao`, and check the installed
version's config names before using them - the config API is younger and moves faster
than core.

## API reference

| Concern | torchao API | Replaces |
|---------|-------------|----------|
| Entry point | `torchao.quantization.quantize_(model, config)` | `torch.ao.quantization.quantize`, `quantize_dynamic` |
| Weight-only int4 | `Int4WeightOnlyConfig(group_size=..., ...)` | eager-mode weight-only flows |
| Weight-only int8 | `Int8WeightOnlyConfig()` (runs on CPU) | `quantize_dynamic` weight-only use |
| Dynamic int8 | `Int8DynamicActivationInt8WeightConfig()` (runs on CPU) | `quantize_dynamic` |
| Float8 | `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())` | Hopper and newer GPUs |
| Per-module configs | `FqnToConfig({...})` | a different config per submodule name |
| PT2E prepare/convert | `torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e` / `convert_pt2e` | `torch.ao.quantization.quantize_fx.prepare_fx` / `convert_fx` |
| Accuracy comparison | `torch.ao.ns.fx.utils.compute_sqnr`, `compute_cosine_similarity`, `compute_normalized_l2_error` | kept public in core |

The migration table straight from the 2.13 quantization docs:

| Existing flow | Status | Migrate to |
|---------------|--------|------------|
| Eager mode (`torch.ao.quantization.quantize`, `quantize_dynamic`) | migrating | torchao `quantize_` |
| FX graph mode (`prepare_fx`, `convert_fx`) | migrating | torchao pt2e `prepare_pt2e` / `convert_pt2e` |
| pt2e quantization | already moved to torchao | `torchao.quantization.pt2e` |

The core docs state the intent to delete `torch.ao.quantization` once blockers clear.
Treat anything under that namespace as legacy.

The quantization schemes, and what each is for:

| Scheme | What is quantized | Calibration | Best for |
|--------|-------------------|-------------|----------|
| Weight-only (int8/int4) | weights only; activations stay bf16 | none | LLM inference - memory-bandwidth bound, minimal accuracy loss |
| Dynamic (int8 weights, int8 activations quantized at runtime) | both, activation scales computed per batch | none | CPU inference, transformers with variable activation ranges |
| Static / PTQ | both, activation scales fixed ahead of time | a calibration pass over representative data | CNNs, edge deployment where a fixed graph is required |
| QAT (quantization-aware training) | simulated during training | full training run | when PTQ loses too much accuracy, especially at int4 |
| Float8 | weights and activations in fp8 | none | training and inference on Hopper and newer |

## Walkthrough

A real torchao quantization on CPU first, then the sizing arithmetic that decides
whether quantization is worth it at LLM scale. Steps 1 to 3 executed on torch 2.15.0 with
torchao 0.18.0 (the walkthrough test skips them when torchao is not installed); steps 4
to 6 need nothing but Python.

**1. Quantize a small model's Linear weights to int8, in place.**

```python
import copy, io, torch, torch.nn as nn
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torch.ao.ns.fx.utils import compute_sqnr
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).eval()
x = torch.randn(16, 512)
ref = model(x)
q = copy.deepcopy(model)
ret = quantize_(q, Int8WeightOnlyConfig())
print("returns:", ret, "| weight type:", type(q[0].weight).__name__, "| bias type:", type(q[0].bias).__name__)
```
```
returns: None | weight type: Int8Tensor | bias type: Parameter
```

`quantize_` returned `None` and swapped each `Linear.weight` for an `Int8Tensor` (a tensor
subclass holding int8 values plus scales). Biases stay float. `model = quantize_(...)`
would have replaced the model with `None`.

**2. What it bought: a checkpoint about 4x smaller.**

```python
def nbytes(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.tell()
print(f"checkpoint: {nbytes(model) / nbytes(q):.1f}x smaller")
```
```
checkpoint: 3.9x smaller
```

Not exactly 4x: the per-row scales and the float biases are stored too.

**3. What it cost, measured rather than assumed.** SQNR (signal-to-quantization-noise
ratio) is `10 * log10(power of the reference output / power of the error)`, in decibels.
Every 10 dB is a 10x smaller error power, so 45 dB means the error carries about 1/30,000
of the signal's power.

```python
print(f"SQNR: {compute_sqnr(ref, q(x)).item():.0f} dB")
```
```
SQNR: 45 dB
```

A cheap proxy only. The real check is the task metric on held-out data.

**4. Weight memory by dtype at scale**, for a 7B-parameter model.

```python
params = 7e9
for name, bits in [("fp32", 32), ("bf16", 16), ("int8", 8), ("int4", 4)]:
    print(f"{name}: {params * bits / 8 / 2**30:6.1f} GiB")
```
```
fp32:   26.1 GiB
bf16:   13.0 GiB
int8:    6.5 GiB
int4:    3.3 GiB
```

bf16 to int4 turns a model that needs two 80 GiB cards into one that fits on a
24 GiB consumer GPU.

**5. Why that is also the speedup.** Single-token decoding reads every weight once
and does one multiply-add per weight - arithmetic intensity of about 2 FLOPs per
byte, far below any modern GPU's ratio of peak FLOP/s to peak bandwidth (hundreds).
Decoding is therefore bandwidth-bound, and time per token scales with bytes moved.
Quartering the bytes roughly quarters the time, which is why weight-only int4 gives
large end-to-end gains on LLM decode while barely helping a batched, compute-bound
prefill.

**6. Group size is the accuracy knob.** `group_size=32` means one scale (and zero
point) per 32 weights rather than per tensor, so the quantization grid tracks local
weight magnitude. The overhead is small - one fp16 scale per group is
`16 / (32 * 4) = 12.5%` on top of the int4 payload - and it is what makes int4
usable at all. Smaller groups mean better accuracy and more overhead.

## Gotchas

- **`torch.ao.quantization` is legacy.** Anything you write against it is scheduled
  for removal. Use torchao.
- **`quantize_` mutates in place** and returns `None`-ish. Do not write
  `model = quantize_(model, cfg)`.
- **Quantize, then compile.** Compiling first and quantizing after means Inductor
  never sees the quantized ops.
- **Config names and arguments differ across torchao versions.** Check
  `dir(torchao.quantization)` on the installed version rather than copying a config
  from an older tutorial.
- **Weight-only quantization helps decode, not prefill.** If your workload is a large
  batched forward pass, you are compute-bound and int4 weights will not help much.
- **Measure accuracy, do not assume it.** `compute_sqnr` between the fp and quantized
  outputs on real inputs is the cheap check; a task metric on a held-out set is the
  real one.
- **int4 without grouping usually breaks the model.** Per-tensor int4 has 16 levels
  for the whole weight matrix.
- **Static PTQ needs a calibration pass** over representative data. Calibrating on
  the wrong distribution is a silent accuracy loss.
- **Quantized models and training do not mix casually.** Fine-tuning a quantized
  model means QAT or QLoRA (adapters in bf16 over a frozen quantized base), not
  ordinary backprop through the quantized weights.
- **Not every backend supports every scheme.** int4 tinygemm kernels are CUDA;
  CPU inference has its own supported set. On torchao 0.18 with torch 2.15.0, CPU int4
  raised `ImportError: Requires mslk >= 1.0.0` (re-checked 2026-11-02), while
  `Int8WeightOnlyConfig` and `Int8DynamicActivationInt8WeightConfig` ran on CPU. Verify on the target
  hardware. `Int4WeightOnlyConfig`'s default `group_size` is 128.

## Alternatives

**Do you actually need quantization?** In order of effort:

| Goal | Try first |
|------|-----------|
| Model does not fit for inference | bf16, then weight-only int8, then int4 |
| Inference too slow, GPU underutilized | `torch.compile(mode="max-autotune")` before quantizing |
| Model does not fit for *training* | activation checkpointing, FSDP2, QLoRA - see 12 and 13 |
| Edge or mobile deployment | `torch.export` -> ExecuTorch with PT2E quantization |

**QLoRA** is the training-side counterpart: freeze a 4-bit quantized base and train
bf16 LoRA adapters on top. It combines `07-fine-tuning.md`'s adapter approach with
this file's memory argument and is the standard way to fine-tune a large model on one
GPU.

**Float8** is the newer direction: on Hopper and later, fp8 gives quantization-like
bandwidth savings with a numerically friendlier format than int8, and unlike int
schemes it is usable for *training*, not just inference.

For the current config catalog, supported hardware, and benchmark numbers, read the
torchao docs directly at docs.pytorch.org/ao - that project moves faster than this
reference.

## Related

- `11-compile-export.md` - compile after quantizing; export for edge deployment.
- `13-performance-profiling.md` - confirm you are bandwidth-bound before quantizing.
- `07-fine-tuning.md` - LoRA, the other half of QLoRA.
- `12-distributed.md` - the other answer to "it does not fit".

## Sources

- docs.pytorch.org/docs/2.13/quantization.html - the torchao centralization notice and
  the per-flow migration list, retrieved 2026-08-25.
- docs.pytorch.org/ao/stable/quick_start.html - `quantize_`, `Int4WeightOnlyConfig`,
  and the quantize-then-compile ordering.
- Sizing arithmetic in the walkthrough is computed from the stated bit widths.
- torchao 0.18.0 introspected on torch 2.15.0: `quantize_(model, config, filter_fn=_is_linear,
  device=None)`, `Int4WeightOnlyConfig(group_size=128, ...)`, the config names in
  `dir(torchao.quantization)`; walkthrough steps 1 to 3 executed there.
- github.com/pytorch/ao docs via Context7 (inference workflow table, first quantization
  example, int4 packing formats by device).
