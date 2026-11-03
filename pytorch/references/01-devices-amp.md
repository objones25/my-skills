# Devices and mixed precision

CPU / CUDA / MPS / XPU placement, `.to()` semantics, pinned memory, autocast, memory accounting.

## Overview

A tensor lives on exactly one device and an op runs where its inputs live. "Using
the GPU" therefore reduces to two rules: put the model and each batch on the same
device before the forward pass, and never copy across the PCIe bus inside a hot
loop. Since 2.6 the `torch.accelerator` namespace selects and queries whatever
accelerator the build supports through one device-agnostic API, so the old
`if cuda ... elif mps ...` ladder is obsolete.

## Recommended approach

Pick the device once, move the model in place, move each batch by reassignment, and
wrap only the forward plus loss in autocast.

```python
import torch

device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

model = MyNet().to(device)                        # modules move IN PLACE
loader = DataLoader(ds, batch_size=64, pin_memory=True)

for x, y in loader:
    x = x.to(device, non_blocking=True)           # tensors move by COPY - reassign
    y = y.to(device, non_blocking=True)
    optimizer.zero_grad()
    with torch.autocast(device.type, dtype=torch.bfloat16):   # forward + loss only
        loss = loss_fn(model(x), y)
    loss.backward()                               # backward OUTSIDE autocast
    optimizer.step()
```

Why these choices: `current_accelerator(check_available=True)` returns a
`torch.device` for the live accelerator or `None`, so `or torch.device("cpu")` gives
one line that runs unmodified on a laptop and a GPU box. `pin_memory=True` plus
`non_blocking=True` overlaps the host-to-device copy with compute. `bfloat16`
autocast gets Tensor-Core throughput with no `GradScaler`. Backward stays outside
the `autocast` block because backward ops reuse the dtype autocast picked for the
matching forward op.

## API reference

Signatures introspected from torch 2.15.0.

| API | Signature | Returns / effect |
|-----|-----------|------------------|
| Pick accelerator | `torch.accelerator.current_accelerator(check_available=False)` | `torch.device` or `None`; index is `None` |
| Accelerator present | `torch.accelerator.is_available()` | `bool` |
| Count | `torch.accelerator.device_count()` | `int` |
| Current index | `torch.accelerator.current_device_index()` | `int` |
| Free / total bytes | `torch.accelerator.get_memory_info(device_index=None, /)` | `tuple[int, int]` |
| Peak allocated | `torch.accelerator.max_memory_allocated(device_index=None, /)` | bytes |
| Reserved | `torch.accelerator.memory_reserved(device_index=None, /)` | bytes (allocator cache) |
| Free cache | `torch.accelerator.empty_cache()` | returns cached blocks to the driver |
| Graph capture | `torch.accelerator.Graph` | device-agnostic graph capture (2.12+) |
| Move tensor | `tensor.to(device, dtype=None, non_blocking=False)` | **new** tensor; returns `self` if device+dtype already match |
| Move module | `module.to(device)` | same module, moved **in place**; returns `self` |
| Register state | `module.register_buffer(name, tensor)` | non-parameter tensor that follows `.to()` |
| Create on device | `torch.randn(*shape, device=device)` | allocated there, no copy |
| Global default | `torch.set_default_device(device)` | later factory calls default to it |
| Autocast | `torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)` | context manager / decorator |
| fp16 scaler | `torch.amp.GradScaler(device="cuda", init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)` | scaler object |
| Autocast supported? | `torch.amp.is_autocast_available(device_type)` | `bool` |

`current_accelerator()` returns a device whose `index` is `None`. When you need the
integer (to log which GPU), call `current_device_index()`. `check_available=True`
adds a runtime `is_available()` check on top of the compile-time one; without it the
call reports what the build *could* use, not what is live.

## Walkthrough

A 3-feature to 1-logit MLP on a batch of 4, moved to the device and run under
autocast. All output below was executed on torch 2.15.0 (CPU box - on a CUDA box the
device strings read `cuda`).

**1. Pick the device.**

```python
import torch
device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
print("device:", device, "| type:", device.type)
```
```
device: cpu | type: cpu
```

**2. Build the model and move it in place.** `.to()` on a module mutates and returns
it, so no reassignment.

```python
model = torch.nn.Sequential(
    torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1),
)
model.to(device)
print("param device:", next(model.parameters()).device)
```
```
param device: cpu
```

**3. Copy versus in-place, for tensors.** A tensor `.to()` returns a *new* tensor. On
CPU a device move is a no-op, so the copy semantics show up with a dtype cast, which
always allocates.

```python
x = torch.randn(4, 3)                 # (4, 3) float32
y = torch.randint(0, 2, (4, 1)).float()   # (4, 1) float targets for BCE
x_half = x.to(torch.float16)          # dtype differs -> NEW tensor
print("x is x_half:", x is x_half, "|", x.dtype, x_half.dtype)
```
```
x is x_half: False | torch.float32 torch.float16
```

`x.to(device)` on its own discards the moved tensor. For tensors always write
`x = x.to(device)`; modules are the exception.

**4. Autocast the forward and loss.** Matmul-heavy ops (`Linear`) emit the autocast
dtype; the loss reduction is forced back to float32.

```python
loss_fn = torch.nn.BCEWithLogitsLoss()
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    pred = model(x.to(device))
    loss = loss_fn(pred, y.to(device))
print("pred:", tuple(pred.shape), pred.dtype, "| loss dtype:", loss.dtype)
print("bf16 autocast available:", torch.amp.is_autocast_available(device.type))
```
```
pred: (4, 1) torch.bfloat16 | loss dtype: torch.float32
bf16 autocast available: True
```

The Autocast Op Reference puts `linear` in the lower-precision set and
`binary_cross_entropy_with_logits` in the fp32 set, so the `Linear` output is
`bfloat16` while the loss is `float32`. Fast matmuls, safe reductions - that split is
the whole point of autocast.

**5. Backward and step, outside autocast.**

```python
loss.backward()                       # gradients flow in the autocast-chosen dtypes
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.step()
optimizer.zero_grad()                 # set_to_none=True by default
```

## Gotchas

- **`x.to(device)` without reassignment is a no-op for tensors.** Modules are the
  exception. A dropped move leaves `x` on the CPU and surfaces as a device mismatch
  at the next matmul.
- **Unregistered tensors do not follow `model.to()`.** `.to()` walks `parameters()`
  and `buffers()` only. `self.scale = torch.ones(3)` stays on the CPU; use
  `self.register_buffer("scale", torch.ones(3))`.
- **`non_blocking=True` needs `pin_memory=True`.** Without pinned source memory the
  async copy silently falls back to blocking - no error, no speedup. Reading the
  tensor on the CPU right after (`.item()`, `print`) forces a sync that erases the
  overlap anyway.
- **`GradScaler` is float16-only.** With `bfloat16` it is pure overhead. The
  `torch.cuda.amp.*` namespace is deprecated; use `torch.amp.GradScaler("cuda")` and
  `torch.autocast("cuda", ...)`.
- **High `nvidia-smi` usage is usually the caching allocator, not a leak.** Compare
  `memory_allocated()` (live) against `memory_reserved()` (cache). A real leak shows
  `memory_allocated()` climbing every iteration.
- **Appending a live `loss` tensor to a list keeps its whole graph alive.** Log
  `loss.item()`.
- **`.cuda()` hard-codes CUDA** and breaks on MPS/CPU. Thread a `device` variable.
- **MPS gaps:** some ops are unimplemented and fall back to CPU (set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`); `float64` is unsupported. `GradScaler` crashed on MPS in 2.5 (it cast the scale to
  float64); the 2.15 source still special-cases `mps` (`amp/grad_scaler.py`), but this skill has not verified fp16 +
  `GradScaler("mps")` on Apple hardware, so test it before relying on it.
- **Tensors created inside `forward` default to CPU.** Pass `device=x.device`.
- **MPS needs macOS 15 or newer as of torch 2.15** (release notes; macOS 14 workarounds were
  removed). 2.15 also adds a public MPS stream API (`torch.mps.Stream`, `torch.mps.stream`).
- **Device-agnostic RNG state lives in `torch.accelerator.random`** (`get_rng_state`,
  `initial_seed`, 2.14+), not on `torch.accelerator`. `get_rng_state` and `initial_seed`
  raise `RuntimeError` on a machine with no accelerator (`get_rng_state_all` returns `[]`). Guard them with `torch.accelerator.is_available()`.

## Alternatives

**Device selection.** `torch.accelerator.current_accelerator()` (2.6+) versus the
classic ladder `"cuda" if torch.cuda.is_available() else "mps" if
torch.backends.mps.is_available() else "cpu"`. Keep the ladder only for older torch.

**Autocast dtype by device:**

| Device | Recommended dtype | GradScaler? |
|--------|-------------------|-------------|
| CUDA, compute capability >= 8.0 (Ampere and newer) | `bfloat16` | no |
| CUDA, capability 7.x (Volta/Turing) | `float16` | **yes** |
| CPU | `bfloat16` | no |
| MPS (Apple) | `float16` | fp16 needs one; `GradScaler("mps")` has a code path, unverified here |

`bfloat16` has the same exponent range as fp32, so it cannot underflow the way fp16
does - that is why it needs no loss scaling.

**Memory stats.** Prefer `torch.accelerator.memory.*` (device-agnostic) over
`torch.cuda.*` in new code, so the same line works on MPS/XPU.

**Multi-GPU.** `DistributedDataParallel` when the model fits on one GPU;
FSDP2 `fully_shard` when it does not; never `DataParallel` (single-process,
GIL-bound, slower). See `12-distributed.md`.

**Matmul precision.** On Ampere+, `torch.set_float32_matmul_precision("high")` lets
fp32 matmuls use TF32 tensor cores - a large speedup for a small precision cost, and
independent of autocast.

## Related

- `03-models-layers.md` - buffers versus parameters decide what `.to()` moves.
- `05-data-training-loop.md` - `pin_memory` is set on the DataLoader; the in-loop
  `.to(device, non_blocking=True)` pairs with it.
- `13-performance-profiling.md` - measuring whether the transfer actually overlaps.
- `10-debugging.md` - device-mismatch and OOM symptom index.

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) (`torch.accelerator.*`, `torch.autocast`,
  `torch.amp.GradScaler`).
- PyTorch 2.15 release notes (macOS 15 requirement, MPS streams).
- docs.pytorch.org/docs/2.13 - `amp.html` (autocast signature and Op Reference),
  `notes/amp_examples.html`, `accelerator.html`.
