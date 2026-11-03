# Performance, profiling, and memory

## Overview

Optimization has an order, and skipping it wastes days. Measure first, because the
bottleneck is usually not where it feels like it is - most "slow GPU training" is a
starved data pipeline. Then fix the cheap things (AMP, DataLoader workers, fewer
syncs), then the structural ones (compile, checkpointing, sharding). This file covers
the measurement tools and the memory levers.

## Recommended approach

Before changing anything, answer one question: is the GPU busy? If `nvidia-smi`
utilization sits low, the model is not the problem - the input pipeline or a
synchronization point is.

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),   # skip warmup steps
    on_trace_ready=tensorboard_trace_handler("./log"),
    record_shapes=True, profile_memory=True, with_stack=True,
) as prof:
    for step, (x, y) in enumerate(loader):
        train_step(x, y)
        prof.step()                                # advance the schedule
        if step >= 6:
            break

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
```

Then read the trace: long gaps between kernels mean data-bound or sync-bound;
back-to-back small kernels mean launch-bound (compile helps); one huge kernel means
you are actually compute-bound and only algorithmic change helps.

## API reference

| Tool | Call | Answers |
|------|------|---------|
| Profiler | `torch.profiler.profile(activities=..., schedule=..., on_trace_ready=..., record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False)`; 2.14 deprecates `with_modules`, removes `use_cuda`, and hides Python function events from `key_averages()`; 2.15 renames the CUPTI monitor to Cuspy (`torch.profiler.CuspyConfig`) | where the time goes |
| Profiler schedule | `schedule(*, wait, warmup, active, repeat=0, skip_first=0)` | which steps to record |
| Trace export | `prof.export_chrome_trace("trace.json")` | timeline in `chrome://tracing` or Perfetto |
| Named region | `with record_function("my_block"): ...` | attribute time to your own labels |
| Microbenchmark | `torch.utils.benchmark.Timer(stmt=..., globals=...).timeit(n)` | correct timing, warmup and sync handled |
| Compare variants | `torch.utils.benchmark.Compare([...])` | A/B table |
| Theoretical FLOPs | `torch.utils.flop_counter.FlopCounterMode` | arithmetic intensity, MFU (Model FLOPs Utilization: achieved FLOP/s over the device's peak) |
| Per-module attribution | `torch.utils.module_tracker.ModuleTracker` | which module an op belongs to |
| Peak memory | `torch.accelerator.max_memory_allocated()` | size the batch |
| Memory timeline | `torch.cuda.memory._record_memory_history()` then `_dump_snapshot("s.pickle")` | view at pytorch.org/memory_viz |
| Activation checkpointing | `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)` | trade compute for memory |
| Reset stats | `torch.accelerator.reset_peak_memory_stats()` | between measurements |

**Never time with `time.time()` around GPU code** without `torch.cuda.synchronize()`
- kernel launches are asynchronous, so you will measure the launch, not the work.
`torch.utils.benchmark.Timer` handles warmup and synchronization for you.

## Walkthrough

Profile a small MLP, then time it properly, then count its FLOPs. Output executed on
torch 2.15.0 (CPU box - on GPU add `ProfilerActivity.CUDA` and sort by CUDA time).

**1. Profile with a named region.**

```python
import torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
x = torch.randn(64, 256)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("fwd"):
        for _ in range(5):
            model(x)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
```
```
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::addmm        41.02%       1.703ms        47.16%       1.958ms     195.818us            10
                   fwd        39.60%       1.644ms       100.00%       4.152ms       4.152ms             1
       aten::clamp_min         6.52%     270.513us         6.52%     270.513us      54.103us             5
           aten::copy_         5.07%     210.444us         5.07%     210.444us      21.044us            10
       aten::transpose         1.88%      78.143us         2.26%      93.837us       9.384us            10
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 4.152ms
```

`aten::addmm` is the two `Linear` layers (10 calls = 2 layers x 5 iterations),
`aten::clamp_min` is the ReLU. The matmuls dominate, as they should.

**2. Time it properly.** `Timer` warms up and synchronizes; a bare `time.time()` loop
would not.

```python
import torch.utils.benchmark as bench
t = bench.Timer(stmt="m(x)", globals={"m": model, "x": x}).timeit(50)
print(f"median {t.median * 1e6:.1f} us")
```
```
median 396.7 us
```

**3. Count the FLOPs** to know whether that time is reasonable.

```python
from torch.utils.flop_counter import FlopCounterMode
with FlopCounterMode(display=False) as fc:
    model(x)
print("FLOPs:", f"{fc.get_total_flops():,}")
```
```
FLOPs: 33,554,432
```

Check the arithmetic: `2 * 64 * 256 * 512` for the first layer plus the same for the
second is `2 * 2 * 64 * 256 * 512 = 33,554,432`. The factor of 2 is one multiply plus
one add per MAC. Dividing FLOPs by measured seconds gives achieved FLOP/s, and
dividing that by the device's peak gives Model FLOPs Utilization - the number that
tells you whether an optimization is worth pursuing. Below about 20% MFU there is
usually easy headroom; above 50% you are near the hardware and only algorithmic
change helps.

**4. Activation checkpointing** recomputes the forward during backward instead of
storing it.

```python
from torch.utils.checkpoint import checkpoint
y = checkpoint(lambda t: model(t), x.requires_grad_(), use_reentrant=False)
print("out:", tuple(y.shape), "| requires_grad:", y.requires_grad)
```
```
out: (64, 256) | requires_grad: True
```

Omitting `use_reentrant` still runs on 2.13 through 2.15, but warns and falls back to the
legacy reentrant path (the 2.14 docs and 2.14.1's warning said 2.9 would raise; 2.14.1 and
2.15.0 both only warn). Pass `use_reentrant=False`, the supported implementation.

## Gotchas

- **Timing GPU code without a sync measures the launch, not the work.** Use
  `torch.utils.benchmark.Timer`.
- **Profile steady state, not warmup.** Use `schedule(wait=, warmup=, active=)` and
  call `prof.step()` each iteration, or the first (compiling, autotuning, allocating)
  steps dominate the table.
- **`with_stack=True` and `record_shapes=True` add real overhead.** Turn them on when
  you need attribution, off when you need timing.
- **Every `.item()`, `.cpu()`, `print(tensor)`, and `assert` on a tensor forces a
  device sync** and serializes the pipeline. Log once per N steps, not per step.
- **`empty_cache()` does not free live tensors** - it returns the allocator's cached
  blocks to the driver. It makes `nvidia-smi` look better and usually makes training
  slower.
- **OOM in `backward` means activations, not weights.** Peak memory hits on the first
  backward, when activations, gradients, and optimizer state all coexist.
- **`cudnn.benchmark = True` re-autotunes on every new input shape** - a loss for
  variable-shape workloads, a win for fixed ones.
- **FLOP counters report theoretical FLOPs**, not what the hardware executed. Use them
  for MFU and for comparing architectures, not as a timing proxy.
- **Do not tune anything with `num_workers=0`** unless you have proven the pipeline is
  not the bottleneck - you will optimize the wrong half of the program.

## Alternatives

**The optimization ladder, cheapest first:**

| Step | Typical gain | Cost |
|------|--------------|------|
| `num_workers>0`, `pin_memory=True`, `persistent_workers=True` | large if data-bound | none |
| Remove per-step `.item()` / `print` syncs | moderate | none |
| bf16 autocast | up to ~2x on Tensor Cores, halves activation memory | none |
| `torch.set_float32_matmul_precision("high")` (TF32) | notable on fp32 matmuls | tiny precision loss |
| `torch.compile(model)` | 1.3-2x, more if launch-bound | compile time |
| Larger batch (once memory allows) | better utilization | retune LR |
| Gradient accumulation | enables a larger effective batch | proportional slowdown |
| Activation checkpointing | large memory saving | roughly +30% compute |
| Fused optimizer (`AdamW(..., fused=True)`) | small, real | mps, cuda, xpu, hpu, cpu, mtia, privateuseone on 2.15.0 (`torch.utils._foreach_utils._get_fused_kernels_supported_devices()`) |
| `nn.LinearCrossEntropyLoss` for large vocab | up to ~4x peak memory | none |
| FSDP2 / tensor parallel | fits what did not fit | coordination, bandwidth |

**Memory reduction, in order:** smaller batch plus accumulation -> AMP -> log
`loss.item()` not `loss` -> `@torch.inference_mode()` at eval -> activation
checkpointing -> optimizer-state sharding (FSDP2) or a factored optimizer
(`Adafactor`). Measure each with
`torch.accelerator.max_memory_allocated()` after `reset_peak_memory_stats()`, not by
watching `nvidia-smi`, which shows the allocator cache.

**Finding a leak.** A real leak shows `memory_allocated()` climbing every iteration
while `memory_reserved()` climbs with it. The usual causes are a live loss tensor or
a list of activations kept across steps. `torch.cuda.memory._record_memory_history()`
plus `_dump_snapshot()` produces a snapshot you can open at pytorch.org/memory_viz to
see exactly which allocation grew.

**The 2.13 profiler additions:** an experimental CUPTI monitor backend (renamed Cuspy in
2.15, configured with `CuspyConfig`; it can attribute kernels replayed from CUDA graphs to
the code that captured them) collects GPU
metrics asynchronously and off the GIL, so profiling perturbs a production loop far
less than the activity-API path; and `CUDAGraph.get_graph_data()` exposes a captured
graph's topology (node types, kernel names, dependency edges) with IDs that line up
with profiler output.

## Related

- `05-data-training-loop.md` - DataLoader knobs, where the syncs live.
- `01-devices-amp.md` - autocast dtype and memory accounting APIs.
- `11-compile-export.md` - what `torch.compile` can and cannot speed up.
- `12-distributed.md` - when single-GPU optimization runs out.
- `10-debugging.md` - OOM and slow-training symptom entries.

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) (`profiler.profile`, `profiler.schedule`,
  `utils.checkpoint.checkpoint`); all walkthrough output executed on the same build.
- docs.pytorch.org/docs/2.13 - `profiler.html`, `benchmark_utils.html`,
  `checkpoint.html`, `torch_cuda_memory.html`, `utils.html` (flop_counter,
  module_tracker).
- PyTorch 2.13 release blog - CUPTI monitor profiler, `CUDAGraph.get_graph_data()`.
