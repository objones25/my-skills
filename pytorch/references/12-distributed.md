# Distributed training

DDP, FSDP2, DTensor, tensor and pipeline parallel, distributed checkpointing.

## Overview

Scaling out has one decision at the top: does one copy of the model plus its
gradients and optimizer state fit on one GPU? If yes, replicate it with
`DistributedDataParallel` and split the data. If no, shard the parameters themselves
with FSDP2. Everything else - tensor parallel, pipeline parallel, DTensor (a tensor whose
shards live on several ranks, described by a device mesh and a placement per mesh dimension) -
is what you add when even sharded data parallelism runs out of room or bandwidth.

One process per GPU, always. `DataParallel` (single process, multi-thread) is
GIL-bound and slower; do not use it.

## Recommended approach

The DDP skeleton. `torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`.

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    dist.init_process_group("nccl")                 # "gloo" on CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)               # BEFORE building the model
    # On 2.15, also pass device_id=torch.device("cuda", local_rank) to init_process_group:
    # the rewritten nccl binds each group to that device and ignores set_device.

    model = MyNet().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(train_ds, shuffle=True)
    loader = DataLoader(train_ds, batch_size=64, sampler=sampler,  # NOT shuffle=True
                        num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)                    # or every rank sees the same order
        for x, y in loader:
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()         # DDP all-reduces grads here
            optimizer.step()

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "ckpt.pth")   # .module unwraps DDP
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

FSDP2 replaces the DDP wrapper with an in-place transform, applied inner-modules-first:

```python
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
for block in model.layers:
    fully_shard(block, mesh=mesh)     # shard each block first
fully_shard(model, mesh=mesh)         # then the root
# model is now an FSDPModule; the training loop is unchanged
```

## API reference

Signatures introspected from torch 2.15.0.

| API | Form |
|-----|------|
| Init | `dist.init_process_group(backend, ...)` - `"nccl"` GPU, `"gloo"` CPU, `"xccl"` XPU |
| Identity | `dist.get_rank()`, `dist.get_world_size()`, `os.environ["LOCAL_RANK"]` |
| Teardown | `dist.destroy_process_group()` |
| DDP | `DistributedDataParallel(module, device_ids=None, ..., find_unused_parameters=False, gradient_as_bucket_view=False, static_graph=False, device_mesh=None, ..., lazy_bucket_allocation=False)`; `lazy_bucket_allocation` and `ddp.finalize_backward()` are new in 2.15; `finalize_backward()` raises unless `ddp.require_manual_backward_finalization = True` is set first (reproduced) |
| Skip sync | `with model.no_sync(): ...` - for gradient accumulation micro-steps |
| FSDP2 | `fully_shard(module, *, mesh=None, reshard_after_forward=None, shard_placement_fn=None, mp_policy=MixedPrecisionPolicy(), offload_policy=OffloadPolicy(), ignored_params=None, dp_mesh_dims=None)` |
| Mixed precision | `MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True, *, param_dtype_override_fn=None)`; the last is 2.15 |
| Mesh | `init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None, backend_override=None)` |
| DTensor | `distribute_tensor(t, mesh, [Shard(0)])`; placements `Shard(dim)`, `Replicate()`, `Partial()` |
| Tensor parallel | `parallelize_module(module, device_mesh, parallelize_plan)` with `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`, `PrepareModuleInput` |
| Pipeline | `torch.distributed.pipelining` - `pipeline`, `PipelineStage`, `Schedule1F1B`, `ScheduleGPipe` |
| Sharded checkpoint | `dcp.save(state_dict, checkpoint_id=...)`, `dcp.load(...)`, `dcp.async_save(...)` |
| HF format | `dcp.HuggingFaceStorageReader` / `HuggingFaceStorageWriter` |

Collectives. **Renamed in 2.13** - the old names still work; they emit `FutureWarning` on
2.13 and 2.14 and are silent aliases on 2.15 (reproduced; `tests/test_claims.py`):

| Use | 2.13 name | Old name |
|-----|-----------|----------|
| Gather shards into one tensor | `dist.all_gather_single` | `all_gather_into_tensor` |
| Reduce and scatter | `dist.reduce_scatter_single` | `reduce_scatter_tensor` |
| Sum across ranks | `dist.all_reduce(t, op=dist.ReduceOp.SUM)` | unchanged |
| Rank 0 to all | `dist.broadcast(t, src=0)` | unchanged |
| Sync all ranks | `dist.barrier()` | unchanged |

`FSDPModule` runtime controls (the methods `fully_shard` adds):

| Method | Use |
|--------|-----|
| `set_requires_gradient_sync(False)` | skip reduce-scatter during accumulation micro-steps |
| `set_separate_reduce_scatter_group(True)` | **new in 2.13** - dedicated communicator so reduce-scatter overlaps all-gather |
| `set_reshard_after_forward(bool)` | trade memory for one less all-gather in backward |
| `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` | manual prefetch tuning |
| `unshard()` / `reshard()` | materialize or release full parameters manually |
| `set_gradient_divide_factor(f)` | custom gradient scaling across the group |
| `set_manual_backward_finalization(True)` + `finalize_backward(async_op=False)` | **2.15** - you decide when backward is finished (custom backward loops, CUDA-graph capture of a whole accumulation step) |
| `set_keep_unsharded_storage(...)` | **2.15** - keep the unsharded buffer allocated between uses |

## Walkthrough

Two halves. First a real process group and a real DDP wrapper, run as a one-process
"world" on CPU with the `gloo` backend, so every line executes here and in `pytest`.
Then the sizing arithmetic that decides DDP versus FSDP2. The multi-process version of
the same program is `assets/ddp_template.py`, which `tests/test_ddp_template.py`
launches with two CPU ranks under `torchrun`. Output executed on torch 2.15.0.

**1. A process group without `torchrun`.** `torchrun` normally supplies the rendezvous
(how ranks find each other). A `HashStore` is an in-memory key-value store, enough for
a world of one, and needs no network port.

```python
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
print("rank", dist.get_rank(), "of", dist.get_world_size(), "| backend:", dist.get_backend())
```
```
rank 0 of 1 | backend: gloo
```

**2. DDP prefixes every state-dict key with `module.`** This is why checkpoints are saved
from `model.module`: a plain model loading the wrapped keys fails with "Missing key(s)".

```python
torch.manual_seed(0)
model = DDP(nn.Linear(4, 2))
print("wrapped keys:  ", list(model.state_dict()))
print("unwrapped keys:", list(model.module.state_dict()))
```
```
wrapped keys:   ['module.weight', 'module.bias']
unwrapped keys: ['weight', 'bias']
```

**3. Collectives work in place on every rank.** With one rank the sum is the input; with
N ranks each rank ends holding the same sum. `all_gather_single` (the 2.13 name for
`all_gather_into_tensor`) fills a preallocated output with every rank's shard.

```python
t = torch.tensor([1.0, 2.0])
dist.all_reduce(t, op=dist.ReduceOp.SUM)
out = torch.empty(2)
dist.all_gather_single(out, torch.tensor([3.0, 4.0]))
print("all_reduce:", t.tolist(), "| all_gather_single:", out.tolist())
dist.destroy_process_group()
```
```
all_reduce: [1.0, 2.0] | all_gather_single: [3.0, 4.0]
```

**4. Count what one rank must hold** for a 7B-parameter model trained with AdamW under
bf16 autocast. Per parameter: 4 bytes fp32 weights (autocast casts to bf16 on the fly,
so no persistent bf16 copy), 4 bytes fp32 gradient, and 8 bytes of Adam moments. The
ZeRO layout (from the DeepSpeed ZeRO paper: bf16 weights and grads plus an fp32 master
copy and fp32 Adam state) also totals 16.
`python scripts/memory_budget.py --params 7e9 --optimizer adamw --world 8` prints the same
table for other optimizers and world sizes.

```python
params = 7e9
per_param = 4 + 4 + 8                # bytes: fp32 weight, fp32 grad, 2 fp32 Adam moments
print(f"model+optimizer state: {params * per_param / 2**30:.0f} GiB per rank (DDP)")
```
```
model+optimizer state: 104 GiB per rank (DDP)
```

That does not fit an 80 GiB H100, before a single activation. DDP is out.

**5. FSDP2 shards all three across the world.** With 8 ranks each holds 1/8 of the
parameters, gradients, and optimizer state.

```python
world = 8
print(f"sharded state: {params * per_param / world / 2**30:.1f} GiB per rank")
```
```
sharded state: 13.0 GiB per rank
```

Now activations have room, and gradient checkpointing (`13-performance-profiling.md`)
buys the rest. What the sharding costs: FSDP2 all-gathers each layer's parameters just
before using it and frees them right after, so it trades communication for memory. The
volume per step is roughly `2 * params * b_param` bytes for the forward and backward
all-gathers plus `params * b_reduce` for the reduce-scatter, where `b_param` is the bytes
per element of `param_dtype` and `b_reduce` of the reduce dtype. With fp32 parameters and
`param_dtype=torch.bfloat16`, that is 2 bytes for the all-gathers, and on 2.15 4 bytes for
the reduce-scatter unless `reduce_dtype` is set (2 on 2.14). If that exceeds what your
interconnect moves in the time the layer takes to compute, you are bandwidth-bound and
`set_separate_reduce_scatter_group(True)` or a larger `reshard_after_forward` window is
where to start.

## Gotchas

- **Use `DistributedSampler`, not `shuffle=True`**, and call `sampler.set_epoch(epoch)`
  every epoch, or every rank sees the same order every epoch.
- **`torch.cuda.set_device(local_rank)` before building the model**, or every rank
  allocates on GPU 0. On 2.15 also pass `device_id=` to `init_process_group`: the rewritten
  `nccl` binds each group to `device_id`, else `LOCAL_RANK % torch.cuda.device_count()`,
  and does not consult `set_device` (release notes; not run here).
- **The per-rank `batch_size` is per rank.** Global batch is `batch_size *
  world_size`; retune the LR accordingly (`06-tuning-reproducibility.md`).
- **Save from rank 0 only, and unwrap: `model.module.state_dict()`** for DDP. For
  FSDP2 the state dict holds DTensors - use `torch.distributed.checkpoint`, not
  `torch.save`, unless you explicitly gather to full tensors first.
- **`find_unused_parameters=True` is a real slowdown.** It exists for models where
  some parameters do not participate every step; if you do not need it, leave it off.
  If DDP hangs at backward, that is the usual reason - but fix the model first.
- **Under gradient accumulation, wrap micro-steps in `model.no_sync()`** (DDP) or
  `set_requires_gradient_sync(False)` (FSDP2), or you all-reduce on every micro-batch.
- **Any rank-conditional code that skips a collective deadlocks.** Logging behind
  `if rank == 0` is fine; a `barrier()` or `all_reduce` behind it is not.
- **`fully_shard` is applied inner-first, then the root.** Sharding only the root
  gives you one giant all-gather and no overlap.
- **FSDP1 `FullyShardedDataParallel` still exists but FSDP2 `fully_shard` is the
  current API** - DTensor-based, composable with tensor parallel, and where new work
  lands.
- **Always `dist.destroy_process_group()`**, including on the error path, or ranks
  hang at exit.
- **FSDP2 gradient dtype changed in 2.15.** With `MixedPrecisionPolicy(reduce_dtype=None)`,
  gradients now accumulate and reduce-scatter in each parameter's `grad_dtype`, which
  defaults to the parameter's original dtype, instead of in `param_dtype`. The common case,
  fp32 parameters with `param_dtype=torch.bfloat16`, reduced in bf16 on 2.14 and reduces in
  fp32 on 2.15: better numerics, twice the reduce-scatter bytes. To keep the 2.14 behavior,
  set `reduce_dtype=torch.bfloat16` explicitly (release notes, PR 194434; reproduced on
  CPU with 2 gloo ranks: the reduce-scatter ran in bf16 on 2.14.1, fp32 on 2.15.0, and bf16
  on 2.15.0 with `reduce_dtype=torch.bfloat16`). `param_dtype_override_fn` (2.15) keeps chosen parameters, such as MoE router
  gates or norm weights, in fp32 while the rest computes in bf16.
- **Distributed checkpoints written by 2.15 do not load in 2.14 or earlier** (they record
  strides). Upgrade every reader first.
- **Debug with `TORCH_DISTRIBUTED_DEBUG=DETAIL`** and `NCCL_DEBUG=INFO`; a hang is
  almost always a collective that not every rank reached.

## Alternatives

**Pick the parallelism by what ran out:**

| Constraint | Strategy |
|------------|----------|
| Model fits on one GPU, want more throughput | DDP |
| Model + optimizer state does not fit | FSDP2 `fully_shard` |
| A single layer does not fit, or activations dominate | tensor parallel (`parallelize_module`) |
| Very deep model, many nodes, limited interconnect | pipeline parallel (`distributed.pipelining`) |
| All of the above at scale | n-D mesh: DP x TP x PP over one `DeviceMesh` |

An n-D mesh composes them. Name the dimensions once and each strategy takes the slice
it needs:

```python
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
parallelize_module(model, mesh["tp"], {"attn.qkv": ColwiseParallel(),
                                       "attn.proj": RowwiseParallel()})
fully_shard(model, mesh=mesh["dp"])
```

**Checkpointing at scale.** `torch.save` of a full state dict requires gathering every
parameter onto one rank - slow, and eventually impossible.
`torch.distributed.checkpoint` writes each rank's shard in parallel and reloads into
a *different* world size, which is what makes elastic restarts work.
`dcp.async_save` overlaps the write with the next training steps.

```python
import torch.distributed.checkpoint as dcp
state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
dcp.save(state, checkpoint_id="ckpt/step_1000")
dcp.load(state, checkpoint_id="ckpt/step_1000")   # load into the SAME dict object
```

**The `nccl` backend was rewritten in 2.15.** torchcomms arrived in 2.13, landed in 2.14
as the opt-in `nccl2` backend, and in 2.15 became what `"nccl"` means. Communicators are
created eagerly with the process group, each group is bound to one CUDA device, and
the object returned by `dist.get_backend_impl(...)` is no longer a `ProcessGroupNCCL`. To get the old backend set
`TORCH_DIST_USE_NCCL2=0` or pass `"nccl-legacy"`; the release notes say both go away in
the next release. A backend-agnostic Flight Recorder writes a post-mortem to disk when a
collective hangs. None of this was run here (no GPUs); it is from the 2.15 release notes.

**When not to go distributed at all:** gradient accumulation, AMP, activation
checkpointing, and a smaller batch solve a surprising number of "I need more GPUs"
problems, at zero coordination cost. Exhaust `13-performance-profiling.md` first.

## Related

- `05-data-training-loop.md` - the loop DDP and FSDP wrap.
- `13-performance-profiling.md` - single-GPU memory reduction to try first.
- `11-compile-export.md` - compiling under DDP/FSDP.
- `01-devices-amp.md` - `MixedPrecisionPolicy` mirrors the autocast dtype choice.
- `10-debugging.md` - hangs, and the renamed-collective `FutureWarning` (2.13, 2.14).

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) (`fully_shard`, `DistributedDataParallel`,
  `init_device_mesh`, `parallelize_module`, `dcp.save`/`load`, `FSDPModule` methods);
  collective renames confirmed by attribute check.
- docs.pytorch.org/docs/2.13 - `distributed.html`, `distributed.fsdp.fully_shard.html`,
  `distributed.tensor.html`, `distributed.checkpoint.html`.
- PyTorch 2.15 release notes (github.com/pytorch/pytorch/releases/tag/v2.15.0) - rewritten
  `nccl`, `nccl-legacy`, FSDP2 `reduce_dtype` change, `finalize_backward`,
  `param_dtype_override_fn`, DCP stride metadata. Walkthrough steps 1 to 3 executed on
  2.15.0 and 2.14.1 (gloo, one rank); `assets/ddp_template.py` run with two gloo ranks.
- PyTorch 2.13 release blog - torchcomms, `set_separate_reduce_scatter_group`,
  collective renaming to the `_single` scheme.
