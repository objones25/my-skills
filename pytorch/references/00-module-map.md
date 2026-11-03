# Module map, 2.13 to 2.15 changes

Verified against torch 2.15.0 / torchvision 0.29.2 by introspection, and re-checked on
torch 2.14.1 / torchvision 0.29.1. Run `python scripts/verify_api.py --docs` to re-check
every name in these references against your install.

## Which submodule owns what

| Submodule | Owns | Reach for it when |
|-----------|------|-------------------|
| `torch` | tensors, factories, math, `save`/`load`, `compile` | always |
| `torch.nn` | layers, losses, containers - stateful `nn.Module` objects | building models |
| `torch.nn.functional` (`F`) | stateless op forms (`F.relu`, `F.cross_entropy`) | inside `forward`, no params |
| `torch.nn.init` | weight initializers (`kaiming_normal_`, `trunc_normal_`) | overriding default init |
| `torch.nn.attention` | `sdpa_kernel`, `SDPBackend`, `flex_attention`, `varlen`, `bias` | attention kernels and masks |
| `torch.autograd` | backward engine, `grad`, `no_grad`, anomaly detection | custom grads, grad debugging |
| `torch.optim` | optimizers + `lr_scheduler` | training |
| `torch.optim.functional` | stateless update functions (`adamw`, `sgd`, `muon`, ...), public in 2.15 | writing your own optimizer loop or a functional training step |
| `torch.utils.data` | `Dataset`, `DataLoader`, samplers, `default_collate` | feeding the loop |
| `torch.amp` | `autocast`, `GradScaler`, `is_autocast_available` | mixed precision |
| `torch.accelerator` | device-agnostic selection, memory, streams, `Graph` | any device code (2.6+) |
| `torch.cuda` / `torch.mps` / `torch.xpu` / `torch.cpu` / `torch.mtia` | per-backend specifics | only when the accelerator API lacks it |
| `torch.compiler` | `compile`, `disable`, `set_stance`, `set_default_backend`, cache artifacts, `precompile` (a module in 2.15: `capture`, `load`) | compilation control |
| `torch.export` | `export`, `Dim`, `ExportedProgram`, `draft_export` | AOT capture for deployment |
| `torch.onnx` | ONNX export (`dynamo=True` path) | ONNX runtimes |
| `torch.distributed` | process groups, collectives, `device_mesh` | multi-process training |
| `torch.distributed.fsdp` | `fully_shard` (FSDP2), `FSDPModule`, legacy `FullyShardedDataParallel` | model too big for one GPU |
| `torch.distributed.tensor` | `DTensor`, `Shard`/`Replicate`/`Partial`, `distribute_tensor` | n-D parallelism |
| `torch.distributed.tensor.parallel` | `parallelize_module`, `ColwiseParallel`, `RowwiseParallel` | tensor parallel |
| `torch.distributed.pipelining` | pipeline schedules and stages | pipeline parallel |
| `torch.distributed.checkpoint` (`dcp`) | sharded `save`/`load`/`async_save`, HF storage readers | checkpointing sharded state |
| `torch.profiler` | `profile`, `schedule`, `record_function`, `tensorboard_trace_handler` | where the time goes |
| `torch.utils.benchmark` | `Timer`, `Compare` - warmup + sync handled | microbenchmarks |
| `torch.utils.flop_counter` | `FlopCounterMode` | theoretical FLOPs per module |
| `torch.utils.checkpoint` | activation checkpointing | trading compute for memory |
| `torch.utils.module_tracker` | which module a op ran under | attributing memory/time |
| `torch.func` | `vmap`, `grad`, `jacrev`, `functional_call` | per-sample grads, jacobians, ensembles |
| `torch.linalg` | decompositions, solves, norms | numerical linear algebra |
| `torch.fft`, `torch.special`, `torch.signal` | spectral, special functions, windows | signal/scientific work |
| `torch.distributions` | distributions, `rsample`, `kl_divergence` | VAEs, RL, probabilistic models |
| `torch.nested` | ragged/jagged tensors | variable-length batches without padding |
| `torch.sparse` | COO/CSR/BSR sparse tensors | sparse data or weights |
| `torch.masked` | masked tensors | NaN-aware reductions |
| `torch.fx` | symbolic tracing, graph rewriting | program transforms |
| `torch.library` | custom op registration, `opcheck` | binding your own kernels (see `15-custom-ops-autograd.md`) |
| `torch.testing` | `assert_close`, `make_tensor` | tests (see `16-testing-pytorch-code.md`) |
| `torch.hub` | loading models from GitHub repos | pretrained weights outside torchvision |
| `torch.ao.quantization` | legacy quantization - **migrating to torchao** | see `14-quantization.md` |
| `torchvision` | image datasets, models, `transforms.v2` | vision |
| `torchaudio` | audio I/O and transforms | audio |

`torchtext` is deprecated. Use Hugging Face `datasets`/`tokenizers` for text pipelines.
`torch.utils.tensorboard` needs the separate `tensorboard` package installed.

## What changed in 2.13

Verified from the 2.13 release notes and by introspection.

**Removed - will break code:**

- **Named tensors are gone.** `Tensor.names` and every associated API were
  hard-removed. `hasattr(torch.randn(2), "names")` is now `False`.
- The Bazel build was removed. CPython 3.13t dropped from the Linux binary matrix.

**Renamed (old names still work; they emit `FutureWarning` on 2.13 and 2.14 and are silent aliases on 2.15, reproduced):**

| Old | New |
|-----|-----|
| `dist.all_gather_into_tensor` | `dist.all_gather_single` |
| `dist.reduce_scatter_tensor` | `dist.reduce_scatter_single` |

**Added** (the release blog labels every one of these API Unstable):

| API | What it does |
|-----|--------------|
| `nn.LinearCrossEntropyLoss` / `F.linear_cross_entropy` | fuses the output projection with cross-entropy, chunking over the vocab dim; up to ~4x lower peak memory for large-vocabulary LMs |
| `torch.load("x.safetensors")` | reads safetensors natively, no extra library |
| `torch.compiler.set_default_backend` | process-wide default backend, like `set_default_dtype` |
| `FSDPModule.set_separate_reduce_scatter_group(True)` | dedicated communicator so reduce-scatter overlaps all-gather |
| FlexAttention on MPS | up to ~12x over SDPA on sparse masks; dense still favors SDPA |
| FlexAttention deterministic backward (CUDA) | opt in via `torch.use_deterministic_algorithms(True)` |
| CuTeDSL Inductor backend | second GPU codegen path alongside Triton for GEMM and RMSNorm |
| torchcomms | new distributed communications backend, better fault tolerance and tracing |
| Python 3.15 / 3.15t wheels | Linux only, via `download.pytorch.org`, not PyPI; `torch.compile` unsupported there (2.15 adds Dynamo support for 3.15) |

**Version-sensitive defaults people misquote:**

| API | Default, 2.13 to 2.15 | Common wrong belief |
|-----|-----------------|---------------------|
| `torch.optim.AdamW(weight_decay=...)` | `0.01` | "it's 0, you must set it" - false, AdamW decays by default |
| `torch.optim.Adam(weight_decay=...)` | `0` | - |
| `torch.export.export(strict=...)` | `False` | "strict tracing is the default" - no longer |
| `torch.load(weights_only=...)` | `None`, meaning `True` unless `pickle_module` is set | "you must pass it" - belt-and-suspenders only |
| `torch.utils.checkpoint.checkpoint(use_reentrant=...)` | `None`: warns and uses the reentrant path | "it raises since 2.9" - the 2.14 docs and the 2.14.1 warning text said 2.9 would raise, yet 2.14.1 and 2.15.0 both only warn; 2.15's text now says "In a future release". Pass `use_reentrant=False` |
| `Optimizer.zero_grad(set_to_none=...)` | `True` | "grads are zeroed, not freed" |
| `nn.MultiheadAttention(batch_first=...)` | `False` - expects `(T, B, E)` | "it's batch-first like everything else" |
| `v2.ToDtype(scale=...)` | `False` | "casting also rescales" - it does not |

Run `python scripts/verify_api.py` to re-check all of these against the installed version.
`tests/test_claims.py` pins the behavioral ones (the checkpoint warning, the tied-weight
count, the `F.batch_norm` default) so an upgrade that changes them fails a test.

## What changed in 2.14

Released 2026-09-02. Checked against the 2.14 release blog and by introspection of
2.14.1 (`scripts/verify_api.py` on 2.14.1 reports these as known changes).

| Change | What to do |
|--------|------------|
| `torch.compile`'s `shapes_spec` keyword renamed `dynamic_shapes` (the 2.13.0 parameter list pinned by the previous version of this skill has `shapes_spec`; 2.14.1 has `dynamic_shapes`) | use the new keyword; it matches `torch.export` |
| `dynamic_spec` / `ShapesSpec` declare dynamic dims once for compile, export and `make_fx` | they live in `torch.fx.experimental.dynamic_spec`, not `torch.export`; `torch.compile` wraps a `ParamsSpec` passed as `dynamic_shapes` into a `ShapesSpec` for you |
| Profiler: `use_cuda` removed, `with_modules` deprecated, Python function events hidden from `key_averages()` | drop `use_cuda`; expect shorter tables |
| `checkpoint()`'s signature lists `preserve_rng_state` and `respect_saved_tensors_hooks` explicitly (`preserve_rng_state` was accepted through `**kwargs` before), and `function` defaults to `None`, so `@checkpoint(use_reentrant=False)` works as a decorator (compared against the 2.13.0 list pinned by the previous version) | still pass `use_reentrant=False` |
| torchcomms lands in-tree as an opt-in `nccl2` c10d backend | it became the default `nccl` in 2.15 (below) |
| `linear_cross_entropy` loses the `balanced` accuracy policy, gains probability targets | pin the version if you rely on its options |
| Device-agnostic RNG: `initial_seed`, `get_rng_state`, `get_rng_state_all` | they are in `torch.accelerator.random`, not on `torch.accelerator` itself; `initial_seed` and `get_rng_state` raise `RuntimeError` when no accelerator is present, `get_rng_state_all` returns `[]` |
| torchvision 0.29 is ABI-stable against torch 2.14 | torchvision 0.29.2 declares `torch>=2.14.0` and is the release used with torch 2.15 here |

## What changed in 2.15

Released 2026-10-28 (dev-discuss key dates post). Checked against the GitHub release
notes for v2.15.0 and by introspection of the installed 2.15.0 CPU build. Items marked
"not run here" need CUDA or several GPUs and were taken from the notes only.

**Breaks code:**

| Change | What to do |
|--------|------------|
| Python 3.10 dropped; 3.11 to 3.15 supported, including free-threaded 3.14t and 3.15t | upgrade the interpreter before upgrading torch |
| `nccl` now means the rewritten backend (ported from torchcomms); groups bind to one CUDA device (`init_process_group(device_id=...)`, else `LOCAL_RANK % device_count()`); the object from `dist.get_backend_impl(...)` is no longer a `ProcessGroupNCCL` | `TORCH_DIST_USE_NCCL2=0` or backend `"nccl-legacy"` reverts (also needed for several GPUs per process), and both are slated for removal in the next release; `"nccl-lazy"` creates communicators on first use (not run here) |
| FSDP2 with `MixedPrecisionPolicy(reduce_dtype=None)` now accumulates and reduces gradients in each parameter's `grad_dtype` (its original dtype, usually fp32) instead of `param_dtype` (usually bf16) | set `reduce_dtype=torch.bfloat16` to keep the 2.14 behavior (reproduced on CPU with 2 gloo ranks: the reduce-scatter runs in bf16 on 2.14.1 and fp32 on 2.15.0) |
| Distributed checkpoints record strides, so 2.15 checkpoints do not load in 2.14 or earlier | upgrade readers before writers |
| `torch.load(weights_only=True)` no longer allowlists nested (jagged or strided) tensors | allow `torch.nested._internal.nested_tensor._rebuild_njt` with `torch.serialization.safe_globals([...])` (reproduced: 2.15.0 refuses, 2.14.1 loads), or `weights_only=False` for files you trust |
| Re-importing torch in one process (`importlib.reload(torch)`, deleting it from `sys.modules`) raises `ImportError` | restart the interpreter instead |
| Profiler: `cupti_monitor` renamed `cuspy` (`CuspyConfig`); `FunctionEvent.metadata_json` removed | rename the config |
| `torch.compiler.precompile` is now a module (`capture`, `load`) | update imports |
| Dynamo: input mutation and aliasing inside higher-order ops are off by default. Reproduced for FlexAttention `score_mod`: an in-place op is an error under `fullgraph=True` on 2.15.0 (accepted on 2.14.1) and a graph break without it. `torch.cond` already refused input mutation on 2.14.1 | make mask and score functions pure |
| MPS requires macOS 15 or newer | |
| Default CUDA wheel is 13.2; CUDA 12.6 and 13.0 wheels are no longer published | pick the matching index URL |

**Deprecations:** passing a `set` of parameters to an optimizer emits `FutureWarning`
("optimizer parameters need to be organized in ordered collections"; reproduced on
2.15.0). Use a list. `torch.autograd.profiler.load_nvprof` and friends are scheduled for
removal in 2.17.

**Added:**

| API | What it does |
|-----|--------------|
| `torch.optim.functional` | public stateless update functions: `adam`, `adamw`, `sgd`, `muon`, and 10 more |
| `torch.foreach` (beta) | the multi-tensor `torch._foreach_*` ops under stable names, e.g. `torch.foreach.mul_(params, 3)` |
| `DeviceMesh.abort()` (experimental) | abort every process group of a root mesh so a peer that exits early does not hang the rest |
| `F.scaled_addmm` / `scaled_addmm_` | scaled matmul with per-operand scale recipes, for low-precision GEMMs |
| `DistributedDataParallel(lazy_bucket_allocation=...)`, `ddp.finalize_backward()` | lazy gradient buckets; manual end-of-backward (set `ddp.require_manual_backward_finalization = True` first, or it raises) when you drive backward yourself |
| `FSDPModule.set_manual_backward_finalization`, `finalize_backward`, `set_keep_unsharded_storage` | the same manual control for FSDP2 |
| `MixedPrecisionPolicy(param_dtype_override_fn=...)` | keep chosen parameters (router gates, norms) in fp32 while the rest trains in bf16 |
| `linear_cross_entropy` | soft-label targets with `reduction="none"` on the chunked path |
| `FlopCounterMode` | now counts `flex_attention` FLOPs |
| One-sided transport API (private `torch.distributed._transport`, NIXL backend), backend-agnostic Flight Recorder | experimental (not run here) |

Also fixed: `Muon` overwrote its momentum buffer with bf16 gradients. If you trained
with Muon on bf16 gradients before 2.15, the momentum was being truncated.

## Ecosystem versions used here

| Package | Version | How checked |
|---------|---------|-------------|
| torch | 2.15.0 (CPU build), re-checked on 2.14.1 | installed, introspected |
| torchvision | 0.29.2 (with 2.15), 0.29.1 (with 2.14.1) | installed, introspected |
| torchao | 0.18.0 | installed; `quantize_` run on CPU in `14-quantization.md` |
| CUDA | 13.2 default wheel for 2.15 | release notes only; no GPU here |
| Python | 3.11 to 3.15 for 2.15 | release notes; this skill's tests ran on 3.13 |

## Sources

- GitHub release notes, github.com/pytorch/pytorch/releases/tag/v2.15.0, read twice on
  2026-11-02: once through WebFetch (truncated in the distributed bug-fix list) and once
  through Firecrawl with direct quotes of the FSDP2 `reduce_dtype`, DCP stride, nested-tensor
  `weights_only` and `torch.foreach` entries.
- dev-discuss.pytorch.org/t/pytorch-release-2-15-key-dates/3434 - release date.
- PyTorchKR summary of the 2.15 release blog, discuss.pytorch.kr/t/pytorch-2-15/12351.
- PyTorch 2.13 release blog, pytorch.org/blog/pytorch-2-13-release-blog (July 8, 2026).
- PyTorch 2.14 release blog, pytorch.org/blog/pytorch-2-14-release-blog (September 2, 2026).
- docs.pytorch.org/docs/2.13 - `pytorch-api.html` module index, `quantization.html`,
  `nn.attention.html`, `torch.compiler_api.html`.
- `inspect.signature` against installed torch 2.15.0 / torchvision 0.29.2 and torch 2.14.1 /
  torchvision 0.29.1 (both CPU builds). The 2.13 rows were introspected on 2.13.0 by the
  previous version of this skill and still hold on 2.15.0 unless a row says otherwise.
