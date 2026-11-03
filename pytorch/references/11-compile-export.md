# torch.compile, torch.export, and deployment

## Overview

Two different tools that are easy to confuse. **`torch.compile`** is a JIT: it traces
your Python at runtime, falls back to eager whenever it cannot trace, and exists to
make training and inference faster in Python. **`torch.export`** is AOT: it produces
a single, complete, serializable graph with no Python dependency, and exists to ship
the model somewhere Python is not. Use compile for speed, export for deployment.

## Recommended approach

Compile after moving the model to its device, and never benchmark the first batch.

```python
model = MyNet().to(device)
model = torch.compile(model)          # returns a wrapper; same call signature

for x, y in loader:                   # batch 1 pays the compile cost
    ...                               # batches 2+ run the fused kernels
```

Export with example inputs, marking the dimensions that vary:

```python
example = (torch.randn(3, 4),)
batch = torch.export.Dim("batch", min=1, max=64)
ep = torch.export.export(model.eval(), example, dynamic_shapes={"x": {0: batch}})
torch.export.save(ep, "model.pt2")
```

`strict` defaults to `False` (2.13 through 2.15) - the non-strict path traces with real tensors
and tolerates more Python. Pass `strict=True` to get TorchDynamo's soundness checks
and a hard error on anything it cannot verify.

## API reference

`torch.compile` signature from torch 2.15.0:

```
torch.compile(model=None, *, fullgraph=False, dynamic=None, backend=None, mode=None,
              options=None, name=None, disable=False, recompile_limit=None,
              isolate_recompiles=False, dynamic_shapes=None)
```

In 2.14 `shapes_spec` was renamed `dynamic_shapes`, the keyword `torch.export` uses.
`dynamic_spec` and `ShapesSpec`, which declare dynamic dimensions once for compile, export
and `make_fx`, live in `torch.fx.experimental.dynamic_spec` (not `torch.export`) on both
2.14.1 and 2.15.0; `torch.compile` wraps a `ParamsSpec` passed as `dynamic_shapes` into a
`ShapesSpec`.

| Argument | Effect |
|----------|--------|
| `fullgraph=True` | error on any graph break instead of falling back - use it to *find* breaks |
| `dynamic` | `None` = auto (recompile once, then generalize), `True` = assume dynamic up front, `False` = force static |
| `mode` | `"default"`, `"reduce-overhead"` (CUDA graphs, small models), `"max-autotune"` (slowest compile, fastest runtime), `"max-autotune-no-cudagraphs"` |
| `backend` | `"inductor"` by default; `torch.compiler.list_backends()` shows the rest |
| `recompile_limit` | cap recompilations before falling back to eager |

Compilation control (`torch.compiler`):

| API | Use |
|-----|-----|
| `torch.compiler.disable(fn)` | exclude a function from compilation (decorator or call) |
| `torch.compiler.set_stance("force_eager" / "eager_on_recompile" / "fail_on_recompile")` | change behavior without editing call sites |
| `torch.compiler.set_default_backend(b)` | process-wide default backend (new in 2.13) |
| `torch.compiler.allow_in_graph(fn)` | treat an opaque function as a single graph op |
| `torch.compiler.nested_compile_region(fn)` | compile a repeated block once, reuse it - cuts compile time on deep stacks; since 2.15 its `options=` accepts only `fallback_by_default` and `max_autotune` |
| `torch.compiler.precompile.capture(...)` / `.load(...)` | AOT-capture a compiled callable and reload it (a module since 2.15) |
| `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()` | ship a warm compile cache so production does not pay cold-start |
| `torch.compiler.reset()` | clear all compiled state (between benchmarks) |
| `torch.compiler.is_compiling()` | branch inside a model on whether it is being traced |

Export:

| API | Signature / use |
|-----|-----------------|
| `torch.export.export` | `(mod, args, kwargs=None, *, dynamic_shapes=None, strict=False, preserve_module_call_signature=(), prefer_deferred_runtime_asserts_over_guards=False)` |
| `torch.export.Dim(name, min=, max=)` | a symbolic dimension |
| `torch.export.ShapesCollection` | build `dynamic_shapes` for nested inputs |
| `torch.export.draft_export` | export that reports every failure at once instead of stopping at the first |
| `ep.module()` | a runnable `nn.Module` from the exported program |
| `ep.graph_signature`, `ep.graph` | inspect inputs, outputs, and the traced ops |
| `torch.export.save` / `load` | `.pt2` archive |
| `ep.run_decompositions(table)` | lower to a smaller core operator set for a backend |
| `torch.onnx.export(..., dynamo=True)` | ONNX via the export path |

Debugging environment variables:

| Variable | Shows |
|----------|-------|
| `TORCH_LOGS="graph_breaks"` | every place compilation fell back to Python |
| `TORCH_LOGS="recompiles"` | why a guard failed and forced a recompile |
| `TORCH_LOGS="output_code"` | the generated Triton/C++ |
| `TORCH_COMPILE_DEBUG=1` | full debug artifacts |

## Walkthrough

Export a tiny model, look at what came out, and make the batch dimension dynamic.
Output executed on torch 2.15.0.

**1. A model and a static export.**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class M(nn.Module):
    def __init__(self):
        super().__init__(); self.l = nn.Linear(4, 2)
    def forward(self, x):
        return F.relu(self.l(x))

m = M().eval()
ep = torch.export.export(m, (torch.randn(3, 4),))
print("type:", type(ep).__name__)
print("graph nodes:", [n.op for n in ep.graph.nodes])
```
```
type: ExportedProgram
graph nodes: ['placeholder', 'placeholder', 'placeholder', 'call_function', 'call_function', 'output']
```

Three placeholders, not one: the weight and bias were lifted out of the module and
became graph inputs, which is what makes the graph self-contained. Two
`call_function` nodes - the linear and the relu.

**2. It runs.**

```python
print("out shape:", tuple(ep.module()(torch.randn(3, 4)).shape))
```
```
out shape: (3, 2)
```

**3. Without `dynamic_shapes` the batch size is baked in at 3.** Mark it symbolic and
the same program accepts any batch in range.

```python
batch = torch.export.Dim("batch", min=1, max=64)
ep2 = torch.export.export(m, (torch.randn(3, 4),), dynamic_shapes={"x": {0: batch}})
print("batch=17:", tuple(ep2.module()(torch.randn(17, 4)).shape))
```
```
batch=17: (17, 2)
```

## Gotchas

- **Do not benchmark the first batch.** Compilation is lazy and the first call pays
  for it. Warm up, then time.
- **Changing input shapes forces recompilation.** Pad to fixed shapes, or accept the
  automatic-dynamic path (compile twice, then generalize). A recompilation loop is
  the usual reason compiled code is *slower*.
- **Graph breaks are silent by default.** `fullgraph=True` turns them into errors so
  you can find them; `TORCH_LOGS="graph_breaks"` lists them without failing.
- **Data-dependent control flow breaks the graph** - `if tensor.item() > 0`, `.cpu()`,
  `print` of a tensor, most numpy interop. Hoist them out of the hot path.
- **Compile after `.to(device)`**, and re-compile if you later move the model.
- **`torch.compile` and `DDP`/`FSDP` ordering matters.** TorchTitan's FSDP2 recipe
  applies activation checkpointing, then `torch.compile` to each transformer block,
  then `fully_shard`; compiling the whole model after wrapping is its stated
  alternative. Per-block compile also compiles one block once and reuses it.
- **`mode="reduce-overhead"` uses CUDA graphs**, which require static input addresses;
  it can silently misbehave with tensors that change identity each step.
- **Export is not compile.** `torch.export` refuses graph breaks by design - there is
  no eager fallback. A model that compiles fine can fail to export.
- **`strict=False` is the default (2.13 through 2.15).** Older code and older docs assume
  `strict=True`; the two paths accept different Python.
- **Higher-order ops reject input mutation.** `torch.cond` already refused an in-place op
  on an input on 2.14.1 ("Encountered input mutation during higher order op tracing"),
  with or without `fullgraph`. New in 2.15: a FlexAttention `score_mod` that mutates an
  input is an error under `fullgraph=True` and a graph break without it (it was accepted
  on 2.14.1). Keep these functions pure.
- **Export captures one control-flow path.** Use `torch.cond` / `torch.while_loop` to
  keep a branch in the graph rather than baking in whichever way the example inputs
  went.
- **`torch.jit.script` / `torch.jit.trace` are legacy.** New deployment work uses
  `torch.export`, ExecuTorch (on-device, now part of PyTorch core), or AOTInductor.

## Alternatives

**Which tool for which job:**

| Goal | Tool |
|------|------|
| Faster training in Python | `torch.compile(model)` |
| Faster inference in Python | `torch.compile(model, mode="max-autotune")` |
| Ship a graph to C++/mobile/edge | `torch.export` -> ExecuTorch or AOTInductor |
| Ship to an ONNX runtime | `torch.onnx.export(..., dynamo=True)` |
| Compile only the hot function | `@torch.compile` on that function, not the whole model |
| Repeated identical blocks, slow compile | `torch.compiler.nested_compile_region` |

**Reducing compile time**, which is the main cost in practice: compile the repeated
block rather than the whole stack (`nested_compile_region`), avoid `max-autotune`
during development, keep shapes stable, and persist the cache with
`save_cache_artifacts` / `load_cache_artifacts` so a second process or a production
replica starts warm. In 2.13 Inductor moved kernel compilation from a thread pool to
a subprocess pool, so compile-time parallelism no longer fights the GIL, and the
CuTeDSL backend gives GEMM and RMSNorm a second codegen path alongside Triton.

**Typical speedups** are 1.3x to 2x on transformer training on a modern GPU and more
on inference, but they depend entirely on whether the model is kernel-launch bound
(large gains) or already dominated by a few big matmuls (small gains). Measure with
`torch.utils.benchmark` - see `13-performance-profiling.md`.

## Related

- `05-data-training-loop.md` - where `torch.compile` slots into the loop.
- `09-attention-transformers.md` - FlexAttention is designed to be compiled.
- `13-performance-profiling.md` - measuring whether compilation actually helped.
- `12-distributed.md` - compiling under DDP/FSDP.
- `10-debugging.md` - `TORCH_LOGS` recipes when compiled code misbehaves.

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) (`torch.compile`, `torch.export.export`,
  `torch.compiler.*`); walkthrough output executed on the same build.
- docs.pytorch.org/docs/2.13 - `torch.compiler_api.html`,
  `user_guide/torch_compiler/export.html`, `torch_environment_variables.html`.
- PyTorch 2.14 release blog - `dynamic_shapes` rename, `@dynamic_spec`; the module path
  checked by import on 2.14.1 and 2.15.0.
- PyTorch 2.15 release notes - `precompile` module, `nested_compile_region(options=)`,
  higher-order-op mutation default.
- Liang et al. (2024), "TorchTitan," arXiv:2410.06511, and the torchtitan `parallelize`
  code - per-block compile after activation checkpointing and before FSDP.
- PyTorch 2.13 release blog - `set_default_backend`, CuTeDSL backend, subprocess
  kernel compilation, ExecuTorch in core.
