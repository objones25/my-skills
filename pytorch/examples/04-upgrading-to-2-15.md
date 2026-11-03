# Upgrading from torch 2.14.1 to 2.15

*Illustrative scenario; the `verify_api.py` output is real, run on torch 2.14.1 against
this skill's 2.15.0 tables. The behavior changes are from the 2.15 release notes unless
marked as reproduced.*

**Question.** "Our training code runs on 2.14.1 with FSDP2 and Muon. What do we need to
change for 2.15?"

## Verdict

Three things can change your results without an error; check them first:

1. **FSDP2 gradient precision.** With `MixedPrecisionPolicy(param_dtype=torch.bfloat16)` and
   `reduce_dtype` unset, 2.14 reduced gradients in bf16; 2.15 reduces them in the parameter's
   own dtype (fp32 for fp32 parameters). Numerics improve and reduce-scatter traffic doubles.
   Set `reduce_dtype` explicitly to choose.
2. **The `nccl` backend is a rewrite.** Same name, new implementation; `"nccl-legacy"` or
   `TORCH_DIST_USE_NCCL2=0` restores the old one for one release.
3. **Muon's momentum bug is fixed.** On 2.14 Muon overwrote its momentum buffer with bf16
   gradients. Runs that used bf16 gradients will train differently on 2.15, so an A/B against
   an old baseline is not apples to apples.

Then the ones that fail loudly: Python 3.11+ is required, distributed checkpoints written by
2.15 cannot be read by 2.14 (upgrade readers first), and a `set` of parameters now emits a
`FutureWarning` (reproduced on 2.15.0).

## Evidence

On the 2.14.1 install, the skill's signature check lists exactly what 2.15 adds:

```
$ python scripts/verify_api.py
installed: torch 2.14.1+cpu
reference: torch 2.15.0 / torchvision 0.29.2

checked 68 defaults + 20 symbols + 39 full signatures
9 difference(s) - trust the installed value, not the docs:

  torch.nn.parallel.DistributedDataParallel(lazy_bucket_allocation=...): doc says False, installed has <absent>
  torch.optim.functional.adamw: missing but expected (public functional optimizers, 2.15)
  torch.compiler.precompile.capture: missing but expected (precompile became a module in 2.15)
  torch.distributed.fsdp.FSDPModule.finalize_backward: missing but expected (2.15)
  torch.nn.parallel.DistributedDataParallel.finalize_backward: missing but expected (2.15)
  torch.profiler.CuspyConfig: missing but expected (renamed from the CUPTI monitor in 2.15)
  torch.foreach.mul_: missing but expected (public foreach module (beta), 2.15)
  torch.distributed.fsdp.MixedPrecisionPolicy: added -, removed ['param_dtype_override_fn']   [known: 2.15 added param_dtype_override_fn]
  torch.nn.parallel.DistributedDataParallel: added -, removed ['lazy_bucket_allocation']   [known: 2.15 added lazy_bucket_allocation]
```

Seven of the nine are pure additions that 2.14-era code cannot be calling. Two mark
breaking changes: `torch.compiler.precompile` was a callable on 2.14.1 and is a module on
2.15.0 (`precompile(fn)` now raises `TypeError: 'module' object is not callable`; use
`precompile.capture`), and `CuspyConfig` replaces the profiler's `cupti_monitor` names
with no alias. Beyond what a signature check can see:

- `torch.load(weights_only=True)` refuses checkpoints containing nested tensors on 2.15
  (reproduced); allow `torch.nested._internal.nested_tensor._rebuild_njt` with
  `torch.serialization.safe_globals`.
- The old collective names (`all_gather_into_tensor`, `reduce_scatter_tensor`) stop warning
  on 2.15; nothing to change, but a warning you were tracking disappears.
- The rewritten `nccl` binds each process group to one device: pass
  `init_process_group(..., device_id=...)`, and use `"nccl-legacy"` if a process drives
  several GPUs.

The FSDP2 change in item 1 was reproduced on CPU with two gloo ranks: the reduce-scatter
ran in bf16 on 2.14.1 and in fp32 on 2.15.0. The rest of the behavioral items come from the
release notes, summarized in `references/00-module-map.md`.

## What would change the answer

If you pass `reduce_dtype` explicitly today, item 1 does not apply. If you do not use bf16
gradients with Muon, item 3 does not apply. If you run the profiler's CUPTI monitor
(`cupti_monitor`), rename it to `cuspy`.
