# Changelog

## 1.2.3, 2026-11-19: fix Pyright diagnostics

Type-checking fixes only; no behaviour change. `pyright pytorch` went from 60 errors to 0.

### Fixed

| Problem | Fix |
|---|---|
| `lint_pytorch.py`: `list[stmt]` passed where `list[AST]` was declared, `lineno` read from plain `AST`, `.func` read from an unnarrowed `expr` | `iter_scope` and `_scope` accept `list[ast.stmt] | list[ast.AST]`; `_is_backward`, `_is_opt_step` and `_is_super_init` return `TypeGuard[ast.Call]`; the assignment scan skips non-assignment nodes (their target list was already empty); `_discarded_to` asserts its caller's invariant |
| `memory_budget.py` `load_model`: `spec_from_file_location` can return `None` | raises `ImportError` naming the file instead of an `AttributeError` on `None` |
| `verify_api.py` `_params`: `resolve` had no return annotation | `resolve() -> Any` |
| `train_template.py`: RNG-state helpers subscripted possibly-`None` values; `torch.compile(model)` typed as a function | `dict \| None` annotations plus asserts for what `same_device` implies; `cast(nn.Module, ...)` on the compiled model |
| `ddp_template.py`: `Module.__getattr__` typing hid `no_sync`, `module` and `set_requires_gradient_sync` | `build_model` returns `nn.Sequential`; `isinstance` asserts on `DDP` and `FSDPModule`; the full state dict cast to `dict[str, Tensor]` |
| Tests: unchecked `spec`/`spec.loader`, `re.Match`, `__doc__` and `current_accelerator()` `None`s; `_torch_minor` returned `tuple[int, ...]` | asserts or explicit narrowing; `_torch_minor` unpacks `(major, minor)`; `import torch.utils.checkpoint` made explicit |

Two `# pyright: ignore[reportPrivateImportUsage]` comments remain, for `torch._dynamo.is_dynamo_supported` (private on purpose) and `torch.distributed.HashStore` (exported at runtime, not re-exported for the checker).

## 1.2.2, 2026-11-16: fix the findings of the 2026-11-16 audit

Evidence for each fix is in `AUDIT.md` (audit of 1.2.1, risk scope, 162 of 1547 items checked).

### Fixed

| Problem | Fix |
|---|---|
| `diagnose.py overfit` used the last dimension as the class count, so `(N, C, d1)` logits with label smoothing got the wrong floor and a memorized batch could be called "stuck" | class count from dim 1 (dim 0 for unbatched logits) |
| Overfit verdict thresholds, the no-trainable guard, `loss_floor`'s edge cases, `memory_budget.py`'s pure-bf16 layout and `muon_split`'s conditions had no test that could fail | thresholds moved into `judge()`; boundary, edge-case and layout tests added, checked with `mutate_check` |
| Linear cross entropy described as chunking the vocab and cutting memory ~4x by default | chunking is over the batch and opt-in (`options=nn.LinearCrossEntropyOptions()`); the default saves nothing; ~4x attributed to the release blog as not reproduced (`00`, `04`, `13`) |
| `torch.load` of safetensors said to need no extra library | it needs the `safetensors` package (`00`, `05`) |
| Nested-tensor `weights_only` remedy incomplete | jagged needs `_rebuild_njt` and `NestedTensor`; strided needs `_rebuild_nested_tensor` (`05`, example 04) |
| `nn.RMSNorm(eps=None)` said to use the input dtype's epsilon | fp32 epsilon for fp32, bf16 and fp16 (`03`, linter message) |
| DenseNet's head said to be a `Sequential` | it is a `Linear` (`07`) |
| LSTM/GRU argument list skipped `bias` | added (`03`) |
| AdamW `weight_decay` given as a default that moved | removed (`SKILL.md`) |
| `--docs` said to resolve every name / ~190 names | names the prefixes it resolves (`SKILL.md`, `AUDIT.md`) |
| `diagnose.py --model file.py:factory` documented | the real CLI: a positional file (`10`) |
| Deprecated collectives said to warn as `UserWarning` | `FutureWarning` on 2.13/2.14, silent on 2.15 (`10`) |
| Quantization step: "two 80 GiB cards", "about 2 FLOPs per byte" | 13.0 to 3.3 GiB; about 1 FLOP/byte for bf16 weights (`14`) |
| `once_differentiable` sentence attributed to `Function.backward` | it is in `FunctionCtx.save_for_backward` (`15`) |
| `verify_api.py` said 2.14 added `preserve_rng_state` | 2.14 made it an explicit keyword; 2.13 read it from `**kwargs` |
| LoRA and RMSNorm research claims stated more strongly than the papers | reworded to what Hu et al., Biderman et al. and Zhang and Sennrich report (`07`, `09`) |
| `score_mod` said not to return an alias | narrowed to what the release notes and a run show (`09`) |
| Example 04's `verify_api.py` header | matches what the script prints |
| Muon "present in 2.14.1 and 2.15.0" | present in 2.12.1 through 2.15.0 (`04`) |

### Test results by machine

| Date | Machine | torch | Result |
|---|---|---|---|
| 2026-11-16, after these fixes | macOS arm64 (Apple Silicon), MPS, Python 3.13.2 | 2.15.0 (torchvision 0.29.2, torchao 0.18.0, safetensors 0.8.0) | 415 passed, 2 skipped (one test is for torch 2.14, one needs a machine with no accelerator) |
| 2026-11-16, after these fixes | same machine | 2.14.1 (torchvision 0.29.1, no safetensors or torchao) | 410 passed, 7 skipped (safetensors absent, four 2.15-only tests, the torchao walkthrough, no-accelerator test) |
| 2026-11-16, before these fixes | same machine | 2.15.0 | 385 passed, 2 skipped |

Mutation checks after the fixes (`mutate_check.py`, same machine): `diagnose.judge` 15/15, `diagnose.loss_floor` 21/21, the `overfit` no-trainable guard 2/2, `memory_budget.bytes_per_param` 21/21, `memory_budget.muon_split` 8/8 killed.

## 1.2.1, 2026-11-03: resume fixed off CPU, machine-specific results labelled

### Fixed

| Problem | Evidence | Fix |
|---|---|---|
| `train_template.py --resume` crashed on MPS with `RNG state must be a torch.ByteTensor` | `map_location=device` moved the saved RNG state off the CPU | Load checkpoints to CPU; `load_state_dict` moves the rest |
| Resume was not bit-exact on MPS (1.7052 versus 1.7077 at epoch 2) | Only the CPU RNG was saved, and dropout on the device draws from the device RNG | Save the accelerator RNG with its device type, and restore it on the same kind of device |
| `references/01-devices-amp.md` documented `device: cpu` as if universal | Walkthrough printed `device: mps` on Apple Silicon | Labelled machine-specific; new `cpu-output` fence is compared only with no accelerator |

### Test results by machine

| Date | Machine | torch | Result |
|---|---|---|---|
| 2026-11-03, after these fixes | macOS arm64 (Apple Silicon), MPS | 2.15.0 | 383 passed, 4 skipped (`safetensors` and `torchao` not installed, one test needs a machine with no accelerator, one is for torch 2.13 and 2.14) |
| 2026-11-03, before these fixes | macOS arm64 (Apple Silicon), MPS | 2.15.0 | 369 passed, 3 failed, 4 skipped |
| 2026-11-02, release 1.2.0 | Not recorded | 2.15.0 | 375 passed, 1 skipped |

The CUDA paths in the fix were not run: no CUDA machine was available. A checkpoint from a different kind of device (including CPU), a different GPU count, or the earlier format now resumes with a printed note and is not bit-exact, because RNG streams cannot be converted between devices.

## 1.2.0, 2026-11-02: torch 2.15, linter, triage and memory scripts, audit fixes

Evidence for each fix is in `AUDIT.md`.

### Fixed

| Problem | Fix |
|---|---|
| Accelerator RNG functions documented on `torch.accelerator` | they are in `torch.accelerator.random` and raise without an accelerator |
| `dynamic_spec` / `ShapesSpec` implied to be in `torch.export` | `torch.fx.experimental.dynamic_spec` |
| `RandomErasing` said to need float input | works on uint8 too (torchvision 0.29.2) |
| Muon's 1-D params sent to "a separate param group" (raises `ValueError`) | a separate AdamW optimizer |
| Signature tables missing parameters (`AdamW`, `SGD`, `linear_cross_entropy`, `BatchNorm2d`) | completed; checked by `verify_api.py --docs` |
| `nn.LSTM` signature presented as introspected | marked as `(*args, **kwargs)` |
| `checkpoint` `use_reentrant` "must be passed" | warns; docs say 2.9 raises, 2.15 does not |
| Python 3.15 `torch.compile` note, fused-optimizer device list | updated for 2.15 |

### Added

- torch 2.15: "What changed in 2.15" in `00-module-map.md`; rewritten `nccl`, FSDP2
  `reduce_dtype`, DCP strides, `finalize_backward`, `param_dtype_override_fn` in `12`;
  Muon details and set-of-params deprecation in `04`; nested-tensor `weights_only` in `05`;
  pure FlexAttention mods in `09`; `precompile`, `nested_compile_region(options=)` and
  higher-order-op mutation in `11`; Cuspy in `13`; macOS 15 for MPS in `01`.
- Scripts: `lint_pytorch.py` (24 codes), `diagnose.py` (overfit, grads, nan),
  `memory_budget.py` (with `--measure`); `verify_api.py --docs` and `--json`.
- Assets: `ddp_template.py`, `test_model_template.py`, `conftest_torch.py`.
- References: `15-custom-ops-autograd.md`, `16-testing-pytorch-code.md`, `sources.md`;
  real API walkthroughs in `08`, `12`, `14`; eight new rows in the `10` symptom table.
- `examples/` (six worked answers, planted-bug models), `evals/evals.json` (17 cases),
  `AUDIT.md`, this file, `tests/test_skill_structure.py`.

### Fixed after the independent fact-check

14 errors and 20 imprecisions in this release's new material, listed in `AUDIT.md`
(collective warnings on 2.15, frozen-layer backward claim, example 04, `verify_api.py
--docs` blind spots, three `memory_budget.py` bugs, `diagnose.py` verdict, two linter
false positives).

### Changed

- Reference version: torch 2.15.0 / torchvision 0.29.2 (was 2.13.0 / 0.28.0), re-checked on
  2.14.1 / 0.29.1. Python 3.11+ for the scripts.
- `pytest.ini` unchanged; the suite grew from 32 to 376 tests.

### Verification at release

| Check | Result |
|---|---|
| `pytest` on torch 2.15.0, machine not recorded | 375 passed, 1 skipped |
| `pytest` on torch 2.14.1, machine not recorded | 372 passed, 4 skipped (torchao walkthrough, 2.15-only docs scan, two 2.15-only claims) |
| `verify_api.py --docs` on 2.15.0 | no drift, no reference findings |
| `verify_api.py --docs` on 2.14.1 | 9 pinned differences and 9 reference names, all 2.15 additions or renames |
| `lint_pytorch.py assets/ scripts/` | clean |
| `diagnose.py` on `examples/models/` | 5 of 5 classified correctly |
| `memory_budget.py --measure` | formula matches measured state for 5 optimizers |

## 1.1.0

Previous release: torch 2.13 reference with a 2.14.1 re-check. That version's files are not kept in this repo.
