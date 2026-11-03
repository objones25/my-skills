# Audit, 2026-11-02

Previous version: 1.1.0 (verified against torch 2.13.0 and re-checked on 2.14.1). SKILL.md,
15 reference files, 2 scripts, 1 asset, 32 tests; no evals, examples, AUDIT, CHANGELOG,
sources file, or structural tests. Its 32 tests still passed on torch 2.15.0, and its
`verify_api.py` reported 4 drifted signatures there (3 known from 2.14, plus DDP's new
`lazy_bucket_allocation`).

## Errors found in the previous version

| Claim | Evidence | Fix |
|---|---|---|
| "`torch.accelerator` gains `initial_seed`, `get_rng_state`, `get_rng_state_all`" (2.14 table) | `torch.accelerator.initial_seed` is an `AttributeError` on 2.14.1 and 2.15.0; the functions are in `torch.accelerator.random`, and they raise `RuntimeError` when no accelerator is present | path and the raise documented in `00` and `01`; pinned in `verify_api.py` PRESENCE (both the real path and the absent one) |
| "`@dynamic_spec` / `ShapesSpec` ... prefer it over per-call `Dim` dicts", listed next to `torch.export` | `torch.export.dynamic_spec` does not exist on 2.14.1 or 2.15.0; both live in `torch.fx.experimental.dynamic_spec` | location stated in `00` and `11`; pinned absent in PRESENCE |
| `RandomErasing` "needs a **float** tensor - place after `ToDtype`" (`08`, three places) | `v2.RandomErasing(p=1.0)` erased a uint8 image without error on torchvision 0.29.2 | reworded: works on either; placing it after `ToDtype` is a convention about fill units |
| Muon's 1-D parameters "stay on AdamW in a separate param group" (`04`) | `torch.optim.Muon([1-D param])` raises `ValueError: Muon only supports 2D parameters`; a param group inside Muon would raise. It must be a separate `AdamW` optimizer | rewritten; the `ValueError` is in the `10` symptom table and in an eval |
| "Muon (in `torch.optim` as of 2.13)" | no 2.13 install was available to confirm the version it arrived in | now "present in 2.14.1 and 2.15.0" |
| Signature tables omitted parameters the skill's own `verify_api.py` listed: `AdamW` without `differentiable`, `SGD` without `maximize` and `differentiable`, `F.linear_cross_entropy` without `options`, `BatchNorm2d` without `track_running_stats` | `inspect.signature` on 2.14.1 and 2.15.0 | tables completed; `verify_api.py --docs` now checks table rows against the install |
| `nn.LSTM` / `nn.GRU` row presented as an introspected signature | the Python signature is `(*args, **kwargs)`; `inspect` cannot read the documented parameters | row says so |
| `checkpoint(use_reentrant=...)`: "`None` - must be passed explicitly" | omitting it warns and falls back to the reentrant path on 2.13 through 2.15; the 2.14 docs say "In version 2.9 we will raise an exception", which 2.15.0 does not do | rewritten as the docs-versus-code disagreement, following the code |
| Python 3.15 wheels: "`torch.compile` unsupported there" | true for 2.13; the 2.15 release notes say `torch.compile` works on 3.15 | qualified |
| Fused optimizer devices "cuda, cpu, mps, xpu, hpu, mtia" | `_get_fused_kernels_supported_devices()` on 2.15.0 also returns `privateuseone` | updated with the call that produced it |

## Gaps found

- No version newer than 2.14: torch 2.15.0 (released 2026-10-28) changed the `nccl` backend,
  FSDP2's gradient dtype, the profiler's CUPTI monitor name, `torch.compiler.precompile`, and
  the `weights_only` allowlist, and dropped Python 3.10. None of it was in the skill.
- Three walkthroughs did not exercise the API they document: `08` did the transform math by
  hand without calling `transforms.v2`, and `12` and `14` were arithmetic only.
- The debugging protocol ("overfit one batch first") had no tool; nothing ran a user's model
  and named the failing layer.
- Nothing checked user code for the non-negotiables the skill lists.
- No coverage of custom `autograd.Function`, `torch.library.custom_op`, `torch.func`, hooks, or
  of testing PyTorch code.
- No multi-process test of distributed code; the DDP skeleton was never run.
- `verify_api.py` checked only its own pinned tables, not the reference text, so a wrong
  default typed into a reference could not be caught.
- No evals, worked examples, sources file, AUDIT, CHANGELOG, or structural tests.

## Added

- `scripts/lint_pytorch.py`: stdlib AST linter with 24 codes; every code has a triggering and
  a near-miss test (133 tests). Its runtime claims (set-of-params `FutureWarning`, checkpoint
  warning, deprecated AMP names, `batch_first` and `eps` defaults) were reproduced on 2.15.0
  and 2.14.1.
- `scripts/diagnose.py`: `overfit`, `grads`, `nan` on any file defining `make_model()` and
  `make_batch()`. It classifies all five planted-bug models in `examples/models/` correctly,
  including a stray `.detach()` that the linter cannot see.
- `scripts/memory_budget.py`: per-rank weights, gradients and optimizer state for DDP or FSDP2;
  `--measure` takes a real optimizer step and matches the formula exactly for AdamW, Adam, SGD
  with momentum, Muon (with AdamW for 1-D params and embeddings), and Adafactor, including
  conv weights and frozen layers (after the fact-check fixes below).
- `scripts/verify_api.py`: rebased on 2.15.0; 68 defaults, 20 presence checks, 39 full
  signatures; `--docs` resolves all ~190 API names in `references/` and checks signature claims;
  `--json`; known-change notes so a 2.14 user sees which drift is expected.
- `assets/ddp_template.py` (DDP or FSDP2 under `torchrun`, runs on CPU gloo; the test launches 2
  ranks and checks identical replicas and disjoint shards), `assets/test_model_template.py` and
  `assets/conftest_torch.py` (10 tests a user copies next to a model; the suite plants six bugs
  and checks each is caught).
- `references/15-custom-ops-autograd.md`, `references/16-testing-pytorch-code.md`,
  `references/sources.md`; a "What changed in 2.15" section in `00`; 2.15 material in `01`,
  `04`, `05`, `09`, `10`, `11`, `12`, `13`; real API walkthroughs in `08`, `12` and `14`.
- `examples/`: six worked answers with real output, five planted-bug models, a buggy script.
- `evals/evals.json` (17 cases); `tests/test_skill_structure.py`.
- Tests: 32 to 376 on torch 2.15.0 (375 pass, 1 skip, machine not recorded; see `CHANGELOG.md` for later results by machine); on 2.14.1, 372 pass and 4 skip
  (torchao walkthrough, the 2.15-only docs scan, and two 2.15-only claims).

## How things were checked

- **inspect**: every signature and default in the references and in `verify_api.py` was read
  with `inspect.signature` on installed torch 2.15.0 and 2.14.1; behavior that a signature does
  not show was read from source (`torch/optim/_muon.py` `_adjust_lr`, `torch/amp/grad_scaler.py`
  for the MPS path) or reproduced by running it. `verify_api.py --docs` reports no findings on
  2.15.0.
- **Context7**: PyTorch 2.14 docs for `torch.utils.checkpoint` and `torch.optim.Muon`, and the
  torchao docs (config table, int4 packing by device). Two disagreements with the code were
  found and resolved in favor of the code (`references/sources.md`).
- **Firecrawl**: the v2.15.0 GitHub release notes, read with direct quotes for the FSDP2
  `reduce_dtype`, DCP stride, nested-tensor `weights_only` and `torch.foreach` entries. This
  second reading corrected a summary taken from a first WebFetch pass, which had left the
  direction of the FSDP2 change ambiguous: on 2.15, fp32 parameters with `param_dtype=bf16`
  reduce in fp32, not bf16. The Firecrawl connection to the user's computer dropped several
  times; the official pytorch.org 2.15 blog page timed out and was not read.
- **WebFetch / WebSearch**: the dev-discuss 2.15 key-dates post (release date) and the
  PyTorchKR summary of the 2.15 blog, as corroboration; CPython gh-84559 for the Python 3.14
  `forkserver` default.
- Every walkthrough, script and template was executed on both versions by the test suite.
- An independent fact-check pass by a separate agent that had not written the material
  (results below).

## Independent fact-check

A separate agent that had not written the material re-read the 2.15 release notes, reran
every example command, introspected the cited APIs on both versions, and tried to make
each script give a wrong answer. It found 14 errors and 20 imprecisions or nits in this
version's new material. All were fixed; each fix was re-run.

| Finding | Fix |
|---|---|
| The renamed collectives were said to emit `FutureWarning`; on 2.15.0 they are silent aliases (release notes: "Stop emitting a `FutureWarning`") | stated per version; pinned in `tests/test_claims.py` for 2.14 and 2.15 |
| `get_rng_state_all` said to raise without an accelerator; it returns `[]` | corrected; pinned |
| `torch.accelerator.max_memory_allocated(device=None)`: the parameter is positional-only `device_index` | corrected; `verify_api.py --docs` now catches this class of error |
| `07`: "autograd still backpropagates through frozen early layers": backward; a frozen prefix records no graph | rewritten |
| Example 04: "all nine are additions"; `torch.compiler.precompile` stopped being callable and `CuspyConfig` replaces names without alias | 7 additions, 2 breaking changes (`precompile` reproduced on both versions); nested-tensor `weights_only` and nccl device binding added |
| `11`: higher-order-op mutation "since 2.15 ... a graph break otherwise"; `torch.cond` already raised on 2.14.1 | rewritten with FlexAttention `score_mod` as the 2.15 change |
| `verify_api.py --docs` missed a two-keyword misquote (`AdamW(params, lr=0.001, weight_decay=0)`) and unknown keywords in spans it read as usage examples | unknown keywords always reported (unless the callable takes `**kwargs`); defaults compared when at least half of two or more comparable keywords match; docstring lists the remaining blind spots; four regression tests |
| `verify_api.py`: `"torchao".startswith("torch")` produced false "unresolved" lines when torchao is absent; missing attributes reported as `ModuleNotFoundError` | exact package roots; `AttributeError` reported as such |
| `memory_budget.py`: Adafactor formula wrong for tensors with 3 or more dims; `--measure` dropped 1-element moment tensors; frozen parameters counted for gradients and state; Muon split sent embeddings to Muon | all four fixed; 13 edge-case tests (conv, 1-element bias, frozen layer, embedding plus head, each for three optimizers) |
| `diagnose.py overfit`: false "stuck" under label smoothing; false "memorized" for a network that only learned the target mean | label-smoothing floor computed, `--floor` for other cases, constant-prediction baseline required; three regression tests |
| `lint_pytorch.py`: `ASSIGN_BEFORE_SUPER` flagged `self.half = math.ceil(d / 2)` at error severity; `DISCARDED_TO` flagged `inputs = tokenizer(...); inputs.to(device)` | narrowed to `torch.nn.*` and local module classes; a name assigned from a non-torch call is unknown; regression rows |

Imprecisions fixed: `ddp.finalize_backward()` needs `require_manual_backward_finalization`
(pinned); the rewritten `nccl` binds to `device_id` and ignores `set_device`;
`get_backend_impl` wording and `nccl-lazy`; the FSDP2 communication-volume formula now
depends on dtype; the FSDP2 `reduce_dtype` change is now marked reproduced (re-run here:
bf16 reduce-scatter on 2.14.1, fp32 on 2.15.0, bf16 on 2.15.0 with `reduce_dtype` set);
`param_dtype_override_fn` is keyword-only; the overfit claim is qualified for label
smoothing; `diagnose.py`'s documented `--lr` matched to its default; `check_env.py` now
reports `is_bf16_supported()` and says `torch.compile` presence is not a test compile;
float `(B,)` targets raise rather than switching to soft targets; the checkpoint warning
history; the `shapes_spec` and `preserve_rng_state` claims now cite the pinned 2.13.0
signature list; the nested-tensor global is named; `.view()` errors only on incompatible
strides; Muon's third `adjust_lr_fn`; the transport API is private; `antialias` is
already the default; a log-uniform weight-decay range cannot include 0; the
double-softmax floor has other possible causes; ZeRO, DTensor and MFU defined at first use.

## Not verified

- Anything needing CUDA, MPS or multiple GPUs: the rewritten NCCL backend, FSDP2's 2.15
  gradient dtype in practice, attention backend selection, fp16 `GradScaler` on MPS, int4 CUDA
  kernels, CUDA graphs. These come from release notes and docs and are marked "not run here".
- Speed and memory figures quoted from release blogs.
- The papers and vendor posts listed as "carried" in `references/sources.md` were not re-read.
