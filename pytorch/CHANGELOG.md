# Changelog

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
| `pytest` on torch 2.15.0 | 375 passed, 1 skipped |
| `pytest` on torch 2.14.1 | 372 passed, 4 skipped (torchao walkthrough, 2.15-only docs scan, two 2.15-only claims) |
| `verify_api.py --docs` on 2.15.0 | no drift, no reference findings |
| `verify_api.py --docs` on 2.14.1 | 9 pinned differences and 9 reference names, all 2.15 additions or renames |
| `lint_pytorch.py assets/ scripts/` | clean |
| `diagnose.py` on `examples/models/` | 5 of 5 classified correctly |
| `memory_budget.py --measure` | formula matches measured state for 5 optimizers |

## 1.1.0

Previous release: torch 2.13 reference with a 2.14.1 re-check (see that version's files).
