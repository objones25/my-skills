---
name: pytorch
description: PyTorch 2.15 engineering reference with signatures verified against the installed library, runnable patterns, a symptom-indexed debugging guide, and scripts that lint training code, triage a model that will not learn or goes NaN, and size memory for DDP or FSDP2. Use when writing, reviewing, testing, or debugging PyTorch code - training loops, nn.Module models, Dataset/DataLoader, losses and optimizers (AdamW, Muon), AMP and device placement, torchvision transforms.v2, fine-tuning and LoRA, attention/SDPA/FlexAttention, custom autograd and custom ops, torch.compile and torch.export, DDP/FSDP2, profiling, quantization via torchao, or upgrading between torch versions. Triggers on torch.* or nn.* API questions, .pt/.pth/.safetensors checkpoints, shape/device/dtype errors, NaN loss, CUDA OOM, "my model won't learn", and "review my training script". Not for general ML theory with no PyTorch code.
license: MIT
compatibility: Scripts need Python 3.11+ (torch 2.15 dropped 3.10) and torch; lint_pytorch.py needs only the standard library. References are verified against torch 2.15.0 / torchvision 0.29.2 / torchao 0.18.0 and re-checked on torch 2.14.1 / torchvision 0.29.1 (CPU builds). Run pytest from the skill root to re-check every walkthrough, script and claim. No network access required.
metadata:
  author: owenbeckettjones
  version: 1.2.0
  verified-against: "torch 2.15.0, 2.14.1"
  pytorch-version: "2.15.0"
  torchvision-version: "0.29.2"
---

# PyTorch

## Start here

**Step 1 - pin the version before quoting any API.** PyTorch's defaults and behavior move
between minor releases (`AdamW`'s `weight_decay`, `torch.load`'s `weights_only`, FSDP2's
gradient dtype in 2.15). Run:

```bash
python scripts/check_env.py
```

It prints the torch/torchvision version, the live accelerator, bf16 support (from
`torch.cuda.is_bf16_supported()` on CUDA), and whether `torch.compile` and distributed
backends are present (presence, not a test compile). If the version is not 2.15, run
`python scripts/verify_api.py --docs` too: it introspects the installed library and
reports every pinned default, full signature, and API name in `references/` that differs
(on 2.14.1: nine pinned differences and nine reference names, all 2.15 additions or
renames). Quote the script's output, not this skill,
when they disagree. If torch is not installed, say the reference targets 2.15 and that
unverified defaults may differ on the user's version.

**Step 2 - route to the reference you need.** Read the one or two files the task touches,
not all of `references/`.

| Read this | When the task is |
|-----------|------------------|
| `references/00-module-map.md` | which `torch.*` submodule owns something; what changed in 2.13, 2.14 and 2.15 |
| `references/01-devices-amp.md` | GPU/MPS placement, `.to(device)`, pinned memory, autocast/`GradScaler`, memory accounting |
| `references/02-tensors-autograd.md` | tensor semantics, `backward()`, `detach`/`no_grad`/`inference_mode`, `nn` vs `nn.functional` |
| `references/03-models-layers.md` | defining an `nn.Module`, layer catalog, shape bookkeeping, containers, initialization |
| `references/04-losses-optimizers.md` | choosing a loss, target conventions, AdamW/SGD/Muon, param groups, LR schedules, clipping |
| `references/05-data-training-loop.md` | `Dataset`/`DataLoader`, the train and eval loop, accumulation, checkpoints, logging |
| `references/06-tuning-reproducibility.md` | seeding, determinism, what to tune, Optuna, experiment tracking |
| `references/07-fine-tuning.md` | pretrained weights, freezing, head swap, discriminative LR, LoRA/PEFT |
| `references/08-transforms-augmentation.md` | `transforms.v2` pipelines, augmentation catalog, MixUp/CutMix, train vs eval |
| `references/09-attention-transformers.md` | SDPA, FlexAttention, RoPE, RMSNorm, KV cache, GQA, the modern block |
| `references/10-debugging.md` | any error message, NaN loss, won't-learn, OOM, slow training |
| `references/11-compile-export.md` | `torch.compile`, graph breaks, `torch.export`, ONNX, deployment |
| `references/12-distributed.md` | DDP, FSDP2 `fully_shard`, DTensor, tensor/pipeline parallel, distributed checkpointing |
| `references/13-performance-profiling.md` | profiling, FLOP counting, benchmarking, memory reduction, kernel selection |
| `references/14-quantization.md` | quantization, torchao, `torch.ao` migration |
| `references/15-custom-ops-autograd.md` | custom `autograd.Function`, `gradcheck`, `torch.library.custom_op`, `torch.func`, hooks |
| `references/16-testing-pytorch-code.md` | writing tests for models and training code, `assert_close` tolerances |
| `references/sources.md` | where each fact came from, how it was checked, what was not verified |

Symptom-first shortcut: any traceback, NaN, OOM, or silent non-learning goes to
`references/10-debugging.md` first. Its table maps error text to a fix.

**Step 3 - write the code, then check it.** Follow the non-negotiables below and the
patterns in the reference file. Adapt a template rather than retyping one:
`assets/train_template.py` (single process, AMP, accumulation, bit-exact resume),
`assets/ddp_template.py` (DDP or FSDP2 under `torchrun`, runs on CPU), and
`assets/test_model_template.py` with `assets/conftest_torch.py` (tests for a user's
model). Before handing code over, run `python scripts/lint_pytorch.py <files>`.

## Scripts

| Script | Does |
|---|---|
| `scripts/check_env.py` | versions, accelerator, bf16 support, whether `torch.compile` is present, distributed backends |
| `scripts/verify_api.py` | pinned defaults, symbols and 39 full signatures against the install; `--docs` resolves every API name in `references/`, reports keywords the installed callable does not accept, and compares defaults in spans that read as signatures (blind spots in its docstring); `--json` |
| `scripts/lint_pytorch.py` | stdlib AST linter, 24 codes: dropped `.to()`, softmax into CrossEntropy, plain list of layers, unregistered tensors, missing `zero_grad`, loss tensors kept alive, `F.dropout` without `training=`, deprecated AMP names, and more; `--list-codes`, `--format json\|github`, `# lint: ignore[CODE] reason` |
| `scripts/diagnose.py` | runs a model defined by `make_model()`/`make_batch()`: `overfit` (one-batch memorization verdict), `grads` (per-parameter table, hidden tensors and modules, softmax-head warning), `nan` (first module to emit NaN/Inf) |
| `scripts/memory_budget.py` | weights, grads and optimizer state per rank for DDP or FSDP2; `--measure` checks the formula against a real optimizer step |

The linter reads code; `diagnose.py` runs it. They catch different bugs:
`examples/README.md` lists which planted bug each one finds.

## Non-negotiables

These hold for every PyTorch task. Violating one produces either a crash or, worse,
training that runs and silently learns nothing. Most have a `lint_pytorch.py` code.

1. **The model's head emits raw logits.** No `softmax` or `sigmoid` on the output layer.
   `CrossEntropyLoss` and `BCEWithLogitsLoss` fuse the activation with the loss for
   numerical stability; adding your own applies it twice.
2. **`CrossEntropyLoss` takes `(B, C)` float logits and `(B,)` int64 class indices** - not
   one-hot, not probabilities. `BCEWithLogitsLoss` takes float targets of the logits' shape.
3. **Tensors move by copy, modules move in place.** `x = x.to(device)` (reassign);
   `model.to(device)` (no reassign).
4. **The step order is `zero_grad` -> `forward` -> `loss` -> `backward` -> `step`.**
   Gradients accumulate by default; skipping `zero_grad` silently corrupts updates.
5. **`model.eval()` for validation, `model.train()` back afterwards**, and guard eval with
   `@torch.inference_mode()`. The mode flips Dropout and BatchNorm only; it does not
   disable gradients.
6. **Save `state_dict`, never the model object**, and load with `weights_only=True` (the
   default since 2.6). To resume, also save optimizer, scheduler, epoch and RNG state.
7. **Register non-parameter state.** A bare `self.scale = torch.ones(3)` is invisible to
   `.to()`, `state_dict()` and the optimizer; layers in a plain `list` are invisible too.
   Use `nn.Parameter`, `register_buffer`, `nn.ModuleList`.
8. **Never claim a signature or default from memory.** Every value in `references/` was
   introspected from torch 2.15.0. Outside that set, check the installed library
   (`inspect.signature`) or run it. When docs and code disagree, follow the code and say so.

## Defaults to reach for

| Decision | Default | Reference |
|----------|---------|-----------|
| Device selection | `torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")` | 01 |
| Mixed precision | `torch.autocast(device.type, dtype=torch.bfloat16)`, no scaler | 01 |
| Optimizer | `AdamW(params, lr=3e-4)`; `weight_decay` already defaults to `0.01` | 04 |
| Muon | 2-D weights only (1-D raises `ValueError`); a separate AdamW for the rest | 04 |
| Multiclass loss | `nn.CrossEntropyLoss()` on raw logits | 04 |
| Large-vocab LM loss | `nn.LinearCrossEntropyLoss` (2.13+, API Unstable, options changed in 2.14 and 2.15) | 04 |
| LR schedule | `CosineAnnealingLR` (per epoch) or `OneCycleLR` (per batch); warmup for transformers | 04 |
| Eval guard | `@torch.inference_mode()` | 02 |
| Attention | `F.scaled_dot_product_attention(..., is_causal=True)`; `flex_attention` for custom masks | 09 |
| Custom backward | `torch.autograd.Function` checked with `gradcheck` in float64 | 15 |
| Speedup | `model = torch.compile(model)` after `.to(device)` | 11 |
| Multi-GPU | DDP if `memory_budget.py` says one copy fits, else FSDP2 `fully_shard`; set FSDP2 `reduce_dtype` explicitly on 2.15 | 12 |
| Pretrained vision | `weights=Weights.DEFAULT` plus `weights.transforms()` | 07 |
| Quantization | torchao `quantize_`, not `torch.ao.quantization`; int8 on CPU, int4 on CUDA | 14 |

## Debugging protocol

When training runs but does not learn, **overfit a single batch first** - before touching
hyperparameters, data, or architecture: `python scripts/diagnose.py all my_model.py`, where
the file defines `make_model()` and `make_batch()`. A correct model with hard targets
drives one batch's loss to near zero in 200 steps (to a known floor with label smoothing,
which the script computes) and beats a constant prediction. If it cannot, the bug is in the model, loss, or step, and the
grads table usually names it (`NONE` = never reached the loss, `HIDDEN` = invisible to the
optimizer). If it can, the bug is downstream: data, labels, augmentation, LR.
`references/10-debugging.md` has the checklist for each outcome;
`examples/01-model-wont-learn.md` is a worked case.

## Writing the answer

- Show real shapes and values. Run the snippet and paste actual output rather than
  describing what it would print. `pytest` re-executes every walkthrough in `references/`
  and fails on any mismatch.
- Thread one toy example through a whole explanation - same tensor, same shapes.
- Annotate every line that changes a shape with the shape before and after.
- Name the version when an API is version-sensitive ("`strict` defaults to `False`, 2.13
  through 2.15").
- Define each non-obvious term, variable, or mechanism on first use.
- For a review, lead with the verdict and the bugs that silently ruin a run, then the rest;
  `examples/05-reviewing-a-training-script.md` shows the shape.

## What is here

`examples/` has six worked answers with real script output and five planted-bug models
(`examples/README.md`). `evals/evals.json` is the regression set. `AUDIT.md` records what
this version found wrong in the previous one and how each fact was checked; `CHANGELOG.md`
lists the changes.

## Extend this skill

1. A new API claim: introspect it on the installed version, add the default to
   `EXPECTED` or the full parameter list to `SIGNATURES` in `scripts/verify_api.py`, and
   say "introspected from torch X" in the reference.
2. A new behavior claim (a warning, an error, a numeric property): pin it with a test in
   `tests/test_claims.py` so an upgrade that changes it fails.
3. A new walkthrough: put it under `## Walkthrough` with each output in a bare fence after
   its code block; `tests/test_walkthroughs.py` runs it.
4. A new lint rule: add the code to `scripts/lint_pytorch.py` with a trigger row and a
   near-miss row in `tests/test_lint_pytorch.py`.
5. A new torch release: run `pytest` and `verify_api.py --docs` on it, read the release
   notes, update `references/00-module-map.md`, log it in `CHANGELOG.md`.
   `tests/test_skill_structure.py` fails on broken paths, unrouted references, untested
   scripts, malformed evals and em dashes.
