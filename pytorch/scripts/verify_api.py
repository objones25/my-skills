#!/usr/bin/env python3
"""Check this skill's documented signatures against the installed PyTorch.

Every default recorded here was introspected from torch 2.15.0 / torchvision 0.29.2
with `inspect.signature`. On a different version, run this to find out exactly which
of them moved, instead of trusting the reference docs blindly.

    python scripts/verify_api.py              # report drift in the pinned tables
    python scripts/verify_api.py --all        # print every checked value too
    python scripts/verify_api.py --docs       # also scan references/*.md for API names
                                              # that do not resolve and signature claims
                                              # whose defaults disagree with the install
    python scripts/verify_api.py --json       # machine-readable result

Exit codes: 0 = no drift, 1 = drift found, 2 = torch missing.

How --docs reads the references: it resolves every backticked `torch.*`, `nn.*`, `F.*`,
`v2.*` name, and it looks at backtick spans `name(...)` and table rows whose first code
cell is a name and whose next cell is a `(...)` parameter list.
  - A keyword that is not a parameter of the installed callable is always reported
    (unless the callable takes **kwargs), in signatures and usage examples alike.
  - Default values are compared only when the span looks like a signature: at least two
    comparable keywords, and at least half of them equal to the installed defaults. A
    usage example such as `AdamW(params, lr=3e-4)` is therefore not reported.
Blind spots: a signature span with one comparable keyword, or with more wrong defaults
than right ones, is read as a usage example and its defaults are not compared; positional
parameters and signatures written outside backticks are not checked. The pinned EXPECTED
table below covers the defaults that matter most.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import sys
import warnings
from pathlib import Path

REFERENCE_VERSION = "torch 2.15.0 / torchvision 0.29.2"
MISSING = object()

# (dotted path, parameter name, value introspected on torch 2.15.0)
EXPECTED: list[tuple[str, str, object]] = [
    # --- optimizers: the defaults most often misquoted -----------------------
    ("torch.optim.AdamW", "lr", 0.001),
    ("torch.optim.AdamW", "weight_decay", 0.01),
    ("torch.optim.AdamW", "betas", (0.9, 0.999)),
    ("torch.optim.AdamW", "eps", 1e-08),
    ("torch.optim.Adam", "weight_decay", 0),
    ("torch.optim.Adam", "decoupled_weight_decay", False),
    ("torch.optim.SGD", "lr", 0.001),
    ("torch.optim.SGD", "momentum", 0),
    ("torch.optim.SGD", "nesterov", False),
    ("torch.optim.Muon", "lr", 0.001),
    ("torch.optim.Muon", "weight_decay", 0.1),
    ("torch.optim.Muon", "momentum", 0.95),
    ("torch.optim.Muon", "nesterov", True),
    ("torch.optim.Muon", "ns_steps", 5),
    ("torch.optim.Adafactor", "lr", 0.01),
    # --- losses --------------------------------------------------------------
    ("torch.nn.CrossEntropyLoss", "ignore_index", -100),
    ("torch.nn.CrossEntropyLoss", "label_smoothing", 0.0),
    ("torch.nn.CrossEntropyLoss", "reduction", "mean"),
    ("torch.nn.BCEWithLogitsLoss", "pos_weight", None),
    ("torch.nn.SmoothL1Loss", "beta", 1.0),
    ("torch.nn.HuberLoss", "delta", 1.0),
    ("torch.nn.LinearCrossEntropyLoss", "bias", False),
    ("torch.nn.LinearCrossEntropyLoss", "ignore_index", None),
    ("torch.nn.functional.cross_entropy", "label_smoothing", 0.0),
    # --- layers --------------------------------------------------------------
    ("torch.nn.Linear", "bias", True),
    ("torch.nn.Conv2d", "padding", 0),
    ("torch.nn.Conv2d", "stride", 1),
    ("torch.nn.BatchNorm2d", "momentum", 0.1),
    ("torch.nn.Dropout", "p", 0.5),
    ("torch.nn.LayerNorm", "eps", 1e-05),
    ("torch.nn.RMSNorm", "eps", None),
    ("torch.nn.MultiheadAttention", "batch_first", False),
    ("torch.nn.Flatten", "start_dim", 1),
    ("torch.nn.functional.dropout", "training", True),
    ("torch.nn.functional.batch_norm", "training", False),
    # --- data ----------------------------------------------------------------
    ("torch.utils.data.DataLoader", "batch_size", 1),
    ("torch.utils.data.DataLoader", "num_workers", 0),
    ("torch.utils.data.DataLoader", "pin_memory", False),
    ("torch.utils.data.DataLoader", "drop_last", False),
    ("torch.utils.data.DataLoader", "persistent_workers", False),
    ("torch.utils.data.DataLoader", "in_order", True),
    # --- amp / devices --------------------------------------------------------
    ("torch.autocast", "enabled", True),
    ("torch.autocast", "dtype", None),
    ("torch.amp.GradScaler", "device", "cuda"),
    ("torch.amp.GradScaler", "init_scale", 65536.0),
    ("torch.accelerator.current_accelerator", "check_available", False),
    # --- serialization / compile / export -------------------------------------
    ("torch.load", "weights_only", None),  # None -> True unless pickle_module is set
    ("torch.load", "mmap", None),
    ("torch.compile", "fullgraph", False),
    ("torch.compile", "dynamic", None),
    ("torch.compile", "backend", None),
    ("torch.compile", "dynamic_shapes", None),  # was shapes_spec in 2.13
    ("torch.export.export", "strict", False),
    ("torch.utils.checkpoint.checkpoint", "use_reentrant", None),
    ("torch.utils.checkpoint.checkpoint", "function", None),  # curried form, 2.14+
    # --- attention -------------------------------------------------------------
    ("torch.nn.attention.flex_attention.flex_attention", "enable_gqa", False),
    ("torch.nn.attention.flex_attention.create_block_mask", "BLOCK_SIZE", 128),
    ("torch.nn.attention.sdpa_kernel", "set_priority", False),
    # --- distributed ------------------------------------------------------------
    ("torch.distributed.fsdp.fully_shard", "reshard_after_forward", None),
    ("torch.distributed.fsdp.fully_shard", "mesh", None),
    ("torch.distributed.fsdp.MixedPrecisionPolicy", "reduce_dtype", None),
    ("torch.nn.parallel.DistributedDataParallel", "find_unused_parameters", False),
    ("torch.nn.parallel.DistributedDataParallel", "static_graph", False),
    ("torch.nn.parallel.DistributedDataParallel", "lazy_bucket_allocation", False),
    # --- testing ------------------------------------------------------------------
    ("torch.testing.assert_close", "equal_nan", False),
    ("torch.autograd.gradcheck", "fast_mode", False),
    # --- torchvision --------------------------------------------------------------
    ("torchvision.transforms.v2.ToDtype", "scale", False),
    ("torchvision.transforms.v2.Normalize", "inplace", False),
]

# Symbols the references assert exist (True) or assert are gone (False) on 2.15.
PRESENCE: list[tuple[str, bool, str]] = [
    ("torch.nn.LinearCrossEntropyLoss", True, "added in 2.13"),
    ("torch.nn.functional.linear_cross_entropy", True, "added in 2.13"),
    ("torch.optim.Muon", True, "in torch.optim since 2.9"),
    ("torch.optim.functional.adamw", True, "public functional optimizers, 2.15"),
    ("torch.accelerator.Graph", True, "device-agnostic graph capture"),
    ("torch.accelerator.random.get_rng_state", True, "2.14; note the .random submodule"),
    ("torch.compiler.set_default_backend", True, "added in 2.13"),
    ("torch.compiler.precompile.capture", True, "precompile became a module in 2.15"),
    ("torch.distributed.all_gather_single", True, "renamed in 2.13"),
    ("torch.distributed.reduce_scatter_single", True, "renamed in 2.13"),
    ("torch.nn.attention.flex_attention.flex_attention", True, "FlexAttention"),
    ("torch.distributed.fsdp.fully_shard", True, "FSDP2"),
    ("torch.distributed.fsdp.FSDPModule.finalize_backward", True, "2.15"),
    ("torch.nn.parallel.DistributedDataParallel.finalize_backward", True, "2.15"),
    ("torch.profiler.CuspyConfig", True, "renamed from the CUPTI monitor in 2.15"),
    ("torch.foreach.mul_", True, "public foreach module (beta), 2.15"),
    ("torch.library.custom_op", True, "custom ops"),
    ("torch.library.opcheck", True, "custom op test helper"),
    ("torch.export.dynamic_spec", False, "lives in torch.fx.experimental.dynamic_spec, not torch.export"),
    ("torch.accelerator.initial_seed", False, "lives in torch.accelerator.random"),
]

# Full parameter-name lists, introspected from torch 2.15.0. Comparing whole lists
# catches renames, additions and removals that single-default checks miss.
SIGNATURES: dict[str, tuple[str, ...]] = {
    'torch.utils.data.DataLoader': ('dataset', 'batch_size', 'shuffle', 'sampler', 'batch_sampler', 'num_workers', 'collate_fn', 'pin_memory', 'drop_last', 'timeout', 'worker_init_fn', 'multiprocessing_context', 'generator', 'prefetch_factor', 'persistent_workers', 'pin_memory_device', 'in_order'),
    'torch.nn.MaxPool2d': ('kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode'),
    'torch.nn.Linear': ('in_features', 'out_features', 'bias', 'device', 'dtype'),
    'torch.nn.Conv2d': ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype'),
    'torch.nn.BatchNorm2d': ('num_features', 'eps', 'momentum', 'affine', 'track_running_stats', 'device', 'dtype', 'bias'),
    'torch.nn.LayerNorm': ('normalized_shape', 'eps', 'elementwise_affine', 'bias', 'device', 'dtype'),
    'torch.nn.RMSNorm': ('normalized_shape', 'eps', 'elementwise_affine', 'device', 'dtype'),
    'torch.nn.Embedding': ('num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', '_weight', '_freeze', 'device', 'dtype'),
    'torch.nn.MultiheadAttention': ('embed_dim', 'num_heads', 'dropout', 'bias', 'add_bias_kv', 'add_zero_attn', 'kdim', 'vdim', 'batch_first', 'device', 'dtype'),
    'torch.nn.CrossEntropyLoss': ('weight', 'size_average', 'ignore_index', 'reduce', 'reduction', 'label_smoothing'),
    'torch.nn.BCEWithLogitsLoss': ('weight', 'size_average', 'reduce', 'reduction', 'pos_weight'),
    'torch.nn.LinearCrossEntropyLoss': ('in_features', 'num_classes', 'out_features', 'bias', 'device', 'dtype', 'reduction', 'weight', 'ignore_index', 'label_smoothing', 'options'),
    'torch.nn.functional.linear_cross_entropy': ('input', 'linear_weight', 'target', 'linear_bias', 'weight', 'reduction', 'ignore_index', 'label_smoothing', 'options'),
    'torch.optim.AdamW': ('params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused'),
    'torch.optim.Adam': ('params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'maximize', 'capturable', 'differentiable', 'fused', 'decoupled_weight_decay'),
    'torch.optim.SGD': ('params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach', 'differentiable', 'fused'),
    'torch.optim.Muon': ('params', 'lr', 'weight_decay', 'momentum', 'nesterov', 'ns_coefficients', 'eps', 'ns_steps', 'adjust_lr_fn'),
    'torch.optim.Adafactor': ('params', 'lr', 'beta2_decay', 'eps', 'd', 'weight_decay', 'foreach', 'maximize'),
    'torch.optim.lr_scheduler.OneCycleLR': ('optimizer', 'max_lr', 'total_steps', 'epochs', 'steps_per_epoch', 'pct_start', 'anneal_strategy', 'cycle_momentum', 'base_momentum', 'max_momentum', 'div_factor', 'final_div_factor', 'three_phase', 'last_epoch'),
    'torch.optim.lr_scheduler.CosineAnnealingLR': ('optimizer', 'T_max', 'eta_min', 'last_epoch'),
    'torch.amp.GradScaler': ('device', 'init_scale', 'growth_factor', 'backoff_factor', 'growth_interval', 'enabled'),
    'torch.autocast': ('device_type', 'dtype', 'enabled', 'cache_enabled'),
    'torch.load': ('f', 'map_location', 'pickle_module', 'weights_only', 'mmap', 'pickle_load_args'),
    'torch.compile': ('model', 'fullgraph', 'dynamic', 'backend', 'mode', 'options', 'name', 'disable', 'recompile_limit', 'isolate_recompiles', 'dynamic_shapes'),
    'torch.export.export': ('mod', 'args', 'kwargs', 'dynamic_shapes', 'strict', 'preserve_module_call_signature', 'prefer_deferred_runtime_asserts_over_guards'),
    'torch.utils.checkpoint.checkpoint': ('function', 'args', 'use_reentrant', 'preserve_rng_state', 'context_fn', 'determinism_check', 'debug', 'early_stop', 'respect_saved_tensors_hooks', 'kwargs'),
    'torch.profiler.profile': ('activities', 'schedule', 'on_trace_ready', 'record_shapes', 'profile_memory', 'with_stack', 'with_flops', 'with_modules', 'experimental_config', 'execution_trace_observer', 'acc_events', 'custom_trace_id_callback', 'post_processing_timeout_s'),
    'torch.profiler.schedule': ('wait', 'warmup', 'active', 'repeat', 'skip_first', 'skip_first_wait'),
    'torch.distributed.fsdp.fully_shard': ('module', 'mesh', 'reshard_after_forward', 'shard_placement_fn', 'mp_policy', 'offload_policy', 'ignored_params', 'dp_mesh_dims'),
    'torch.distributed.fsdp.MixedPrecisionPolicy': ('param_dtype', 'reduce_dtype', 'output_dtype', 'cast_forward_inputs', 'param_dtype_override_fn'),
    'torch.nn.parallel.DistributedDataParallel': ('module', 'device_ids', 'output_device', 'dim', 'broadcast_buffers', 'init_sync', 'process_group', 'bucket_cap_mb', 'find_unused_parameters', 'check_reduction', 'gradient_as_bucket_view', 'static_graph', 'delay_all_reduce_named_params', 'param_to_hook_all_reduce', 'mixed_precision', 'device_mesh', 'skip_all_reduce_unused_params', 'bucket_cap_mb_list', 'batched_grad_copy', 'forward_sync_buffers', 'lazy_bucket_allocation'),
    'torch.distributed.device_mesh.init_device_mesh': ('device_type', 'mesh_shape', 'mesh_dim_names', 'backend_override'),
    'torch.nn.attention.flex_attention.flex_attention': ('query', 'key', 'value', 'score_mod', 'block_mask', 'scale', 'enable_gqa', 'return_lse', 'kernel_options', 'return_aux'),
    'torch.nn.attention.flex_attention.create_block_mask': ('mask_mod', 'B', 'H', 'Q_LEN', 'KV_LEN', 'device', 'BLOCK_SIZE', '_compile', 'separate_full_blocks', 'compute_dq_write_order', 'dq_kv_order'),
    'torch.nn.attention.sdpa_kernel': ('backends', 'set_priority'),
    'torch.autograd.gradcheck': ('func', 'inputs', 'eps', 'atol', 'rtol', 'raise_exception', 'nondet_tol', 'check_undefined_grad', 'check_grad_dtypes', 'check_batched_grad', 'check_batched_forward_grad', 'check_forward_ad', 'check_backward_ad', 'fast_mode', 'masked'),
    'torch.library.custom_op': ('name', 'fn', 'mutates_args', 'device_types', 'schema', 'tags'),
    'torch.testing.assert_close': ('actual', 'expected', 'allow_subclasses', 'rtol', 'atol', 'equal_nan', 'check_device', 'check_dtype', 'check_layout', 'check_stride', 'msg'),
    'torch.func.functional_call': ('module', 'parameter_and_buffer_dicts', 'args', 'kwargs', 'tie_weights', 'strict'),
}

# What each known drift means, so a user on an older install knows it is expected.
KNOWN_CHANGES: dict[str, str] = {
    "torch.compile": "2.14 renamed shapes_spec to dynamic_shapes",
    "torch.utils.checkpoint.checkpoint": "2.14 added preserve_rng_state, respect_saved_tensors_hooks and the curried form (function=None)",
    "torch.profiler.profile": "2.14 removed use_cuda",
    "torch.nn.parallel.DistributedDataParallel": "2.15 added lazy_bucket_allocation",
    "torch.distributed.fsdp.MixedPrecisionPolicy": "2.15 added param_dtype_override_fn",
}

ALIASES = {
    "nn": "torch.nn", "F": "torch.nn.functional", "v2": "torchvision.transforms.v2",
    "dist": "torch.distributed", "dcp": "torch.distributed.checkpoint",
    "bench": "torch.utils.benchmark",
}


def resolve(path: str):
    parts = path.split(".")
    if parts[0] in ALIASES:
        parts = ALIASES[parts[0]].split(".") + parts[1:]
    # Prefer the attribute route (torch.autograd.gradcheck is both a module and the
    # function inside it; the function is what the docs mean), then a whole module.
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
            return obj
        except AttributeError as exc:
            try:
                return importlib.import_module(".".join(parts))
            except ModuleNotFoundError as missing:
                if (missing.name or "").split(".")[0] not in {"torch", "torchvision"}:
                    raise missing from None   # an optional dependency, e.g. tensorboard
                raise exc from None
            except ImportError:
                raise exc from None
    return importlib.import_module(".".join(parts))


def _params(path: str):
    obj = resolve(path)
    target = obj.__init__ if inspect.isclass(obj) else obj
    return inspect.signature(target).parameters


def actual_params(path: str) -> tuple[str, ...]:
    return tuple(n for n in _params(path) if n != "self")


def signature_drift(path: str, want: tuple[str, ...]) -> str | None:
    """Return a one-line description of how the installed parameter list differs, or None."""
    got = actual_params(path)
    if got == want:
        return None
    added = [n for n in got if n not in want]
    removed = [n for n in want if n not in got]
    if not added and not removed:
        return f"{path}: same parameters, new order {got}"
    return f"{path}: added {added or '-'}, removed {removed or '-'}"


def actual_default(path: str, param: str):
    params = _params(path)
    if param not in params:
        return MISSING
    default = params[param].default
    return MISSING if default is inspect.Parameter.empty else default


# --------------------------------------------------------------------------- #
# --docs: scan the reference files
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(r"`((?:torch|nn|F|v2|dist|dcp|torchvision|torchao)\.[A-Za-z_][\w.]*\*?)")
CALL_RE = re.compile(r"`((?:torch|nn|F|v2|torchvision)\.[A-Za-z_][\w.]*)\(([^`]*)\)`")
ROW_RE = re.compile(r"^\|\s*`((?:torch|nn|F|v2|torchvision)\.[A-Za-z_][\w.]*)`[^|]*\|\s*`\(([^`]*)\)`", re.M)
KW_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)=([^,()\[\]]+(?:\([^()]*\))?)")


def _literal(text: str):
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return MISSING


def scan_docs(ref_dir: Path) -> list[dict]:
    findings: list[dict] = []
    for md in sorted(ref_dir.glob("*.md")):
        text = md.read_text()
        for name in sorted({m.group(1).rstrip(".") for m in NAME_RE.finditer(text)}):
            if name.endswith((".html", ".md")) or "*" in name:
                continue
            try:
                resolve(name)
            except ModuleNotFoundError as exc:
                root = (exc.name or "").split(".")[0]
                if root and root not in {"torch", "torchvision"}:
                    continue  # an optional dependency (tensorboard, torchao) is not installed
                findings.append({"file": md.name, "kind": "unresolved", "name": name,
                                 "detail": type(exc).__name__})
            except (ImportError, AttributeError) as exc:
                findings.append({"file": md.name, "kind": "unresolved", "name": name,
                                 "detail": type(exc).__name__})
        claims = [(m.group(1), m.group(2)) for m in CALL_RE.finditer(text)]
        claims += [(m.group(1), m.group(2)) for m in ROW_RE.finditer(text)]
        for name, args in claims:
            findings.extend(_check_claim(md.name, name, args))
    return findings


def _check_claim(file: str, name: str, args: str) -> list[dict]:
    try:
        params = _params(name)
    except (ImportError, AttributeError, ValueError, TypeError):
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()) and len(params) <= 3:
        return []  # (*args, **kwargs) wrappers such as nn.LSTM cannot be checked
    pairs = [(k, v) for k, v in KW_RE.findall(args)]
    if not pairs:
        return []
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    unknown, mismatched, matches, compared = [], [], 0, 0
    for key, raw in pairs:
        if key not in params:
            if not accepts_kwargs:   # a misspelled or removed keyword is wrong in any snippet
                unknown.append({"file": file, "kind": "unknown-param", "name": name,
                                "detail": f"{key}= is not a parameter on this install"})
            continue
        claimed, actual = _literal(raw), params[key].default
        if claimed is MISSING or claimed is Ellipsis or actual is inspect.Parameter.empty:
            continue
        compared += 1
        if claimed == actual:
            matches += 1
        else:
            mismatched.append({"file": file, "kind": "default", "name": name,
                               "detail": f"{key}: doc says {claimed!r}, installed has {actual!r}"})
    # A span is read as a signature claim when at least half of its comparable keywords
    # match the installed defaults; otherwise it is a usage example and its non-default
    # values are intentional.
    is_signature = compared >= 2 and matches * 2 >= compared
    return unknown + (mismatched if is_signature else [])


# --------------------------------------------------------------------------- #
def run(check_docs: bool, show_all: bool) -> dict:
    import torch

    out: dict = {"torch": torch.__version__, "reference": REFERENCE_VERSION,
                 "drift": [], "ok": [], "docs": []}
    try:
        import torchvision
        out["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        out["torchvision"] = None

    for path, param, want in EXPECTED:
        try:
            got = actual_default(path, param)
        except (ImportError, AttributeError) as exc:
            out["drift"].append(f"{path}.{param}: symbol unavailable ({type(exc).__name__})")
            continue
        if got is MISSING or got != want:
            shown = "<absent>" if got is MISSING else repr(got)
            out["drift"].append(f"{path}({param}=...): doc says {want!r}, installed has {shown}")
        elif show_all:
            out["ok"].append(f"{path}({param}={want!r})")

    for path, should_exist, note in PRESENCE:
        try:
            resolve(path)
            exists = True
        except (ImportError, AttributeError):
            exists = False
        if exists != should_exist:
            verb = "missing but expected" if should_exist else "present but documented as absent"
            out["drift"].append(f"{path}: {verb} ({note})")
        elif show_all:
            out["ok"].append(f"{path} {'present' if exists else 'absent'} ({note})")

    for path, want in SIGNATURES.items():
        try:
            msg = signature_drift(path, want)
        except (ImportError, AttributeError) as exc:
            out["drift"].append(f"{path}: symbol unavailable ({type(exc).__name__})")
            continue
        if msg:
            note = KNOWN_CHANGES.get(path)
            out["drift"].append(msg + (f"   [known: {note}]" if note else ""))
        elif show_all:
            out["ok"].append(f"{path} full signature ({len(want)} params)")

    if hasattr(torch.randn(1), "names"):
        out["drift"].append("Tensor.names present: named tensors were removed in 2.13")

    if check_docs:
        out["docs"] = scan_docs(Path(__file__).resolve().parent.parent / "references")
    out["counts"] = {"defaults": len(EXPECTED), "symbols": len(PRESENCE),
                     "signatures": len(SIGNATURES)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="print matches too")
    ap.add_argument("--docs", action="store_true", help="scan references/*.md as well")
    ap.add_argument("--json", action="store_true", help="print JSON")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"torch not importable: {exc}")
        return 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(args.docs, args.all)

    bad = bool(result["drift"] or result["docs"])
    if args.json:
        print(json.dumps(result, indent=2))
        return 1 if bad else 0

    print(f"installed: torch {result['torch']}")
    print(f"           torchvision {result['torchvision'] or '-- not installed (its checks are skipped)'}")
    print(f"reference: {result['reference']}\n")
    for line in result["ok"]:
        print(f"  ok  {line}")
    c = result["counts"]
    print(f"\nchecked {c['defaults']} defaults + {c['symbols']} symbols + {c['signatures']} full signatures"
          + (" + reference files" if args.docs else ""))
    if not bad:
        print("no drift - the reference docs match this install.")
        return 0
    if result["drift"]:
        print(f"{len(result['drift'])} difference(s) - trust the installed value, not the docs:\n")
        print("\n".join(f"  {d}" for d in result["drift"]))
    if result["docs"]:
        print(f"\n{len(result['docs'])} reference-file finding(s):\n")
        for f in result["docs"]:
            print(f"  {f['file']}: {f['kind']}: {f['name']}: {f['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
