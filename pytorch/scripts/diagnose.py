#!/usr/bin/env python3
"""Triage a PyTorch model that trains but does not learn, or goes NaN.

Point it at a Python file that defines two functions (the same contract as
`assets/test_model_template.py`):

    make_model() -> nn.Module
    make_batch() -> (inputs, targets)          # one batch, any shapes the model accepts
    loss_fn = nn.CrossEntropyLoss()            # optional module-level name; this is the default

Subcommands:

    python scripts/diagnose.py overfit my_model.py [--steps 200 --lr 3e-3 --floor F]
        Train on that one batch. A correct model, loss and optimizer step drive one batch's
        loss toward its floor (zero for hard targets; the smoothed-target entropy for
        CrossEntropyLoss(label_smoothing=...), computed automatically; --floor otherwise).
        "memorized" also requires beating a constant-prediction baseline, so a network that
        only learned the target mean is not passed. Verdict: memorized, stuck, or diverged
        (exit 1 unless memorized).

    python scripts/diagnose.py grads my_model.py
        One forward and backward. Per-parameter table: gradient norm, norm relative to the
        weight, and a flag (NONE = never reached the loss, ZERO, NONFINITE, HUGE). Also lists
        tensors and modules hidden from PyTorch: plain tensor attributes (not Parameter or
        buffer) and modules kept in a plain list or dict (exit 1 on any flag).

    python scripts/diagnose.py nan my_model.py
        Hooks every module and reports the first one whose forward output contains NaN or
        Inf, then whether any gradient is non-finite (exit 1 if one is found).

    python scripts/diagnose.py all my_model.py      # grads, nan, then overfit

Add --json for machine-readable output. Everything runs on CPU unless --device is given.

What it cannot see: bugs in your real data pipeline (it uses the one batch you give it),
learning-rate schedules, and anything that only appears at scale. If overfit passes but the
real run does not learn, the bug is downstream of the model: data, labels, augmentation, LR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

HUGE_GRAD = 1e3


def load_target(path: str):
    """Import the user's file by path and return (model, batch, loss_fn)."""
    file = Path(path)
    spec = importlib.util.spec_from_file_location(f"_diag_{file.stem}", file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("make_model", "make_batch"):
        if not callable(getattr(mod, name, None)):
            raise SystemExit(f"{path} must define {name}()")
    loss_fn = getattr(mod, "loss_fn", None) or nn.CrossEntropyLoss()
    return mod.make_model, mod.make_batch, loss_fn


def _to(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Hidden state: tensors and modules PyTorch cannot see
# --------------------------------------------------------------------------- #
def hidden_state(model: nn.Module) -> list[str]:
    """Plain tensor attributes and modules stored in plain lists/dicts/tuples."""
    found = []
    for mod_name, mod in model.named_modules():
        registered = set(mod._parameters) | set(mod._buffers) | set(mod._modules)
        for attr, value in vars(mod).items():
            if attr.startswith("_") or attr in registered:
                continue
            where = f"{mod_name + '.' if mod_name else ''}{attr}"
            if isinstance(value, torch.Tensor):
                found.append(f"{where}: plain tensor attribute {tuple(value.shape)} "
                             "(not moved by .to(), not saved, not trained; use nn.Parameter "
                             "or register_buffer)")
            elif isinstance(value, (list, tuple)) and any(isinstance(v, nn.Module) for v in value):
                found.append(f"{where}: modules in a plain {type(value).__name__} "
                             "(their parameters are invisible to the optimizer; use nn.ModuleList)")
            elif isinstance(value, dict) and any(isinstance(v, nn.Module) for v in value.values()):
                found.append(f"{where}: modules in a plain dict (use nn.ModuleDict)")
    return found


# --------------------------------------------------------------------------- #
# grads
# --------------------------------------------------------------------------- #
def grad_report(model: nn.Module, batch, loss_fn) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    x, y = batch
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    warnings = []
    if (isinstance(out, torch.Tensor) and out.is_floating_point() and out.dim() >= 2
            and isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss, nn.BCEWithLogitsLoss))):
        o = out.detach().float()
        if (o >= 0).all() and torch.allclose(o.sum(-1), torch.ones_like(o.sum(-1)), atol=1e-4):
            warnings.append("model output rows are non-negative and sum to 1: the head applies "
                            "softmax, but the loss expects raw logits (it applies its own)")
        elif isinstance(loss_fn, nn.BCEWithLogitsLoss) and (o >= 0).all() and (o <= 1).all():
            warnings.append("model output is inside [0, 1]: a sigmoid head before "
                            "BCEWithLogitsLoss applies the sigmoid twice")
    rows = []
    for name, p in model.named_parameters():
        row = {"name": name, "shape": list(p.shape), "requires_grad": p.requires_grad}
        if not p.requires_grad:
            row["flag"] = "FROZEN"
        elif p.grad is None:
            row["flag"] = "NONE"
        else:
            g = p.grad.detach().float()
            gn, pn = g.norm().item(), p.detach().float().norm().item()
            row.update(grad_norm=gn, rel=gn / pn if pn > 0 else math.inf)
            if not math.isfinite(gn):
                row["flag"] = "NONFINITE"
            elif gn == 0.0:
                row["flag"] = "ZERO"
            elif gn > HUGE_GRAD:
                row["flag"] = "HUGE"
            else:
                row["flag"] = "ok"
        rows.append(row)
    return {"loss": loss.item(), "params": rows, "hidden": hidden_state(model),
            "warnings": warnings}


# --------------------------------------------------------------------------- #
# nan
# --------------------------------------------------------------------------- #
def _nonfinite(t) -> bool:
    if isinstance(t, torch.Tensor):
        return t.is_floating_point() and not torch.isfinite(t).all().item()
    if isinstance(t, (list, tuple)):
        return any(_nonfinite(v) for v in t)
    if isinstance(t, dict):
        return any(_nonfinite(v) for v in t.values())
    return False


def nan_report(model: nn.Module, batch, loss_fn) -> dict:
    x, y = batch
    first: list[str] = []
    handles = []

    def hook(name):
        def fn(module, inputs, output):
            if not first and _nonfinite(output):
                first.append(f"{name or '<root>'} ({type(module).__name__})")
        return fn

    for name, m in model.named_modules():
        handles.append(m.register_forward_hook(hook(name)))
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        bad_grads = None                       # None = not checked (loss already non-finite)
        if loss.requires_grad and math.isfinite(loss.item()):
            loss.backward()
            bad_grads = [n for n, p in model.named_parameters()
                         if p.grad is not None and not torch.isfinite(p.grad).all()]
    finally:
        for h in handles:
            h.remove()
    return {"input_nonfinite": _nonfinite(x), "first_nonfinite_module": first[0] if first else None,
            "loss": loss.item(), "nonfinite_grads": bad_grads}


# --------------------------------------------------------------------------- #
# overfit
# --------------------------------------------------------------------------- #
def loss_floor(loss_fn, out: torch.Tensor, y: torch.Tensor) -> float:
    """The lowest loss a perfect memorizer can reach, when it is known and above zero.

    CrossEntropyLoss with label_smoothing=eps and integer targets cannot go below the entropy
    of the smoothed target: (1 - eps + eps/C) on the true class and eps/C on the others.
    Everything else is taken as 0; pass --floor for other cases (soft targets, KL losses).
    """
    eps = getattr(loss_fn, "label_smoothing", 0.0)
    if isinstance(loss_fn, nn.CrossEntropyLoss) and eps > 0 and not y.is_floating_point():
        c = out.shape[-1]
        hi, lo = 1 - eps + eps / c, eps / c
        return -(hi * math.log(hi) + (c - 1) * lo * math.log(lo))
    return 0.0


def constant_baseline(loss_fn, out: torch.Tensor, y: torch.Tensor, steps: int = 300) -> float:
    """Best loss reachable by one learned output shared by every example (inputs ignored).

    A model whose loss does not get clearly below this is not using its inputs: it has
    learned the target mean or the class prior, which is what a detached or dead network
    still manages.
    """
    c = torch.zeros_like(out[:1]).requires_grad_(True)
    opt = torch.optim.AdamW([c], lr=0.1, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(c.expand_as(out), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return loss_fn(c.expand_as(out), y).item()


def overfit(model: nn.Module, batch, loss_fn, steps: int, lr: float, floor: float | None = None) -> dict:
    x, y = batch
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"verdict": "stuck", "reason": "no parameter requires grad", "curve": []}
    with torch.no_grad():
        out0 = model(x)
    baseline = constant_baseline(loss_fn, out0.detach(), y) if isinstance(out0, torch.Tensor) else None
    if floor is None:
        floor = loss_floor(loss_fn, out0, y) if isinstance(out0, torch.Tensor) else 0.0
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    curve = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not math.isfinite(loss.item()):
            curve.append(loss.item())
            return {"verdict": "diverged", "reason": f"loss became {loss.item()} at step {step}",
                    "curve": curve, "baseline": baseline, "floor": floor}
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            curve.append(loss.item())
    first, last = curve[0], curve[-1]
    near_floor = (last - floor) < 0.1 * (first - floor) or (last - floor) < 1e-2
    beats_baseline = baseline is None or last < baseline - 0.5 * (baseline - floor)
    summary = f"loss {first:.4f} -> {last:.4f} in {steps} steps"
    if baseline is not None:
        summary += f"; constant-prediction baseline {baseline:.4f}"
    if floor > 0:
        summary += f"; floor {floor:.4f}"
    if near_floor and beats_baseline:
        return {"verdict": "memorized", "reason": summary, "curve": curve,
                "baseline": baseline, "floor": floor}
    if baseline is not None and last >= floor + 0.9 * (baseline - floor):
        why = ("no better than predicting one constant for every example, so the model is not "
               "using its inputs: look for detached or frozen layers and dead activations (grads)")
    else:
        why = ("a correct model memorizes one batch, so look at gradient flow (grads), the "
               "loss/target convention, and frozen or detached layers; if the loss has a floor "
               "above zero this script does not know (soft targets, duplicate inputs with "
               "different labels), pass --floor")
    return {"verdict": "stuck", "reason": f"{summary}; {why}", "curve": curve,
            "baseline": baseline, "floor": floor}


# --------------------------------------------------------------------------- #
def _print_grads(r: dict) -> None:
    print(f"loss {r['loss']:.4f}")
    print(f"{'parameter':<40} {'shape':<16} {'grad norm':>11} {'grad/weight':>12}  flag")
    for row in r["params"]:
        gn = f"{row['grad_norm']:.3e}" if "grad_norm" in row else "-"
        rel = f"{row['rel']:.3e}" if "rel" in row else "-"
        print(f"{row['name']:<40} {str(tuple(row['shape'])):<16} {gn:>11} {rel:>12}  {row['flag']}")
    for h in r["hidden"]:
        print(f"HIDDEN {h}")
    for w in r["warnings"]:
        print(f"WARNING {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["overfit", "grads", "nan", "all"])
    ap.add_argument("target", help="file defining make_model() and make_batch()")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--floor", type=float, default=None,
                    help="known lowest achievable loss (default: 0, or the label-smoothing floor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    make_model, make_batch, loss_fn = load_target(args.target)
    device = torch.device(args.device)
    result: dict = {}
    problems = 0

    def fresh():
        torch.manual_seed(args.seed)
        return make_model().to(device), _to(make_batch(), device)

    if args.command in ("grads", "all"):
        model, batch = fresh()
        r = grad_report(model, batch, loss_fn)
        result["grads"] = r
        problems += (sum(row["flag"] not in ("ok", "FROZEN") for row in r["params"])
                     + len(r["hidden"]) + len(r["warnings"]))
        if not args.json:
            print("== grads"); _print_grads(r)
    if args.command in ("nan", "all"):
        model, batch = fresh()
        r = nan_report(model, batch, loss_fn)
        result["nan"] = r
        problems += bool(r["input_nonfinite"] or r["first_nonfinite_module"] or r["nonfinite_grads"])
        if not args.json:
            print("== nan")
            print(f"input non-finite: {r['input_nonfinite']}")
            print(f"first module with non-finite output: {r['first_nonfinite_module'] or 'none'}")
            print(f"loss: {r['loss']}")
            grads = r["nonfinite_grads"]
            print("parameters with non-finite grads: " + (
                "not checked (the loss is already non-finite)" if grads is None
                else ", ".join(grads) or "none"))
    if args.command in ("overfit", "all"):
        model, batch = fresh()
        r = overfit(model, batch, loss_fn, args.steps, args.lr, args.floor)
        result["overfit"] = r
        problems += r["verdict"] != "memorized"
        if not args.json:
            print("== overfit")
            print(f"verdict: {r['verdict']} ({r['reason']})")
            print("curve:", " ".join(f"{v:.4f}" for v in r["curve"]))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
