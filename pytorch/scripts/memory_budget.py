#!/usr/bin/env python3
"""Per-rank memory for weights, gradients and optimizer state, before activations.

Answers "does this fit, and does it need DDP or FSDP2?" with arithmetic you can check,
and, given a real model, measures the optimizer state instead of assuming it.

    python scripts/memory_budget.py --params 7e9 --optimizer adamw --world 8
    python scripts/memory_budget.py --model my_model.py --optimizer muon --measure

    --params N            parameter count (or use --model FILE, which defines make_model())
    --optimizer           adamw | adam | sgd | sgd-momentum | muon | adafactor
    --precision           fp32        weights, grads, state all fp32 (also what bf16 autocast
                                      keeps: autocast casts on the fly, parameters stay fp32)
                          bf16        weights, grads and state all bf16 (pure low precision)
                          bf16-master bf16 weights and grads plus an fp32 master copy and fp32
                                      state (the ZeRO / mixed-precision-training layout)
    --world N             number of ranks; with --strategy fsdp every term is divided by N
    --strategy            ddp (replicate) | fsdp (shard weights, grads and state)
    --gpu-gib G           compare against a device size (default 80)
    --measure             build the model, take one optimizer step on CPU, and report the
                          optimizer state bytes PyTorch actually allocated

Activations are not counted: they depend on batch size, sequence length and checkpointing,
and they are usually the term that decides the batch size. Measure them with
torch.accelerator.max_memory_allocated() after one real step.

Gradients and optimizer state are counted over trainable parameters only
(requires_grad=True), weights over all of them, so a frozen backbone or a LoRA setup is
sized correctly when given --model.

Muon keeps one momentum buffer per 2-D matrix and accepts only 2-D parameters. The split
here sends 2-D weights to Muon except nn.Embedding weights and any parameter whose name
matches --adamw-regex (pass e.g. 'lm_head' for an untied output layer); everything else,
including all 1-D parameters, goes to AdamW. It needs --model. Adafactor's factored state
depends on tensor shapes, so it also needs --model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

GIB = 2**30
STATE_COPIES = {"adamw": 2, "adam": 2, "sgd": 0, "sgd-momentum": 1}


def bytes_per_param(precision: str, optimizer_copies: float) -> dict[str, float]:
    """Return bytes per parameter for each term, given how many state copies the optimizer keeps."""
    if precision == "fp32":
        return {"weights": 4, "grads": 4, "master": 0, "optimizer": 4 * optimizer_copies}
    if precision == "bf16":
        return {"weights": 2, "grads": 2, "master": 0, "optimizer": 2 * optimizer_copies}
    if precision == "bf16-master":
        return {"weights": 2, "grads": 2, "master": 4, "optimizer": 4 * optimizer_copies}
    raise ValueError(precision)


def load_model(path: str):
    file = Path(path)
    spec = importlib.util.spec_from_file_location(f"_mb_{file.stem}", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_model()


def muon_split(model, adamw_regex: str | None):
    """(Muon params, AdamW params) among trainable ones; tied tensors appear once."""
    import re

    import torch.nn as nn

    embedding_weights = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        to_adamw = (p.ndim != 2 or id(p) in embedding_weights
                    or (adamw_regex is not None and re.search(adamw_regex, name)))
        (adamw if to_adamw else muon).append(p)
    return muon, adamw


def split_counts(model, adamw_regex: str | None = None) -> dict[str, int]:
    """Parameter counts; shared (tied) tensors are counted once, as parameters() does."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    muon, adamw = muon_split(model, adamw_regex)
    return {"total": total, "trainable": trainable,
            "muon": sum(p.numel() for p in muon), "muon_adamw": sum(p.numel() for p in adamw)}


def adafactor_state_elems(model) -> int:
    """Elements Adafactor keeps per trainable tensor of shape (..., R, C): a row factor of
    prod(shape[:-1]) and a column factor of prod(shape[:-2]) * C; a full vector for 1-D."""
    import math

    n = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            n += math.prod(p.shape[:-1]) + math.prod(p.shape[:-2]) * p.shape[-1]
        else:
            n += p.numel()
    return n


def measure_optimizer_state(model, optimizer: str, adamw_regex: str | None = None) -> int:
    """Take one real step on CPU and sum the bytes of every tensor in optimizer.state,
    except the scalar `step` counters."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    opts = []
    if optimizer == "muon":
        muon, rest = muon_split(model, adamw_regex)
        if muon:
            opts.append(torch.optim.Muon(muon))
        if rest:
            opts.append(torch.optim.AdamW(rest))
    else:
        cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD,
               "sgd-momentum": torch.optim.SGD, "adafactor": torch.optim.Adafactor}[optimizer]
        kw = {"momentum": 0.9} if optimizer == "sgd-momentum" else {}
        opts.append(cls(params, **kw))
    for p in params:
        p.grad = torch.zeros_like(p)
    for opt in opts:
        opt.step()
    total = 0
    for opt in opts:
        for state in opt.state.values():
            for key, v in state.items():
                if torch.is_tensor(v) and key != "step":   # step is a 0-dim counter
                    total += v.numel() * v.element_size()
    return total


def budget(args) -> dict:
    model = load_model(args.model) if args.model else None
    if model is not None:
        counts = split_counts(model, args.adamw_regex)
    else:
        counts = {"total": int(args.params), "trainable": int(args.params)}
    n, nt = counts["total"], counts["trainable"]
    notes = []
    if nt < n:
        notes.append(f"{n - nt:,} frozen parameters: weights only, no gradient or optimizer state")
    if args.optimizer in STATE_COPIES:
        copies = STATE_COPIES[args.optimizer]
    elif args.optimizer == "muon":
        if model is None:
            raise SystemExit("--optimizer muon needs --model (2-D params go to Muon, the rest to AdamW)")
        copies = (counts["muon"] * 1 + counts["muon_adamw"] * 2) / max(nt, 1)
        notes.append(f"Muon on {counts['muon']:,} params (1 buffer), AdamW on "
                     f"{counts['muon_adamw']:,} (2 buffers: 1-D, embeddings, --adamw-regex matches)")
    else:  # adafactor
        if model is None:
            raise SystemExit("--optimizer adafactor needs --model (factored state depends on shapes)")
        copies = adafactor_state_elems(model) / max(nt, 1)
        notes.append("Adafactor keeps row and column factors per matrix instead of a full second moment")
    per = bytes_per_param(args.precision, copies)
    shard = args.world if args.strategy == "fsdp" else 1
    count_for = {"weights": n, "grads": nt, "master": nt, "optimizer": nt}
    terms = {k: count_for[k] * v / shard for k, v in per.items()}
    total = sum(terms.values())
    out = {"params": n, "trainable": nt, "precision": args.precision, "optimizer": args.optimizer,
           "strategy": args.strategy, "world": args.world,
           "bytes_per_param": {k: round(v, 4) for k, v in per.items()},
           "per_rank_gib": {k: v / GIB for k, v in terms.items()},
           "total_gib": total / GIB, "gpu_gib": args.gpu_gib,
           "fits": total / GIB < args.gpu_gib, "notes": notes}
    if args.measure:
        if model is None:
            raise SystemExit("--measure needs --model")
        measured = measure_optimizer_state(model, args.optimizer, args.adamw_regex)
        predicted = nt * bytes_per_param("fp32", copies)["optimizer"]
        out["measured_optimizer_bytes_fp32"] = measured
        out["predicted_optimizer_bytes_fp32"] = predicted
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", type=float)
    src.add_argument("--model")
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adam", "sgd", "sgd-momentum", "muon", "adafactor"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "bf16-master"])
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--strategy", default=None, choices=["ddp", "fsdp"])
    ap.add_argument("--gpu-gib", type=float, default=80.0)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--adamw-regex", default=None,
                    help="with --optimizer muon: parameter names matching this go to AdamW")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.strategy is None:
        args.strategy = "fsdp" if args.world > 1 else "ddp"

    r = budget(args)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    print(f"params {r['params']:,} | {r['optimizer']} | {r['precision']} | "
          f"{r['strategy']} x {r['world']}")
    print(f"{'term':<10} {'bytes/param':>12} {'GiB per rank':>13}")
    for k in ("weights", "grads", "master", "optimizer"):
        print(f"{k:<10} {r['bytes_per_param'][k]:>12g} {r['per_rank_gib'][k]:>13.2f}")
    print(f"{'total':<10} {'':>12} {r['total_gib']:>13.2f}" if r["trainable"] < r["params"] else
          f"{'total':<10} {sum(r['bytes_per_param'].values()):>12g} {r['total_gib']:>13.2f}")
    verdict = "fits" if r["fits"] else "does NOT fit"
    print(f"{verdict} in {r['gpu_gib']:g} GiB before activations")
    for note in r["notes"]:
        print(f"note: {note}")
    if "measured_optimizer_bytes_fp32" in r:
        m, p = r["measured_optimizer_bytes_fp32"], r["predicted_optimizer_bytes_fp32"]
        print(f"optimizer state, one real CPU step in fp32: measured {m:,} bytes, "
              f"formula {p:,.0f} bytes ({'match' if abs(m - p) <= 0.001 * max(p, 1) else 'DIFFERENT'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
