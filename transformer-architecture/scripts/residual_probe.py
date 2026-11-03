#!/usr/bin/env python3
"""
Measure what each transformer block actually does to the residual stream.

This is the width-vs-depth diagnostic. A block that writes a small update
nearly parallel to the stream it received is not doing much: cosine
similarity between a block's input and output near 1.0, with a small
relative write norm, means the block is close to a no-op. When the last
third of a stack looks like that, adding more layers is buying less than
widening would.

Three numbers per block:

  cos(x_in, x_out)   1.00 = the block left the stream's direction alone.
                     Healthy mid-stack values sit around 0.90-0.99 and
                     drift upward with depth in every real model; what
                     matters is how *fast* and how *far*.
  ||delta||/||x_in|| how big the block's write is relative to what it got.
                     Falls with depth in every model. Watch for a cliff.
  ||x|| growth       the residual stream norm at each block boundary.
                     Pre-norm stacks grow it monotonically; exponential
                     growth is the pre-norm failure mode (see
                     references/stability.md).

Also reports OLMo 2's growth exponent, lambda = (1/L) * log(||x_L||/||x_1||),
where 0 means the stream neither explodes nor collapses across depth. OLMo 2 used
it to compare initializations; no threshold for trained models is established.

Prior work and caveats. 1 - cos(x_in, x_out) is ShortGPT's Block Influence
(Men et al., 2024), and Gromov et al. (2024) prune by the related angular
distance. Both are pruning results: they show which blocks can be removed and
healed, not that the parameters would do better as width. Men et al. note that
pre-norm alone raises input/output similarity, and Gromov et al. find the final
block is never safe to drop. The first token carries "massive activations" (Sun
et al., 2024) up to ~100,000x the rest, so it is excluded from the averages by
default (--keep-first-token to include it). The 0.995 / 0.05 / 0.35 thresholds
below are heuristics of this skill, not published values.

Usage:
    python residual_probe.py --model HuggingFaceTB/SmolLM2-135M
    python residual_probe.py --model ./my-checkpoint --text-file sample.txt
    python residual_probe.py --model gpt2 --plot probe.png --json probe.json

For a model you wrote yourself, import and call directly:

    from residual_probe import probe_blocks, render
    stats = probe_blocks(model, blocks=model.layers, input_ids=ids)
    print(render(stats))
"""

from __future__ import annotations

import argparse
import json
import math
import sys

try:
    import torch
except ImportError:
    sys.exit("residual_probe.py needs PyTorch: pip install torch")


DEFAULT_TEXT = (
    "The residual stream carries information from the embedding layer to the "
    "unembedding, and every block reads from it and writes back into it. "
    "Attention moves information between positions; the MLP transforms it in "
    "place. A block that writes little and changes direction little is close "
    "to an identity function, which is a useful thing to be able to measure."
)


def _find_blocks(model):
    """Locate the list of decoder blocks in a Hugging Face or custom model."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers",
                 "gpt_neox.layers", "layers", "blocks", "transformer.blocks"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj):
            return list(obj), path
    raise RuntimeError(
        "could not locate the decoder blocks; pass blocks=... explicitly to "
        "probe_blocks()"
    )


def _as_hidden(x):
    """Blocks often return a tuple (hidden_states, attn_weights, ...)."""
    return x[0] if isinstance(x, (tuple, list)) else x


@torch.no_grad()
def probe_blocks(model, input_ids, blocks=None, attention_mask=None,
                 skip_first: int = 1) -> list[dict]:
    """
    Run one forward pass with hooks on every block. Returns one dict per block.

    Statistics are computed per token and then averaged, not computed on a
    mean-pooled vector, averaging the vectors first would wash out exactly
    the per-token variation we care about.
    """
    if blocks is None:
        blocks, _ = _find_blocks(model)

    captured: list[tuple] = [None] * len(blocks)

    def make_hook(i):
        def hook(_mod, args, kwargs, output):
            x_in = _as_hidden(args[0] if args else kwargs.get("hidden_states"))
            x_out = _as_hidden(output)
            captured[i] = (x_in.detach().float(), x_out.detach().float())
        return hook

    handles = [b.register_forward_hook(make_hook(i), with_kwargs=True)
               for i, b in enumerate(blocks)]
    try:
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            # A model you wrote yourself probably takes ids positionally.
            model(input_ids)
    finally:
        for h in handles:
            h.remove()

    stats = []
    for i, cap in enumerate(captured):
        if cap is None:
            continue
        x_in, x_out = cap
        if skip_first and x_in.dim() == 3 and x_in.shape[1] > skip_first:
            x_in, x_out = x_in[:, skip_first:], x_out[:, skip_first:]
        delta = x_out - x_in
        n_in = x_in.norm(dim=-1)
        n_out = x_out.norm(dim=-1)
        n_delta = delta.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(x_in, x_out, dim=-1)
        rel = n_delta / n_in.clamp_min(1e-6)
        stats.append({
            "block": i,
            "cos_in_out": cos.mean().item(),
            "cos_p05": cos.flatten().quantile(0.05).item(),
            "rel_write": rel.mean().item(),
            "norm_in": n_in.mean().item(),
            "norm_out": n_out.mean().item(),
        })
    return stats


def growth_exponent(stats: list[dict]) -> float:
    """OLMo 2's lambda: mean log-growth of the residual norm per layer."""
    if len(stats) < 2:
        return 0.0
    first, last = stats[0]["norm_in"], stats[-1]["norm_out"]
    return math.log(max(last, 1e-9) / max(first, 1e-9)) / len(stats)


def _bar(v: float, lo: float, hi: float, width: int = 24) -> str:
    frac = 0.0 if hi <= lo else min(max((v - lo) / (hi - lo), 0.0), 1.0)
    filled = round(frac * width)
    return "#" * filled + "." * (width - filled)


def render(stats: list[dict]) -> str:
    if not stats:
        return "no blocks captured"
    L = []
    L.append(f"{'blk':>4} {'cos(in,out)':>12} {'':<26} {'|d|/|x|':>9} {'||x||':>10}")
    L.append("-" * 66)
    for s in stats:
        L.append(f"{s['block']:>4} {s['cos_in_out']:>12.4f} "
                 f"{_bar(s['cos_in_out'], 0.5, 1.0):<26} "
                 f"{s['rel_write']:>9.4f} {s['norm_out']:>10.2f}")
    lam = growth_exponent(stats)
    n = len(stats)
    tail = stats[max(0, n - max(2, n // 3)):]
    head = stats[:max(2, n // 3)]
    tail_cos = sum(s["cos_in_out"] for s in tail) / len(tail)
    head_cos = sum(s["cos_in_out"] for s in head) / len(head)
    tail_rel = sum(s["rel_write"] for s in tail) / len(tail)
    head_rel = sum(s["rel_write"] for s in head) / len(head)

    L.append("")
    L.append(f"blocks                {n}")
    L.append(f"first third  cos      {head_cos:.4f}    rel write  {head_rel:.4f}")
    L.append(f"last third   cos      {tail_cos:.4f}    rel write  {tail_rel:.4f}")
    L.append(f"growth exponent lam   {lam:+.4f}   (0 = residual norm flat across depth)")
    L.append("")
    L.append("Reading it:")
    if tail_cos > 0.995 and tail_rel < 0.05:
        L.append("  The last third is close to an identity map: those blocks barely")
        L.append("  write to the stream or turn it. They are pruning candidates in the")
        L.append("  ShortGPT / Gromov et al. sense. Whether the parameters would do")
        L.append("  better as width is untested; check with a matched-parameter run.")
    elif tail_cos > 0.99:
        L.append("  The last third turns the stream very little but is still writing.")
        L.append("  Late layers refining magnitude rather than direction is normal;")
        L.append("  this is suggestive, not conclusive. Compare against a shallower,")
        L.append("  wider config at matched parameters and matched tokens.")
    else:
        L.append("  Every third of the stack is still meaningfully rewriting the")
        L.append("  stream. Depth is being used. If you need capacity, this config")
        L.append("  does not argue for trading layers away.")
    if lam > 0.35:
        L.append("")
        L.append(f"  lambda = {lam:+.3f} is large (heuristic cutoff 0.35): the residual norm grows fast with")
        L.append("  depth, the pre-norm failure mode. See references/stability.md")
        L.append("  (Peri-LN, OLMo 2's reordered norm, QK-norm).")
    return "\n".join(L)


def plot(stats: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return
    idx = [s["block"] for s in stats]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax[0].plot(idx, [s["cos_in_out"] for s in stats], marker="o", ms=3)
    ax[0].set_title("cos(x_in, x_out)"); ax[0].set_xlabel("block")
    ax[1].plot(idx, [s["rel_write"] for s in stats], marker="o", ms=3, color="tab:orange")
    ax[1].set_title("||delta|| / ||x_in||"); ax[1].set_xlabel("block")
    ax[2].plot(idx, [s["norm_out"] for s in stats], marker="o", ms=3, color="tab:green")
    ax[2].set_title("residual norm"); ax[2].set_xlabel("block")
    for a in ax:
        a.grid(alpha=0.3)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}", file=sys.stderr)


def self_test() -> int:
    """
    Probe the untrained reference model from reference_block.py. This checks
    the hook plumbing offline; the numbers themselves are not meaningful,
    because an untrained model has learned nothing to measure.
    """
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    from reference_block import Cfg, Model
    torch.manual_seed(0)
    c = Cfg(n_layers=8)
    m = Model(c).eval()
    ids = torch.randint(0, c.vocab, (2, c.seq_len))
    stats = probe_blocks(m, ids, blocks=list(m.blocks))
    ok = len(stats) == c.n_layers and all(
        0.0 <= s["cos_in_out"] <= 1.0001 and s["rel_write"] >= 0 for s in stats)
    print(render(stats))
    print(f"\n[{'PASS' if ok else 'FAIL'}] captured {len(stats)}/{c.n_layers} blocks "
          "with in-range statistics (untrained model, numbers are not meaningful)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="probe the untrained reference model; no download needed")
    ap.add_argument("--model", help="HF model id or local path")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--plot", metavar="PATH")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--keep-first-token", action="store_true",
                    help="include position 0, where massive activations sit")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.model:
        ap.error("need --model or --self-test")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return int(bool(sys.stderr.write(
            "the --model path needs transformers: pip install transformers\n"
            "(or import probe_blocks directly for a model you built yourself)\n")))

    text = a.text
    if a.text_file:
        text = open(a.text_file).read()
    text = text or DEFAULT_TEXT

    tok = AutoTokenizer.from_pretrained(a.model)
    kwargs = {"attn_implementation": "eager"}
    try:            # transformers renamed torch_dtype -> dtype; the old name warns
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, **kwargs)
    model = model.to(a.device).eval()
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=a.max_len).input_ids.to(a.device)

    stats = probe_blocks(model, ids, skip_first=0 if a.keep_first_token else 1)
    print(render(stats))
    if a.plot:
        plot(stats, a.plot)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"model": a.model, "n_tokens": int(ids.numel()),
                       "growth_exponent": growth_exponent(stats),
                       "blocks": stats}, fh, indent=2)
        print(f"wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
