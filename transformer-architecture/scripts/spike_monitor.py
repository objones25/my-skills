#!/usr/bin/env python3
"""
Read a training log and report the stability signals worth acting on, before the loss curve
makes them obvious.

    python spike_monitor.py log.csv                 # columns: step, loss, grad_norm, ...
    python spike_monitor.py log.jsonl --json
    python spike_monitor.py --demo                  # writes and analyses a synthetic log

What it computes, per column that is present:

  spike score       percentage of values at least 7 standard deviations from the rolling
                    average of the previous 1,000 values (OLMo 2's definition [2501.00656 §3.2]).
                    Std is the rolling std of the same window. Computed on loss and grad_norm.
                    Early in training the window still contains the loss decay, which inflates
                    the std; a 7-sigma rule then misses real spikes. Use --window 200 (or detrend)
                    for the first few thousand steps.
  spikes            the steps that crossed, so you can pull those batches (OLMo 2 traced many
                    spikes to long repeated n-grams in the data)
  slow growth       least-squares slope of log(grad_norm) against step over the last half of the
                    run, reported as growth per 10k steps. OLMo 2's n-gram filter removed spikes
                    but "had no effect on the slow growth in gradient norm", which is an init and
                    architecture signal, not a data one
  attention logits  if max_attn_logit is logged: steps above 1e4. Wortsman et al. [2309.14322]
                    found every run whose attention logits passed 1e4 diverged
  output norm       if output_norm (final-layer output norm) is logged: its growth over the run.
                    Chameleon [2405.09818] found this norm's growth correlated with later divergence

The script reads CSV (header row) or JSON lines. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

WINDOW, SIGMAS = 1000, 7.0
ATTN_LOGIT_LIMIT = 1e4


def load(path: Path) -> dict[str, list[float]]:
    text = path.read_text()
    rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix in (".jsonl", ".ndjson") else list(csv.DictReader(text.splitlines())))
    cols: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            if v not in ("", None):
                cols.setdefault(k, []).append(float(v))
    if "step" not in cols:
        n = max(len(v) for v in cols.values())
        cols["step"] = [float(i) for i in range(n)]
    return cols


def spikes(values: list[float], window: int = WINDOW, sigmas: float = SIGMAS) -> list[int]:
    """Indices i where |v_i - mean(prev window)| >= sigmas * std(prev window).
    Needs at least 30 previous values before it starts judging."""
    out, prev = [], []
    s = s2 = 0.0
    for i, v in enumerate(values):
        n = len(prev)
        if n >= 30:
            mean = s / n
            var = max(s2 / n - mean * mean, 0.0)
            std = math.sqrt(var)
            if std > 0 and abs(v - mean) >= sigmas * std:
                out.append(i)
        prev.append(v)
        s += v
        s2 += v * v
        if len(prev) > window:
            old = prev.pop(0)
            s -= old
            s2 -= old * old
    return out


def log_slope_per_10k(steps: list[float], values: list[float]) -> float:
    """Growth factor per 10k steps of a positive series, fit on the last half."""
    half = len(values) // 2
    xs, ys = steps[half:], [math.log(max(v, 1e-12)) for v in values[half:]]
    if len(xs) < 3:
        return float("nan")
    slope = statistics.linear_regression(xs, ys).slope
    return math.exp(slope * 1e4)


def analyse(cols: dict[str, list[float]], window: int = WINDOW, sigmas: float = SIGMAS) -> dict:
    steps = cols["step"]
    report: dict = {"steps": len(steps), "window": window, "sigmas": sigmas, "series": {}, "findings": []}
    for name in ("loss", "grad_norm"):
        if name not in cols:
            continue
        idx = spikes(cols[name], window, sigmas)
        score = 100.0 * len(idx) / len(cols[name])
        report["series"][name] = {"spike_score_pct": round(score, 4),
                                  "spike_steps": [int(steps[i]) for i in idx][:50]}
        if idx:
            report["findings"].append(
                f"{name}: {len(idx)} spikes (spike score {score:.3f}%), first at step {int(steps[idx[0]])}. "
                "Pull those batches and check for long repeated n-grams before touching the model")
    if "grad_norm" in cols:
        g = log_slope_per_10k(steps, cols["grad_norm"])
        report["series"]["grad_norm"]["growth_per_10k_steps"] = round(g, 4)
        if g > 1.1:
            report["findings"].append(
                f"grad_norm grows {g:.2f}x per 10k steps over the second half. Slow growth is an "
                "init / norm-placement signal (OLMo 2: data filtering did not fix it); see stability.md")
    if "max_attn_logit" in cols:
        over = [int(steps[i]) for i, v in enumerate(cols["max_attn_logit"]) if v > ATTN_LOGIT_LIMIT]
        report["series"]["max_attn_logit"] = {"max": max(cols["max_attn_logit"]), "steps_over_1e4": over[:50]}
        if over:
            report["findings"].append(
                f"max attention logit passed 1e4 at step {over[0]}: in Wortsman et al. every such run "
                "diverged. Add QK-norm (or softcapping / QK-clip) or lower the LR")
    if "output_norm" in cols:
        o = cols["output_norm"]
        q = max(1, len(o) // 10)
        ratio = statistics.mean(o[-q:]) / statistics.mean(o[:q])
        report["series"]["output_norm"] = {"last_tenth_over_first_tenth": round(ratio, 3)}
        if ratio > 2:
            report["findings"].append(
                f"final-layer output norm grew {ratio:.1f}x from the first to the last tenth of the log; "
                "Chameleon saw this growth precede divergence. Consider z-loss and norm reordering")
    if not report["findings"]:
        report["findings"].append("no spikes or growth signals found")
    return report


def demo_log(n: int = 6000, seed: int = 0) -> list[dict]:
    """Synthetic run: smooth loss decay, two loss spikes, one burst of grad-norm spikes,
    slow grad-norm growth, and attention logits that drift past 1e4 near the end."""
    rng = random.Random(seed)
    rows = []
    for t in range(n):
        loss = 2.2 + 8.6 * math.exp(-t / 400) + rng.gauss(0, 0.01)
        grad = 0.25 * math.exp(t / 20000) * (1 + rng.gauss(0, 0.03))
        attn = 120 * math.exp(t / 1000)
        outn = 40 + 0.004 * t
        if t in (2100, 4300):
            loss += 0.9
        if 3500 <= t < 3504:
            grad *= 6
        rows.append({"step": t, "loss": round(loss, 5), "grad_norm": round(grad, 5),
                     "max_attn_logit": round(attn, 2), "output_norm": round(outn, 3)})
    return rows


def render(r: dict) -> str:
    lines = [f"# Training log: {r['steps']} rows, spike rule = {r['sigmas']:g} sigma vs previous {r['window']}", ""]
    lines += ["| series | value |", "|---|---|"]
    for name, s in r["series"].items():
        for k, v in s.items():
            if isinstance(v, list) and len(v) > 6:
                v = f"{v[:5]} ... ({len(v)}{'+' if len(v) == 50 else ''} total)"
            lines.append(f"| {name}.{k} | {v} |")
    lines += ["", "## Findings", ""] + [f"- {f}" for f in r["findings"]]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--sigmas", type=float, default=SIGMAS)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", metavar="OUT_CSV", nargs="?", const="demo_training_log.csv",
                    help="write a synthetic log to OUT_CSV and analyse it")
    a = ap.parse_args()
    if a.demo:
        rows = demo_log()
        with open(a.demo, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        a.log = Path(a.demo)
    if not a.log:
        ap.error("give a log file or --demo")
    r = analyse(load(a.log), a.window, a.sigmas)
    print(json.dumps(r, indent=2) if a.json else render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
