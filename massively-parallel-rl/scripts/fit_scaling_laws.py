#!/usr/bin/env python3
"""Fit and apply RL compute scaling laws, following Zheng et al. (2026), Sec. 3.

Three subcommands, one per stage of the paper's method:

  hparams   Stage 1. From a grid search (C, model, batch size B, learning rate lr,
            seed, score), bootstrap over seeds, pick the best (B, lr) in each
            resample (a "competitive configuration"), then fit
                B*(C)  = aB * C**dB        lr*(C) = a_lr * C**d_lr
            by least squares in log-log space over all competitive configs.

  isoflops  Stage 2. From runs at fixed budgets C with varying model size M
            (FLOPs per frame, FPF), fit a parabola score ~ log10 M per budget by
            weighted least squares (weights = 1 / variance across seeds), take its
            vertex as (M*(C), score*(C)), then fit
                M*(C)  = aM * C**dM        score*(C) = d_s * log10 C + a_s
            The log-linear score law is meant for Elo, which is already a log scale.

  predict   Apply fitted coefficients at a new budget C: B rounded to the nearest
            power of 2 with lr rescaled by sqrt(B_rounded / B) (the square-root rule
            the paper cites from Malladi et al. 2022), M*, frames D = C / M*, score*.

Scores are aggregated across seeds with the interquartile mean (IQM, mean of the
middle 50%), as in the paper. Budgets listed in --holdout are excluded from the fit
and reported as out-of-sample checks.

CSV columns
  hparams : flops, model, batch_size, lr, seed, score
  isoflops: flops, fpf, seed, score

Usage
  python fit_scaling_laws.py hparams grid.csv --boot 1000 --holdout 1e17 --json hp.json
  python fit_scaling_laws.py isoflops iso.csv --holdout 1e18 --json iso.json
  python fit_scaling_laws.py predict --hp hp.json --iso iso.json --flops 1e19
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def iqm(x) -> float:
    return float(trim_mean(np.asarray(x, dtype=float), 0.25))


def fit_power_law(c: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y = a * c**d, least squares on log10. Returns (a, d)."""
    if np.any(np.asarray(y) <= 0) or np.any(np.asarray(c) <= 0):
        raise ValueError("power-law fit needs positive values")
    d, log_a = np.polyfit(np.log10(c), np.log10(y), 1)
    return float(10 ** log_a), float(d)


def fit_log_linear(c: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """s = d * log10 c + a. Returns (a, d)."""
    d, a = np.polyfit(np.log10(c), np.asarray(s, dtype=float), 1)
    return float(a), float(d)


def parse_budgets(values: list[str] | None) -> set[float]:
    return {float(v) for v in values} if values else set()


# ---------------------------------------------------------------- stage 1 ----

def competitive_configs(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """One competitive (batch_size, lr) per bootstrap resample per budget."""
    rng = np.random.default_rng(seed)
    rows = []
    for c, g in df.groupby("flops"):
        groups = [(k, v["score"].to_numpy()) for k, v in g.groupby(["model", "batch_size", "lr"])]
        too_few = [k for k, v in groups if len(v) < 2]
        if too_few:
            print(f"warning: {len(too_few)} configs at C={c:g} have <2 seeds; bootstrap cannot reduce their noise",
                  file=sys.stderr)
        for _ in range(n_boot):
            best_key, best_val = None, -math.inf
            for key, scores in groups:
                sample = scores[rng.integers(0, len(scores), size=len(scores))]
                val = iqm(sample)
                if val > best_val:
                    best_key, best_val = key, val
            rows.append({"flops": c, "model": best_key[0], "batch_size": best_key[1], "lr": best_key[2]})
    return pd.DataFrame(rows)


def run_hparams(args) -> dict:
    df = pd.read_csv(args.csv)
    need = {"flops", "model", "batch_size", "lr", "seed", "score"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"hparams CSV missing columns: {sorted(missing)}")
    hold = parse_budgets(args.holdout)
    comp = competitive_configs(df, args.boot, args.seed)
    fit = comp[~comp["flops"].isin(hold)]
    if fit["flops"].nunique() < 3:
        raise SystemExit("need at least 3 fitted budgets for a power law (fewer cannot show curvature or noise)")
    a_b, d_b = fit_power_law(fit["flops"].to_numpy(), fit["batch_size"].to_numpy())
    a_lr, d_lr = fit_power_law(fit["flops"].to_numpy(), fit["lr"].to_numpy())

    print(f"B*(C)  = {a_b:.3g} * C^{d_b:.3f}   (x{10 ** d_b:.2f} per 10x compute)")
    print(f"lr*(C) = {a_lr:.3g} * C^{d_lr:.3f}   (x{10 ** d_lr:.2f} per 10x compute)")
    per_c = comp.groupby("flops").agg(batch_geo=("batch_size", lambda x: 10 ** np.log10(x).mean()),
                                      lr_geo=("lr", lambda x: 10 ** np.log10(x).mean()))
    print("\nbudget      mean B*    mean lr*   pred B*    pred lr*   role")
    for c, r in per_c.iterrows():
        role = "holdout" if c in hold else "fit"
        print(f"{c:<10.3g}  {r.batch_geo:9.4g}  {r.lr_geo:9.3g}  {a_b * c ** d_b:9.4g}  {a_lr * c ** d_lr:9.3g}   {role}")

    out = {"aB": a_b, "dB": d_b, "a_lr": a_lr, "d_lr": d_lr, "n_boot": args.boot,
           "fit_budgets": sorted(set(fit["flops"])), "holdout": sorted(hold)}
    for c in sorted(hold):
        g = df[df["flops"] == c]
        agg = g.groupby(["model", "batch_size", "lr"])["score"].agg(iqm)
        bm, bb, bl = agg.idxmax()
        pred_b, pred_lr = a_b * c ** d_b, a_lr * c ** d_lr
        print(f"\nholdout C={c:g}: empirical best model={bm}, B={bb:g}, lr={bl:.3g}, score {agg.max():.4g}; "
              f"predicted B={pred_b:.4g}, lr={pred_lr:.3g}. Check the grid cell nearest the prediction.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------- stage 2 ----

def isoflop_vertex(g: pd.DataFrame) -> dict:
    stats = g.groupby("fpf")["score"].agg(score=iqm, var="var", n="count").reset_index()
    if len(stats) < 3:
        raise ValueError("need >=3 model sizes per budget to fit a parabola")
    x = np.log10(stats["fpf"].to_numpy())
    y = stats["score"].to_numpy()
    var = stats["var"].fillna(0).to_numpy()
    floor = max(np.median(var[var > 0]) * 1e-3, 1e-12) if np.any(var > 0) else 1.0
    w = 1.0 / np.maximum(var, floor)
    a2, a1, a0 = np.polyfit(x, y, 2, w=np.sqrt(w))  # polyfit weights multiply residuals
    if a2 >= 0:
        raise ValueError("parabola opens upward: the grid does not bracket an optimum")
    xv = -a1 / (2 * a2)
    bracketed = bool(x.min() <= xv <= x.max())
    return {"m_star": float(10 ** xv), "score_star": float(a0 - a1 ** 2 / (4 * a2)),
            "curvature": float(a2), "bracketed": bracketed,
            "grid": [float(10 ** x.min()), float(10 ** x.max())]}


def run_isoflops(args) -> dict:
    df = pd.read_csv(args.csv)
    need = {"flops", "fpf", "seed", "score"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"isoflops CSV missing columns: {sorted(missing)}")
    hold = parse_budgets(args.holdout)
    verts = {}
    for c, g in df.groupby("flops"):
        v = isoflop_vertex(g)
        verts[float(c)] = v
        if not v["bracketed"]:
            print(f"warning: optimum at C={c:g} lies outside the tested sizes "
                  f"[{v['grid'][0]:.3g}, {v['grid'][1]:.3g}]; extend the grid", file=sys.stderr)
    fit_c = np.array([c for c in verts if c not in hold])
    if len(fit_c) < 3:
        raise SystemExit("need at least 3 fitted budgets")
    a_m, d_m = fit_power_law(fit_c, np.array([verts[c]["m_star"] for c in fit_c]))
    a_s, d_s = fit_log_linear(fit_c, np.array([verts[c]["score_star"] for c in fit_c]))

    print(f"M*(C)     = {a_m:.3g} * C^{d_m:.3f}   (x{10 ** d_m:.2f} per 10x compute)")
    print(f"score*(C) = {d_s:.2f} * log10 C + {a_s:.1f}")
    print("\nbudget      M* (vertex)  pred M*    score*   pred score*  role")
    for c in sorted(verts):
        v = verts[c]
        role = "holdout" if c in hold else "fit"
        print(f"{c:<10.3g}  {v['m_star']:11.4g}  {a_m * c ** d_m:9.4g}  {v['score_star']:7.1f}"
              f"  {d_s * math.log10(c) + a_s:11.1f}  {role}")
    for c in sorted(hold):
        if c in verts:
            v = verts[c]
            pm, ps = a_m * c ** d_m, d_s * math.log10(c) + a_s
            dev = 1.0 / (1.0 + 10.0 ** (-(ps - v["score_star"]) / 400.0)) - 0.5
            print(f"\nholdout C={c:g}: M* relative error {100 * (pm / v['m_star'] - 1):+.1f}%, "
                  f"score error {ps - v['score_star']:+.1f} (if Elo: win-rate deviation {100 * dev:+.2f}%)")
    out = {"aM": a_m, "dM": d_m, "a_score": a_s, "d_score": d_s,
           "vertices": {str(c): v for c, v in verts.items()}, "holdout": sorted(hold)}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------- predict ----

def predict(hp: dict | None, iso: dict | None, c: float) -> dict:
    out: dict = {"flops": c}
    if hp:
        b = hp["aB"] * c ** hp["dB"]
        lr = hp["a_lr"] * c ** hp["d_lr"]
        b2 = 2 ** round(math.log2(b))
        out.update(batch_raw=b, lr_raw=lr, batch=int(b2), lr=lr * math.sqrt(b2 / b),
                   log2_lr=math.log2(lr * math.sqrt(b2 / b)))
    if iso:
        m = iso["aM"] * c ** iso["dM"]
        out.update(fpf=m, frames=c / m, score=iso["d_score"] * math.log10(c) + iso["a_score"])
    return out


def run_predict(args) -> dict:
    hp = json.load(open(args.hp)) if args.hp else None
    iso = json.load(open(args.iso)) if args.iso else None
    if not hp and not iso:
        raise SystemExit("give --hp and/or --iso")
    res = predict(hp, iso, args.flops)
    for k, v in res.items():
        print(f"{k:>10}: {v:.6g}" if isinstance(v, float) else f"{k:>10}: {v}")
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hparams")
    h.add_argument("csv")
    h.add_argument("--boot", type=int, default=1000)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--holdout", nargs="*")
    h.add_argument("--json")
    i = sub.add_parser("isoflops")
    i.add_argument("csv")
    i.add_argument("--holdout", nargs="*")
    i.add_argument("--json")
    r = sub.add_parser("predict")
    r.add_argument("--hp")
    r.add_argument("--iso")
    r.add_argument("--flops", type=float, required=True)
    args = p.parse_args()
    {"hparams": run_hparams, "isoflops": run_isoflops, "predict": run_predict}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
