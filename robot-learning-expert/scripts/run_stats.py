"""Aggregate RL results over seeds the way Agarwal et al. (NeurIPS 2021, "Deep RL at the
Edge of the Statistical Precipice") recommend: IQM with stratified-bootstrap percentile
confidence intervals, plus probability of improvement. numpy only.

Input: CSV with header containing at least  method,seed,score  and optionally  task.
    python run_stats.py results.csv                 # table per method
    python run_stats.py results.csv --compare A B   # P(A > B) with CI
    python run_stats.py results.csv --reps 5000 --alpha 0.05

Definitions
    IQM        mean of the middle 50% of all pooled runs (25% trimmed on each side).
    stratified bootstrap  resample runs with replacement separately inside each task,
               recompute the statistic, repeat; take the alpha/2, 1-alpha/2 percentiles.
    P(X > Y)   per task, fraction of (x run, y run) pairs with x > y (ties count 1/2),
               averaged over tasks. Says nothing about the size of the gap.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

Array = np.ndarray


def iqm(x: Array) -> float:
    x = np.sort(np.asarray(x, float).ravel())
    n = len(x)
    if n == 0:
        raise ValueError("empty sample")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def stratified_bootstrap_ci(
    by_task: dict[str, Array], stat=iqm, reps: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of stat(pooled runs), resampling runs within each task."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([np.asarray(v, float) for v in by_task.values()])
    point = stat(pooled)
    boots = np.empty(reps)
    arrays = [np.asarray(v, float) for v in by_task.values()]
    for b in range(reps):
        boots[b] = stat(np.concatenate([v[rng.integers(len(v), size=len(v))] for v in arrays]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def prob_improvement(x_by_task: dict[str, Array], y_by_task: dict[str, Array]) -> float:
    tasks = sorted(set(x_by_task) & set(y_by_task))
    if not tasks:
        raise ValueError("no shared tasks")
    vals = []
    for t in tasks:
        x = np.asarray(x_by_task[t], float)[:, None]
        y = np.asarray(y_by_task[t], float)[None, :]
        vals.append(((x > y) + 0.5 * (x == y)).mean())
    return float(np.mean(vals))


def prob_improvement_ci(x_by_task, y_by_task, reps: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    point = prob_improvement(x_by_task, y_by_task)
    tasks = sorted(set(x_by_task) & set(y_by_task))
    boots = np.empty(reps)
    for b in range(reps):
        xs = {t: (v := np.asarray(x_by_task[t]))[rng.integers(len(v), size=len(v))] for t in tasks}
        ys = {t: (v := np.asarray(y_by_task[t]))[rng.integers(len(v), size=len(v))] for t in tasks}
        boots[b] = prob_improvement(xs, ys)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def curve_band(curves: Array, reps: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[Array, Array, Array]:
    """Learning curves (n_seeds, n_steps) -> (mean, lo, hi) per step via bootstrap over seeds."""
    curves = np.asarray(curves, float)
    rng = np.random.default_rng(seed)
    n = curves.shape[0]
    idx = rng.integers(n, size=(reps, n))
    means = curves[idx].mean(axis=1)  # (reps, n_steps)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2], axis=0)
    return curves.mean(axis=0), lo, hi


def load_csv(path: str) -> dict[str, dict[str, list[float]]]:
    """-> {method: {task: [scores]}}; task defaults to 'all'."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"method", "seed", "score"}
        if not need <= set(reader.fieldnames or []):
            raise ValueError(f"CSV needs columns {sorted(need)} (+ optional task); got {reader.fieldnames}")
        for row in reader:
            out[row["method"]][row.get("task") or "all"].append(float(row["score"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--compare", nargs=2, metavar=("X", "Y"))
    args = ap.parse_args()
    data = load_csv(args.csv)
    pct = int(round(100 * (1 - args.alpha)))
    print(f"{'method':<20}{'runs':>6}{'mean':>10}{'IQM':>10}   {pct}% CI (IQM)")
    for m, by_task in sorted(data.items()):
        pooled = np.concatenate([np.asarray(v) for v in by_task.values()])
        p, lo, hi = stratified_bootstrap_ci({k: np.asarray(v) for k, v in by_task.items()},
                                            reps=args.reps, alpha=args.alpha)
        print(f"{m:<20}{len(pooled):>6}{pooled.mean():>10.4g}{p:>10.4g}   [{lo:.4g}, {hi:.4g}]")
        if len(pooled) < 5:
            print(f"{'':<20}warning: {len(pooled)} runs; intervals will be wide and unreliable")
    if args.compare:
        x, y = args.compare
        p, lo, hi = prob_improvement_ci(data[x], data[y], reps=args.reps, alpha=args.alpha)
        print(f"P({x} > {y}) = {p:.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
