"""Analyze a num_envs sweep the way evaluation.md recommends, on the G1 project's real results.

Input: assets/g1_sweep_results.csv (a Unitree G1 velocity-tracking sweep on mjlab 1.6.0 + rsl_rl 5.4.2,
Oct 29 2026, one A100). Columns: mark_minutes, num_envs, seed, run_id, iteration,
train_minutes, vel_error_xy, vel_error_yaw, no_fall_fraction, mean_time_to_fall_s,
mean_tracking_reward. One row per run and evaluation mark.

    python examples/analyze_sweep.py                      # table at 6/12/24/45 min + comparisons
    python examples/analyze_sweep.py --metric no_fall_fraction --mark 12
    python examples/analyze_sweep.py --csv path/to/other_results.csv

What it prints
    1. Seed-mean of the metric per (num_envs, mark), with min-max over seeds and iterations reached.
    2. The "standing still" reference: mean commanded speed over the 6 evaluation commands.
    3. P(row beats column) at the chosen mark, lower metric is better (run_stats.prob_improvement).
       With 3 seeds per arm this is coarse: each value is a fraction of 9 pairs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stats  # noqa: E402

EVAL_COMMANDS = [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.0, -0.5)]
LOWER_IS_BETTER = {"vel_error_xy", "vel_error_yaw"}


def load(path: Path) -> dict[tuple[int, float], list[dict]]:
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "-long-" in row["run_id"]:
                continue  # the 90-min long run is a separate experiment
            groups[(int(row["num_envs"]), float(row["mark_minutes"]))].append(row)
    return groups


def standing_still_error() -> float:
    """vel_error_xy of a robot that never moves = mean over commands of |command xy|."""
    return float(np.mean([math.hypot(vx, vy) for vx, vy, _ in EVAL_COMMANDS]))


def summary(groups, metric: str, marks: list[float]) -> list[dict]:
    rows = []
    for n in sorted({k[0] for k in groups}):
        row: dict = {"num_envs": n}
        for m in marks:
            rs = groups.get((n, m), [])
            if not rs:
                row[f"{m:g} min"] = "-"
                continue
            x = np.array([float(r[metric]) for r in rs])
            row[f"{m:g} min"] = f"{x.mean():.3f} [{x.min():.3f},{x.max():.3f}]"
        last = groups.get((n, marks[-1]), [])
        row["iters@last"] = int(np.mean([int(r["iteration"]) for r in last])) if last else "-"
        rows.append(row)
    return rows


def pairwise(groups, metric: str, mark: float) -> tuple[list[int], np.ndarray]:
    ns = sorted({k[0] for k in groups if k[1] == mark})
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    M = np.full((len(ns), len(ns)), np.nan)
    for i, a in enumerate(ns):
        for j, b in enumerate(ns):
            if i != j:
                xa = {"g1": sign * np.array([float(r[metric]) for r in groups[(a, mark)]])}
                xb = {"g1": sign * np.array([float(r[metric]) for r in groups[(b, mark)]])}
                M[i, j] = run_stats.prob_improvement(xa, xb)
    return ns, M


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=ROOT / "assets" / "g1_sweep_results.csv")
    p.add_argument("--metric", default="vel_error_xy")
    p.add_argument("--marks", type=float, nargs="+", default=[6, 12, 24, 45])
    p.add_argument("--mark", type=float, default=45, help="mark for the pairwise comparison")
    a = p.parse_args(argv)
    groups = load(a.csv)
    rows = summary(groups, a.metric, a.marks)
    keys = list(rows[0])
    print(f"{a.metric}: seed mean [min, max] by training-clock minute")
    print(" | ".join(f"{k:>22}" if "min" in k else f"{k:>9}" for k in keys))
    for r in rows:
        print(" | ".join(f"{str(r[k]):>22}" if "min" in k else f"{str(r[k]):>9}" for k in keys))
    ref = standing_still_error()
    print(f"\nstanding-still vel_error_xy = mean |command xy| over the 6 eval commands = {ref:.3f} m/s")
    ns, M = pairwise(groups, a.metric, a.mark)
    print(f"\nP(row better than column) on {a.metric} at {a.mark:g} min (3 seeds each, so steps of 1/9):")
    print("        " + "".join(f"{n:>8}" for n in ns))
    for i, n in enumerate(ns):
        print(f"{n:>8}" + "".join("       -" if i == j else f"{M[i, j]:8.2f}" for j in range(len(ns))))
    return {"rows": rows, "standing_still": ref, "pairwise": (ns, M)}


if __name__ == "__main__":
    main()
