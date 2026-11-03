#!/usr/bin/env python3
"""Reliability arithmetic for agent evals: pass@k, pass^k, paired comparison, run budgets.

Run with no arguments for a worked example, or import the functions.

Definitions
  n  trials of one task, c of them passed.
  pass@k  probability that at least one of k fresh attempts passes.
          Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al. 2021, arXiv 2107.03374).
  pass^k  probability that all k fresh attempts pass.
          Estimator C(c, k) / C(n, k) (tau-bench, Yao et al. 2024, arXiv 2406.12045).
  Both need n >= k. Inspect AI ships the same two estimators as the
  `pass_at(k)` and `pass_k(k)` epoch reducers.

Usage
  python reliability.py                         worked example
  python reliability.py passk --n 8 --c 7 --k 4
  python reliability.py compare old.json new.json   per-task pass counts, see compare()
  python reliability.py budget --tasks 40 --trials 5 --versions 2 --cost 0.30
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """At least one of k attempts passes. Unbiased for k <= n."""
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """All k attempts pass (pass^k)."""
    _check(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n):
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}; run more trials")


def suite_metric(results: Mapping[str, tuple[int, int]], k: int, metric: str = "pass^k") -> float:
    """Mean of a per-task metric. results maps task_id -> (n, c)."""
    fn = pass_hat_k if metric == "pass^k" else pass_at_k
    vals = [fn(n, c, k) for n, c in results.values()]
    return sum(vals) / len(vals)


@dataclass
class Comparison:
    delta: float          # new minus old, mean per-task pass rate
    ci_low: float         # 95% paired bootstrap interval on delta
    ci_high: float
    n_tasks: int

    @property
    def distinguishable(self) -> bool:
        """True when the interval excludes zero. Not a substitute for reading transcripts."""
        return self.ci_low > 0 or self.ci_high < 0


def compare(old: Mapping[str, tuple[int, int]], new: Mapping[str, tuple[int, int]],
            iters: int = 10_000, seed: int = 0) -> Comparison:
    """Paired bootstrap over tasks on the per-task pass rate c/n.

    Resampling tasks (not trials) keeps the pairing: each resampled task
    contributes its old and new rate together, so task difficulty cancels.
    """
    shared = sorted(set(old) & set(new))
    if len(shared) < 2:
        raise ValueError("need at least two tasks present in both runs")
    diffs = [new[t][1] / new[t][0] - old[t][1] / old[t][0] for t in shared]
    rng = random.Random(seed)
    m = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(iters))
    return Comparison(
        delta=sum(diffs) / m,
        ci_low=boots[int(0.025 * iters)],
        ci_high=boots[int(0.975 * iters) - 1],
        n_tasks=m,
    )


def run_budget(tasks: int, trials: int, versions: int, cost_per_run: float | None = None) -> dict:
    """Agent runs (and optional dollars) an eval plan costs: tasks x trials x versions."""
    runs = tasks * trials * versions
    out = {"runs": runs}
    if cost_per_run is not None:
        out["cost"] = round(runs * cost_per_run, 2)
    return out


def per_trial_rate_needed(target_pass_hat_k: float, k: int) -> float:
    """Per-attempt success p such that p**k equals the target (independent trials).

    Shows why consistency is expensive: pass^8 >= 0.9 needs p of about 0.987.
    """
    return target_pass_hat_k ** (1.0 / k)


def _load(path: str) -> dict[str, tuple[int, int]]:
    """JSON {task_id: [n, c]} or {task_id: [true, false, ...]}."""
    raw = json.load(open(path))
    out = {}
    for t, v in raw.items():
        if v and isinstance(v[0], bool):
            out[t] = (len(v), sum(v))
        else:
            out[t] = (int(v[0]), int(v[1]))
    return out


def _example() -> None:
    print("One task, 8 trials, 7 passed:")
    for k in (1, 2, 4, 8):
        print(f"  k={k}: pass@k={pass_at_k(8, 7, k):.3f}  pass^k={pass_hat_k(8, 7, k):.3f}")
    print("\nPer-attempt success needed for pass^k = 0.90:")
    for k in (1, 3, 8):
        print(f"  k={k}: p >= {per_trial_rate_needed(0.9, k):.3f}")
    print("\nBudget, 40 tasks x 5 trials x 2 versions at $0.30/run:", run_budget(40, 5, 2, 0.30))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("passk"); a.add_argument("--n", type=int, required=True)
    a.add_argument("--c", type=int, required=True); a.add_argument("--k", type=int, required=True)
    b = sub.add_parser("compare"); b.add_argument("old"); b.add_argument("new")
    d = sub.add_parser("budget"); d.add_argument("--tasks", type=int, required=True)
    d.add_argument("--trials", type=int, required=True); d.add_argument("--versions", type=int, default=2)
    d.add_argument("--cost", type=float)
    args = p.parse_args(argv)
    if args.cmd == "passk":
        print(json.dumps({"pass@k": pass_at_k(args.n, args.c, args.k),
                          "pass^k": pass_hat_k(args.n, args.c, args.k)}, indent=2))
    elif args.cmd == "compare":
        r = compare(_load(args.old), _load(args.new))
        print(json.dumps({**r.__dict__, "distinguishable": r.distinguishable}, indent=2))
    elif args.cmd == "budget":
        print(json.dumps(run_budget(args.tasks, args.trials, args.versions, args.cost), indent=2))
    else:
        _example()


if __name__ == "__main__":
    main()
