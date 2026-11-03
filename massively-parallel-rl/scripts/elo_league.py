#!/usr/bin/env python3
"""Single-agent Elo league, the performance metric of Zheng et al. (2026), Sec. 3.3.

Idea: every agent plays the SAME fixed set of evaluation environment instances
(same seeds / levels). For a pair of agents, each instance is one "game": the
agent with the higher episode return wins, a tie counts 0.5 to each. Win rates
over all instances form a matrix W (W[i, j] = win rate of i against j). A
Bradley-Terry model, P(i beats j) = 1 / (1 + 10 ** ((R_j - R_i) / 400)), is fit
to W by maximum likelihood, giving one Elo rating R_i per agent. Elo is only
defined up to a constant, so ratings are shifted so a reference agent (the
uniform random policy in the paper) sits at 0.

Input: a CSV in wide format, one row per agent:
    agent,ep0,ep1,...,epE-1
    random,0.0,1.0,...
    ppo_1e15_w8,12.0,9.5,...
Columns ep* must be the SAME evaluation instances for every agent.

Usage:
    python elo_league.py returns.csv --reference random
    python elo_league.py returns.csv --reference random --bootstrap 200 --json out.json
    python elo_league.py --winrate-from-diff 40.8

The last form converts an Elo gap to the win-rate deviation from 50% that the
paper reports as "win rate error" (Tables 2 and 3).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys

import numpy as np
from scipy.optimize import minimize

ELO_SCALE = 400.0 / math.log(10.0)  # converts natural-log strength to Elo points


def win_rate_matrix(returns: np.ndarray) -> np.ndarray:
    """returns: (A agents, E instances) -> W: (A, A), W[i, j] = wins of i over j / E."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"returns must be 2-D (agents, instances), got shape {r.shape}")
    if np.isnan(r).any():
        raise ValueError("returns contain NaN; every agent needs a return on every instance")
    greater = (r[:, None, :] > r[None, :, :]).mean(axis=-1)
    equal = (r[:, None, :] == r[None, :, :]).mean(axis=-1)
    w = greater + 0.5 * equal
    np.fill_diagonal(w, 0.5)
    return w


def fit_bradley_terry(w: np.ndarray, games_per_pair: int, l2: float = 1e-4) -> np.ndarray:
    """Maximum-likelihood Elo ratings (mean zero) from a win-rate matrix.

    A tiny L2 penalty keeps ratings finite when an agent wins or loses every game,
    which would otherwise send its MLE rating to +/- infinity.
    """
    a = w.shape[0]
    wins = w * games_per_pair
    off = ~np.eye(a, dtype=bool)

    def nll(s: np.ndarray) -> tuple[float, np.ndarray]:
        diff = s[:, None] - s[None, :]
        log_p = -np.logaddexp(0.0, -diff)  # log sigmoid(diff), stable
        ll = (wins * log_p)[off].sum()
        p = np.exp(log_p)
        g_mat = wins * (1.0 - p)  # d ll / d diff
        grad = (g_mat * off).sum(axis=1) - (g_mat * off).sum(axis=0)
        return -ll + l2 * (s @ s), -grad + 2.0 * l2 * s

    res = minimize(nll, np.zeros(a), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {res.message}")
    s = res.x - res.x.mean()
    return s * ELO_SCALE


def elo_league(returns: np.ndarray, reference: int | None = None) -> np.ndarray:
    w = win_rate_matrix(returns)
    elo = fit_bradley_terry(w, games_per_pair=returns.shape[1])
    if reference is not None:
        elo = elo - elo[reference]
    return elo


def bootstrap_elo(returns: np.ndarray, reference: int | None, n: int, seed: int = 0) -> np.ndarray:
    """Resample evaluation instances with replacement; returns (n, A) ratings."""
    rng = np.random.default_rng(seed)
    e = returns.shape[1]
    out = np.empty((n, returns.shape[0]))
    for b in range(n):
        idx = rng.integers(0, e, size=e)
        out[b] = elo_league(returns[:, idx], reference)
    return out


def winrate_deviation(elo_diff: float) -> float:
    """Win rate of an agent rated elo_diff above its opponent, minus 0.5."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)) - 0.5


def undefeated_or_winless(w: np.ndarray) -> list[int]:
    a = w.shape[0]
    off = ~np.eye(a, dtype=bool)
    flagged = []
    for i in range(a):
        row = w[i][off[i]]
        if np.all(row == 1.0) or np.all(row == 0.0):
            flagged.append(i)
    return flagged


def distinct_instances(returns: np.ndarray) -> int:
    """Number of evaluation instances that differ in at least one agent's return.

    Identical columns are the same game played again (deterministic env and policy);
    they add no information, only false confidence.
    """
    return int(np.unique(np.asarray(returns), axis=1).shape[1])


def read_wide_csv(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if len(header) < 2:
        raise ValueError("CSV needs an agent column plus at least one return column")
    names = [r[0] for r in body]
    data = np.array([[float(x) for x in r[1:]] for r in body])
    return names, data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", help="wide CSV: agent,ep0,ep1,...")
    p.add_argument("--reference", help="agent name pinned to Elo 0 (e.g. random)")
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples of eval instances for 90%% CIs")
    p.add_argument("--json", help="write results to this JSON file")
    p.add_argument("--winrate-from-diff", type=float, help="print win-rate deviation for an Elo gap and exit")
    args = p.parse_args()

    if args.winrate_from_diff is not None:
        print(f"{100 * winrate_deviation(args.winrate_from_diff):+.2f}%")
        return 0
    if not args.csv:
        p.error("csv is required unless --winrate-from-diff is given")

    names, returns = read_wide_csv(args.csv)
    ref = None
    if args.reference is not None:
        if args.reference not in names:
            p.error(f"reference agent {args.reference!r} not in CSV")
        ref = names.index(args.reference)

    k = distinct_instances(returns)
    if k < min(returns.shape[1], 32):
        print(f"warning: only {k} of {returns.shape[1]} evaluation instances differ; the league is "
              f"effectively {k} games. Make instances stochastic (e.g. sticky actions) before trusting "
              f"Elo differences or bootstrap intervals.", file=sys.stderr)
    w = win_rate_matrix(returns)
    for i in undefeated_or_winless(w):
        print(f"warning: {names[i]} wins or loses every game; its Elo is set by the L2 prior, not the data",
              file=sys.stderr)
    elo = elo_league(returns, ref)

    lo = hi = None
    if args.bootstrap:
        boots = bootstrap_elo(returns, ref, args.bootstrap)
        lo, hi = np.percentile(boots, [5, 95], axis=0)

    order = np.argsort(-elo)
    width = max(len(n) for n in names)
    print(f"{'agent':<{width}}  {'elo':>8}" + ("  90% CI" if lo is not None else ""))
    for i in order:
        line = f"{names[i]:<{width}}  {elo[i]:8.1f}"
        if lo is not None:
            line += f"  [{lo[i]:.1f}, {hi[i]:.1f}]"
        print(line)

    if args.json:
        out = {"agents": names, "elo": elo.tolist(), "reference": args.reference,
               "instances": int(returns.shape[1])}
        if lo is not None:
            out["ci90"] = [lo.tolist(), hi.tolist()]
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
