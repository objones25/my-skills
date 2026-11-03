"""Why undirected exploration fails and resets help, computed exactly (Notes 5 §3.1, Notes 7 §3.2).

On the combination lock (mdp_tools.combination_lock), a uniformly random policy reaches step h
of the chain with probability A**-h, so the expected number of episodes to see the goal once is
A**H. A reset-based explorer that checkpoints the deepest state reached needs about H*(A+1)/2 steps
(local_sim.frontier_search_lock). This script prints both, plus the expected visit counts
N_h(s) after T episodes that Notes 5 uses to show "something is going wrong".

    python examples/exploration_chain.py
    python examples/exploration_chain.py --A 4 --H 2 4 6 8 10 --episodes 10000

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_sim  # noqa: E402
import mdp_tools  # noqa: E402


def reach_probabilities(P: np.ndarray, H: int) -> np.ndarray:
    """State distribution at each step h = 0..H under the uniform policy, starting at state 0."""
    S, A, _ = P.shape
    pi = np.full((S, A), 1.0 / A)
    d = np.zeros((H + 1, S))
    d[0, 0] = 1.0
    for h in range(H):
        d[h + 1] = np.einsum("s,sa,sat->t", d[h], pi, P)
    return d


def reset_explorer_steps(H: int, A: int, seed: int) -> int:
    P, r, _ = mdp_tools.combination_lock(H, A, seed=seed)
    mu = np.zeros(H + 2)
    mu[0] = 1.0
    sim = local_sim.TabularSim(P, r, mu, H=H + 1, protocol="local", seed=seed)
    return local_sim.frontier_search_lock(sim, dead_obs=H + 1, goal_obs=H)


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--A", type=int, default=3)
    p.add_argument("--H", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--episodes", type=int, default=1000, help="T for the visit-count table")
    p.add_argument("--seeds", type=int, default=20)
    a = p.parse_args(argv)
    out = []
    print(f"A = {a.A} actions")
    print(f"{'H':>4}{'P(goal) random':>18}{'E[episodes] random':>22}{'reset steps (mean)':>22}{'H(A+1)/2':>10}")
    for H in a.H:
        P, _, _ = mdp_tools.combination_lock(H, a.A, seed=0)
        d = reach_probabilities(P, H)
        p_goal = d[H, H]
        steps = np.mean([reset_explorer_steps(H, a.A, s) for s in range(a.seeds)])
        out.append({"H": H, "p_goal": p_goal, "episodes": 1 / p_goal, "reset_steps": steps})
        print(f"{H:>4}{p_goal:>18.3e}{1 / p_goal:>22,.0f}{steps:>22.1f}{H * (a.A + 1) / 2:>10.1f}")
    H = a.H[len(a.H) // 2]
    P, _, _ = mdp_tools.combination_lock(H, a.A, seed=0)
    d = reach_probabilities(P, H)
    print(f"\nexpected visits to chain state h at step h after T = {a.episodes} random episodes (H = {H}):")
    print("  " + "  ".join(f"N_{h}(s_{h})={a.episodes * d[h, h]:.3g}" for h in range(H + 1) if h < H) )
    return out


if __name__ == "__main__":
    main()
