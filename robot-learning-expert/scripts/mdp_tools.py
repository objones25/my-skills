"""Ground-truth tools for small tabular MDPs.

Use these as ORACLES: exact answers to check learned or approximate
quantities against (Q-learning estimates, a model-based policy, RVFS output on
a toy problem). They are written for clarity and cross-checking, not speed.

Array conventions (also in references/notation.md):
    P   (S, A, S)   P[s, a, s'] = P(s' | s, a); each P[s, a] sums to 1.
    r   (S, A)      r[s, a], deterministic reward.
    pi  (S, A)      pi[s, a] = probability of action a in state s.
    Q   (S, A),  V (S,),  mu (S,) initial-state distribution.
Finite-horizon versions accept time-indexed P (H, S, A, S) and r (H, S, A).

Self-check:  python mdp_tools.py
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Validation (fail fast with a message that says what is wrong)
# ---------------------------------------------------------------------------
def check_mdp(P: Array, r: Array, atol: float = 1e-8) -> tuple[int, int]:
    """Validate shapes and probability rows. Returns (S, A)."""
    if P.ndim != 3:
        raise ValueError(f"P must have shape (S, A, S); got {P.shape}")
    S, A, S2 = P.shape
    if S != S2:
        raise ValueError(f"P last axis must equal first axis (S); got {P.shape}")
    if r.shape != (S, A):
        raise ValueError(f"r must have shape (S, A) = {(S, A)}; got {r.shape}")
    if (P < -atol).any():
        raise ValueError("P has negative entries")
    row = P.sum(axis=2)
    bad = np.argwhere(np.abs(row - 1.0) > atol)
    if bad.size:
        s, a = bad[0]
        raise ValueError(f"P[{s}, {a}] sums to {row[s, a]:.6g}, not 1")
    return S, A


def check_policy(pi: Array, S: int, A: int, atol: float = 1e-8) -> None:
    if pi.shape != (S, A):
        raise ValueError(f"pi must have shape (S, A) = {(S, A)}; got {pi.shape}")
    if (pi < -atol).any() or np.abs(pi.sum(axis=1) - 1).max() > atol:
        raise ValueError("each pi[s] must be a probability vector")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def random_mdp(S: int, A: int, seed: int, alpha: float = 1.0) -> tuple[Array, Array]:
    """Dirichlet(alpha) transitions, Uniform[0, 1] rewards.

    Small alpha (e.g. 0.1) gives near-deterministic transitions; large alpha
    gives near-uniform ones. Vary it: bugs that hide on one regime show on the other.
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha * np.ones(S), size=(S, A))
    r = rng.uniform(0.0, 1.0, size=(S, A))
    return P, r


def combination_lock(H: int, A: int, seed: int, p_slip: float = 0.0) -> tuple[Array, Array, Array]:
    """Hard-exploration chain ("combination lock").

    States 0..H-1 are the chain, state H is the goal (absorbing, reward 1 per
    step for any action), state H+1 is a dead state (absorbing, reward 0).
    At chain state h exactly one secret action advances to h+1; every other
    action drops to the dead state. With p_slip > 0 the correct action still
    drops to the dead state with that probability.

    A uniformly random policy reaches the goal with probability A**(-H), so
    episodic exploration from s0 = 0 needs on the order of A**H episodes,
    while an agent that can reset to the furthest chain state it has reached
    needs on the order of H * A tries. That gap is the point of
    local-simulator / reset access.

    Returns (P, r, secret) where secret[h] is the correct action at state h.
    """
    if not (0.0 <= p_slip < 1.0):
        raise ValueError("p_slip must be in [0, 1)")
    rng = np.random.default_rng(seed)
    secret = rng.integers(A, size=H)
    S = H + 2
    goal, dead = H, H + 1
    P = np.zeros((S, A, S))
    r = np.zeros((S, A))
    for h in range(H):
        for a in range(A):
            if a == secret[h]:
                P[h, a, h + 1] += 1.0 - p_slip
                P[h, a, dead] += p_slip
            else:
                P[h, a, dead] = 1.0
    P[goal, :, goal] = 1.0
    P[dead, :, dead] = 1.0
    r[goal, :] = 1.0
    return P, r, secret


# ---------------------------------------------------------------------------
# Exact solvers (infinite-horizon discounted)
# ---------------------------------------------------------------------------
def P_pi_matrix(P: Array, pi: Array) -> Array:
    """State-action closed-loop transition matrix (SA, SA), Notes 1 Def. 13.

    P_pi[(s, a), (s', a')] = P(s' | s, a) * pi(a' | s').
    """
    S, A, _ = P.shape
    return np.einsum("ijk,kl->ijkl", P, pi).reshape(S * A, S * A)


def policy_eval_exact(P: Array, r: Array, gamma: float, pi: Array) -> tuple[Array, Array]:
    """Q^pi = (I - gamma P^pi)^{-1} r by a linear solve (Notes 1 eq. 9).

    Returns (Q, V). Independent of any Bellman iteration, so it makes a good
    cross-check for iterative policy evaluation.
    """
    S, A = check_mdp(P, r)
    check_policy(pi, S, A)
    M = np.eye(S * A) - gamma * P_pi_matrix(P, pi)
    Q = np.linalg.solve(M, r.reshape(-1)).reshape(S, A)
    V = (pi * Q).sum(axis=1)
    return Q, V


def bellman_Q_opt(P: Array, r: Array, gamma: float, Q: Array) -> Array:
    """(T* Q)(s, a) = r(s, a) + gamma * E_{s'} max_a' Q(s', a')."""
    return r + gamma * P @ Q.max(axis=1)


def q_value_iteration(
    P: Array, r: Array, gamma: float, tol: float = 1e-10, max_iters: int = 100_000
) -> tuple[Array, int, float]:
    """Q-value iteration from Q = 0 until the a-posteriori error bound <= tol.

    Uses ||Q_k - Q*||_inf <= gamma / (1 - gamma) * ||Q_k - Q_{k-1}||_inf,
    which follows from the gamma-contraction of T*.
    Returns (Q, iterations, certified error bound).
    """
    check_mdp(P, r)
    if not (0.0 <= gamma < 1.0):
        raise ValueError("discounted solver needs gamma in [0, 1)")
    Q = np.zeros_like(r, dtype=float)
    for k in range(1, max_iters + 1):
        Q_new = bellman_Q_opt(P, r, gamma, Q)
        bound = gamma / (1 - gamma) * np.abs(Q_new - Q).max() if gamma > 0 else 0.0
        Q = Q_new
        if bound <= tol:
            return Q, k, bound
    raise RuntimeError(f"no convergence to tol={tol} in {max_iters} iterations")


def greedy(Q: Array) -> Array:
    """One-hot greedy policy (S, A). Ties go to the lowest action index, like np.argmax."""
    S, A = Q.shape
    pi = np.zeros((S, A))
    pi[np.arange(S), Q.argmax(axis=1)] = 1.0
    return pi


def policy_iteration_exact(P: Array, r: Array, gamma: float, max_iters: int = 10_000) -> tuple[Array, Array, int]:
    """Policy iteration with exact evaluation; terminates in finitely many steps.

    Starts from 'action 0 everywhere'. Returns (pi, Q, iterations).
    """
    S, A = check_mdp(P, r)
    pi = np.zeros((S, A))
    pi[:, 0] = 1.0
    for k in range(1, max_iters + 1):
        Q, _ = policy_eval_exact(P, r, gamma, pi)
        # Keep the current action on ties so the loop cannot cycle between equal-valued actions.
        cur = pi.argmax(axis=1)
        best = Q.max(axis=1)
        keep = np.isclose(Q[np.arange(S), cur], best, rtol=0, atol=1e-12)
        new_a = np.where(keep, cur, Q.argmax(axis=1))
        new_pi = np.zeros((S, A))
        new_pi[np.arange(S), new_a] = 1.0
        if np.array_equal(new_pi, pi):
            return pi, Q, k
        pi = new_pi
    raise RuntimeError("policy iteration did not terminate")


def discounted_occupancy(P: Array, pi: Array, mu: Array, gamma: float) -> Array:
    """State-action occupancy d^pi(s, a) = (1 - gamma) sum_t gamma^t Pr(s_t = s, a_t = a).

    Shape (S, A), sums to 1. Needed for coverage / concentrability quantities.
    """
    S, A, _ = P.shape
    start = (mu[:, None] * pi).reshape(-1)  # Pr(s_0, a_0)
    M = np.eye(S * A) - gamma * P_pi_matrix(P, pi)
    d = (1 - gamma) * np.linalg.solve(M.T, start)
    return d.reshape(S, A)


def suboptimality(P: Array, r: Array, gamma: float, pi: Array, mu: Array | None = None) -> float:
    """E_{s~mu}[V*(s) - V^pi(s)] (max over s if mu is None). Ground truth for a learned policy."""
    Qs, _, _ = q_value_iteration(P, r, gamma, tol=1e-12)
    Vs = Qs.max(axis=1)
    _, Vpi = policy_eval_exact(P, r, gamma, pi)
    gap = Vs - Vpi
    return float(gap.max() if mu is None else mu @ gap)


# ---------------------------------------------------------------------------
# Finite horizon
# ---------------------------------------------------------------------------
def finite_horizon_dp(P: Array, r: Array, H: int) -> tuple[Array, Array, Array]:
    """Backward induction for the undiscounted H-step objective (Notes 1 §8).

    P may be (S, A, S) or time-indexed (H, S, A, S); r may be (S, A) or (H, S, A).
    Index h = 0..H-1 here corresponds to the notes' h = 1..H.
    Returns (Q, V, pi) with shapes (H, S, A), (H + 1, S), (H, S, A); V[H] = 0.
    The optimal policy is generally NOT stationary: pi[h] differs across h.
    """
    Ph = P if P.ndim == 4 else np.broadcast_to(P, (H, *P.shape))
    rh = r if r.ndim == 3 else np.broadcast_to(r, (H, *r.shape))
    if Ph.shape[0] != H or rh.shape[0] != H:
        raise ValueError("time-indexed P and r must have leading dimension H")
    S, A = rh.shape[1:]
    V = np.zeros((H + 1, S))
    Q = np.zeros((H, S, A))
    pi = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        Q[h] = rh[h] + Ph[h] @ V[h + 1]
        V[h] = Q[h].max(axis=1)
        pi[h, np.arange(S), Q[h].argmax(axis=1)] = 1.0
    return Q, V, pi


def finite_horizon_eval(P: Array, r: Array, pi: Array, H: int) -> tuple[Array, Array]:
    """Exact H-step value of a (possibly time-varying) policy.

    pi may be (S, A) stationary or (H, S, A). Returns (Q, V) with shapes (H, S, A), (H + 1, S).
    J(pi) = mu @ V[0]. Use this to score a learned policy against finite_horizon_dp's optimum.
    """
    Ph = P if P.ndim == 4 else np.broadcast_to(P, (H, *P.shape))
    rh = r if r.ndim == 3 else np.broadcast_to(r, (H, *r.shape))
    pih = pi if pi.ndim == 3 else np.broadcast_to(pi, (H, *pi.shape))
    S, A = rh.shape[1:]
    V = np.zeros((H + 1, S))
    Q = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        Q[h] = rh[h] + Ph[h] @ V[h + 1]
        V[h] = (pih[h] * Q[h]).sum(axis=1)
    return Q, V


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_next_states(P: Array, s: Array, a: Array, rng: np.random.Generator) -> Array:
    """Vectorized s' ~ P(.|s, a) for integer arrays s, a of the same shape (inverse-CDF)."""
    cdf = np.cumsum(P[s, a], axis=-1)
    u = rng.random(size=np.shape(s))[..., None]
    nxt = (u > cdf).sum(axis=-1)
    return np.minimum(nxt, P.shape[2] - 1)  # guards rows whose cdf ends at 1 - eps


def _self_check() -> None:
    P, r = random_mdp(6, 3, seed=0, alpha=0.5)
    g = 0.9
    Q_vi, k, bound = q_value_iteration(P, r, g, tol=1e-11)
    pi_pi, Q_pi, n_pi = policy_iteration_exact(P, r, g)
    Q_eval, _ = policy_eval_exact(P, r, g, greedy(Q_vi))
    assert np.allclose(Q_vi, Q_pi, atol=1e-8), "VI and PI disagree"
    assert np.allclose(Q_vi, Q_eval, atol=1e-8), "greedy(Q*) is not optimal under exact evaluation"
    Pl, rl, secret = combination_lock(H=5, A=3, seed=1)
    Qf, Vf, pif = finite_horizon_dp(Pl, rl, H=6)
    assert Vf[0, 0] == 1.0, "finite-horizon DP should find the lock's single rewarding path"
    print(f"ok: VI {k} iters (bound {bound:.1e}), PI {n_pi} iters, lock secret {secret.tolist()}")


if __name__ == "__main__":
    _self_check()
