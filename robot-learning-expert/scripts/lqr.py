"""LQR ground truth: Riccati recursion, DARE, rollouts, and least-squares system ID.

Notes 2 call LQR the "sanity check" for RL algorithms: if a method cannot learn
LQR, something is probably wrong with it. This file gives the exact answer to
compare against.

Conventions (Notes 2, cost form):
    s_{t+1} = A s_t + B a_t + w_t,   cost c(s, a) = s'Qs + a'Ra,  Q >= 0, R > 0
    A (n, n), B (n, m), Q (n, n), R (m, m), gain K (m, n) with a = K s.
Note the sign: the notes write a = K s with K = -(R + B'P'B)^{-1} B'P'A.
AJKS (2026 draft) ch. 14 instead writes pi*(x) = -K* x, with x for states and u for actions.

Self-check:  python lqr.py
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _check(A: Array, B: Array, Q: Array, R: Array) -> tuple[int, int]:
    n, m = B.shape
    if A.shape != (n, n) or Q.shape != (n, n) or R.shape != (m, m):
        raise ValueError(f"shape mismatch: A{A.shape} B{B.shape} Q{Q.shape} R{R.shape}")
    if np.linalg.eigvalsh((R + R.T) / 2).min() <= 0:
        raise ValueError("R must be positive definite")
    if np.linalg.eigvalsh((Q + Q.T) / 2).min() < -1e-12:
        raise ValueError("Q must be positive semidefinite")
    return n, m


def riccati_step(A: Array, B: Array, Q: Array, R: Array, P_next: Array) -> tuple[Array, Array]:
    """One backward Bellman step (Notes 2, Thm 2 proof). Returns (P, K)."""
    K = -np.linalg.solve(R + B.T @ P_next @ B, B.T @ P_next @ A)
    Acl = A + B @ K
    P = Q + K.T @ R @ K + Acl.T @ P_next @ Acl
    return (P + P.T) / 2, K


def finite_horizon_lqr(A: Array, B: Array, Q: Array, R: Array, H: int) -> tuple[Array, Array]:
    """Finite-horizon gains for objective sum_{t=1}^{H} c(s_t, a_t) (Notes 2 eq. 3).

    Base case V_H(s) = s'Qs with a_H = 0, as in the notes.
    Returns P (H, n, n) with V_h(s) = s' P[h] s, and K (H, m, n) with a_h = K[h] s (K[H-1] = 0).
    Index h = 0..H-1 corresponds to the notes' h = 1..H.
    """
    n, m = _check(A, B, Q, R)
    Ps = np.zeros((H, n, n))
    Ks = np.zeros((H, m, n))
    Ps[H - 1] = Q
    for h in range(H - 2, -1, -1):
        Ps[h], Ks[h] = riccati_step(A, B, Q, R, Ps[h + 1])
    return Ps, Ks


def dare(A: Array, B: Array, Q: Array, R: Array, tol: float = 1e-12, max_iters: int = 100_000) -> tuple[Array, Array]:
    """Infinite-horizon, undiscounted solution of the DARE.

    Uses scipy.linalg.solve_discrete_are when scipy is installed, else iterates
    the Riccati step to a fixed point (value iteration on quadratic V).
    Returns (P, K) with a = K s.
    """
    _check(A, B, Q, R)
    try:
        from scipy.linalg import solve_discrete_are

        P = solve_discrete_are(A, B, Q, R)
    except ImportError:
        P = Q.copy()
        for _ in range(max_iters):
            P_new, _ = riccati_step(A, B, Q, R, P)
            if np.abs(P_new - P).max() <= tol * max(1.0, np.abs(P).max()):
                P = P_new
                break
            P = P_new
        else:
            raise RuntimeError("Riccati iteration did not converge; is (A, B) stabilizable?")
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def dare_residual(A: Array, B: Array, Q: Array, R: Array, P: Array) -> float:
    """Max-abs residual of Q + A'PA - A'PB (R + B'PB)^{-1} B'PA - P."""
    G = A.T @ P @ B
    res = Q + A.T @ P @ A - G @ np.linalg.solve(R + B.T @ P @ B, G.T) - P
    return float(np.abs(res).max())


def rollout_cost(
    A: Array, B: Array, Q: Array, R: Array, K: Array, s0: Array, H: int,
    noise_cov: Array | None = None, seed: int = 0,
) -> float:
    """Total cost of a = K s (K may be (m, n) or time-varying (H, m, n)) for H steps from s0."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s0, dtype=float)
    total = 0.0
    for t in range(H):
        Kt = K[t] if K.ndim == 3 else K
        a = Kt @ s
        total += float(s @ Q @ s + a @ R @ a)
        w = rng.multivariate_normal(np.zeros(len(s)), noise_cov) if noise_cov is not None else 0.0
        s = A @ s + B @ a + w
    return total


def estimate_AB(S: Array, U: Array, S_next: Array) -> tuple[Array, Array]:
    """Least-squares system ID (Notes 4 §4.1): argmin ||A s + B a - s'||^2.

    S (N, n), U (N, m), S_next (N, n). Query with isotropic Gaussian s and a so
    the regressors are well conditioned.
    """
    X = np.hstack([S, U])
    Theta, *_ = np.linalg.lstsq(X, S_next, rcond=None)  # (n + m, n)
    n = S.shape[1]
    return Theta[:n].T, Theta[n:].T


def is_stable(A_cl: Array) -> bool:
    """Discrete-time stability: spectral radius of the closed-loop matrix < 1."""
    return bool(np.abs(np.linalg.eigvals(A_cl)).max() < 1.0)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n, m = 4, 2
    A = rng.normal(size=(n, n)) * 0.6
    B = rng.normal(size=(n, m))
    Q, R = np.eye(n), 0.1 * np.eye(m)
    P, K = dare(A, B, Q, R)
    assert dare_residual(A, B, Q, R, P) < 1e-8
    assert is_stable(A + B @ K)
    Ps, Ks = finite_horizon_lqr(A, B, Q, R, H=400)
    assert np.allclose(Ps[0], P, atol=1e-6), "long finite horizon should approach the DARE solution"
    s0 = rng.normal(size=n)
    assert abs(rollout_cost(A, B, Q, R, K, s0, H=2000) - s0 @ P @ s0) < 1e-6 * (1 + s0 @ P @ s0)
    N = 2000
    Sx, Ux = rng.normal(size=(N, n)), rng.normal(size=(N, m))
    Sn = Sx @ A.T + Ux @ B.T + 0.01 * rng.normal(size=(N, n))
    Ah, Bh = estimate_AB(Sx, Ux, Sn)
    assert np.abs(Ah - A).max() < 0.01 and np.abs(Bh - B).max() < 0.01
    print("ok: DARE residual, finite->infinite horizon limit, rollout cost = s0'Ps0, sys-ID")


if __name__ == "__main__":
    _self_check()
