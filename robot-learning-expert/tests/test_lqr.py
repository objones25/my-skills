import numpy as np
import pytest

import lqr


def _system(seed=0, n=3, m=2):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n)) * 0.5, rng.normal(size=(n, m)), np.eye(n), 0.5 * np.eye(m)


def test_scalar_dare_closed_form():
    a, b, q, r = 1.2, 1.0, 1.0, 1.0
    # scalar DARE: p = q + a^2 p - (a b p)^2 / (r + b^2 p); unstable a, so K must stabilize
    P, K = lqr.dare(np.array([[a]]), np.array([[b]]), np.array([[q]]), np.array([[r]]))
    p = P[0, 0]
    assert p == pytest.approx(q + a * a * p - (a * b * p) ** 2 / (r + b * b * p))
    assert abs(a + b * K[0, 0]) < 1


def test_finite_horizon_matches_brute_force_two_steps():
    A, B, Q, R = _system()
    Ps, Ks = lqr.finite_horizon_lqr(A, B, Q, R, H=2)
    s = np.array([1.0, -2.0, 0.5])
    # minimize c(s, a) + s'^T Q s' over a, s' = A s + B a
    H2 = R + B.T @ Q @ B
    a_star = -np.linalg.solve(H2, B.T @ Q @ A @ s)
    s2 = A @ s + B @ a_star
    assert np.allclose(Ks[0] @ s, a_star)
    assert s @ Ps[0] @ s == pytest.approx(s @ Q @ s + a_star @ R @ a_star + s2 @ Q @ s2)


def test_fallback_iteration_matches_scipy(monkeypatch):
    A, B, Q, R = _system(1)
    P_scipy, _ = lqr.dare(A, B, Q, R)
    import builtins
    real_import = builtins.__import__

    def no_scipy(name, *args, **kw):
        if name.startswith("scipy"):
            raise ImportError
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    P_iter, _ = lqr.dare(A, B, Q, R)
    assert np.allclose(P_iter, P_scipy, atol=1e-8)


def test_rejects_indefinite_R():
    A, B, Q, _ = _system()
    with pytest.raises(ValueError, match="positive definite"):
        lqr.dare(A, B, Q, np.zeros((2, 2)))
