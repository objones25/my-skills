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


def test_rollout_cost_matches_hand_computation():
    # a = -1.5 s on s' = 2 s + a: states 1, 0.5, 0.25; costs 3.25 + 0.8125 + 0.203125
    A, B, Q, R = (np.array([[v]]) for v in (2.0, 1.0, 1.0, 1.0))
    K = np.array([[-1.5]])
    assert lqr.rollout_cost(A, B, Q, R, K, np.array([1.0]), H=3) == pytest.approx(4.265625)
    K_tv = np.stack([K, K, K])  # time-varying gains, shape (H, m, n)
    assert lqr.rollout_cost(A, B, Q, R, K_tv, np.array([1.0]), H=3) == pytest.approx(4.265625)


def test_rollout_cost_under_dare_gain_is_quadratic_value():
    A, B, Q, R = _system(2)
    P, K = lqr.dare(A, B, Q, R)
    s0 = np.array([0.3, -1.0, 2.0])
    assert lqr.rollout_cost(A, B, Q, R, K, s0, H=3000) == pytest.approx(s0 @ P @ s0, rel=1e-8)


def test_rollout_cost_noise_is_seeded_and_adds_cost():
    A, B, Q, R = _system(2)
    _, K = lqr.dare(A, B, Q, R)
    s0 = np.zeros(3)
    noisy = lqr.rollout_cost(A, B, Q, R, K, s0, H=50, noise_cov=0.1 * np.eye(3), seed=4)
    assert noisy == lqr.rollout_cost(A, B, Q, R, K, s0, H=50, noise_cov=0.1 * np.eye(3), seed=4)
    assert noisy > 0.0 == lqr.rollout_cost(A, B, Q, R, K, s0, H=50)


def test_estimate_AB_recovers_noiseless_system_exactly():
    A, B, _, _ = _system(3)
    rng = np.random.default_rng(3)
    S, U = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
    A_hat, B_hat = lqr.estimate_AB(S, U, S @ A.T + U @ B.T)
    assert A_hat.shape == (3, 3) and B_hat.shape == (3, 2)
    assert np.allclose(A_hat, A, atol=1e-10) and np.allclose(B_hat, B, atol=1e-10)


def test_rollout_cost_adds_noise_to_the_next_state():
    # s' = 0.5 s + w with a = 0: cost = s0^2 + (0.5 s0 + w0)^2, w0 from the same seeded generator
    A, B, Q, R = (np.array([[v]]) for v in (0.5, 0.0, 1.0, 1.0))
    cov = np.array([[0.2]])
    w0 = np.random.default_rng(9).multivariate_normal(np.zeros(1), cov)[0]
    got = lqr.rollout_cost(A, B, Q, R, np.zeros((1, 1)), np.array([1.0]), H=2, noise_cov=cov, seed=9)
    assert got == pytest.approx(1.0 + (0.5 + w0) ** 2)


def test_dare_residual_is_zero_only_at_the_solution():
    A, B, Q, R = _system(5)
    P, _ = lqr.dare(A, B, Q, R)
    assert lqr.dare_residual(A, B, Q, R, P) < 1e-8
    assert lqr.dare_residual(A, B, Q, R, P + 0.1 * np.eye(3)) > 1e-3


def test_is_stable_uses_spectral_radius():
    assert lqr.is_stable(np.diag([0.9, -0.5]))
    assert not lqr.is_stable(np.diag([0.9, -1.0]))
    assert not lqr.is_stable(np.array([[0.0, 2.0], [-2.0, 0.0]]))  # complex pair with |lambda| = 2
