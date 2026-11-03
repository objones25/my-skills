import numpy as np
import pytest

import mdp_tools as mt


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_vi_pi_and_exact_eval_agree(seed, alpha):
    P, r = mt.random_mdp(7, 3, seed, alpha)
    Q_vi, _, _ = mt.q_value_iteration(P, r, 0.9, tol=1e-11)
    _, Q_pi, _ = mt.policy_iteration_exact(P, r, 0.9)
    assert np.allclose(Q_vi, Q_pi, atol=1e-8)


def test_uniform_mdp_value_is_closed_form():
    S, A, g = 4, 2, 0.8
    P = np.full((S, A, S), 1 / S)
    r = np.ones((S, A))
    Q, _, _ = mt.q_value_iteration(P, r, g, tol=1e-12)
    assert np.allclose(Q, 1 / (1 - g))


def test_enticing_trap_finite_vs_infinite_horizon():
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = 1.0
    P[1, :, 1] = 1.0
    r = np.array([[0.5, 1.0], [0.0, 0.0]])
    _, V, pi = mt.finite_horizon_dp(P, r, H=5)
    assert V[0, 0] == pytest.approx(0.5 * 4 + 1.0)
    assert pi[:4, 0].argmax(axis=1).tolist() == [0, 0, 0, 0] and pi[4, 0].argmax() == 1
    Q, _, _ = mt.q_value_iteration(P, r, 0.9, tol=1e-12)
    assert Q[0].argmax() == 0  # 0.5 / (1 - 0.9) = 5 > 1


def test_occupancy_is_a_distribution_and_matches_value():
    P, r = mt.random_mdp(5, 2, 3)
    pi = np.full((5, 2), 0.5)
    mu = np.full(5, 0.2)
    d = mt.discounted_occupancy(P, pi, mu, 0.9)
    assert d.sum() == pytest.approx(1.0) and (d >= -1e-12).all()
    _, V = mt.policy_eval_exact(P, r, 0.9, pi)
    assert (d * r).sum() / (1 - 0.9) == pytest.approx(mu @ V)


def test_sampler_matches_P():
    P, _ = mt.random_mdp(4, 2, 5)
    rng = np.random.default_rng(0)
    n = 200_000
    s = np.zeros(n, int)
    a = np.ones(n, int)
    counts = np.bincount(mt.sample_next_states(P, s, a, rng), minlength=4) / n
    assert np.abs(counts - P[0, 1]).max() < 0.01


def test_check_mdp_rejects_bad_rows():
    P = np.full((2, 1, 2), 0.4)
    with pytest.raises(ValueError, match="sums to"):
        mt.check_mdp(P, np.zeros((2, 1)))


def test_lock_needs_secret_path():
    P, r, secret = mt.combination_lock(H=4, A=3, seed=0)
    _, V, pi = mt.finite_horizon_dp(P, r, H=5)
    assert V[0, 0] == 1.0
    assert [pi[h, h].argmax() for h in range(4)] == secret.tolist()


def test_finite_horizon_eval_matches_dp_for_optimal_and_brute_force():
    import itertools

    P, r = mt.random_mdp(3, 2, 7, alpha=0.3)
    H = 3
    _, Vs, pis = mt.finite_horizon_dp(P, r, H)
    _, Ve = mt.finite_horizon_eval(P, r, pis, H)
    assert np.allclose(Vs, Ve)
    # optimum over all deterministic time-varying policies by enumeration
    best = -np.inf
    for choice in itertools.product(range(2), repeat=H * 3):
        pi = np.zeros((H, 3, 2))
        for i, a in enumerate(choice):
            pi[i // 3, i % 3, a] = 1.0
        best = max(best, mt.finite_horizon_eval(P, r, pi, H)[1][0, 0])
    assert best == pytest.approx(Vs[0, 0])
