import numpy as np
import pytest

import rl_math as rm


def _rollout(T=7, N=4, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(T, N))
    v = rng.normal(size=(T, N))
    nv = np.vstack([v[1:], rng.normal(size=(1, N))])
    return r, v, nv


def test_gae_lambda_one_equals_return_minus_value():
    r, v, nv = _rollout()
    z = np.zeros_like(r)
    adv, targets = rm.gae(r, v, nv, z, z, 0.97, 1.0)
    G = rm.discounted_returns(r, z, z, nv, 0.97)
    assert np.allclose(adv, G - v)
    assert np.allclose(targets, G)


def test_gae_lambda_zero_is_td_error():
    r, v, nv = _rollout()
    z = np.zeros_like(r)
    adv, _ = rm.gae(r, v, nv, z, z, 0.9, 0.0)
    assert np.allclose(adv, r + 0.9 * nv - v)


def test_termination_cuts_bootstrap_and_propagation():
    r, v, nv = _rollout(T=4, N=1)
    term = np.zeros_like(r)
    term[1] = 1
    z = np.zeros_like(r)
    adv, _ = rm.gae(r, v, nv, term, z, 0.9, 0.95)
    assert np.isclose(adv[1, 0], r[1, 0] - v[1, 0])  # no gamma * V(s') after a fall
    # step 0 sees step 1's advantage, step 1 does not see step 2's
    adv_alt, _ = rm.gae(r, v, nv * 0 + 123.0, term, z, 0.9, 0.95)
    assert np.isclose(adv_alt[1, 0], adv[1, 0])


def test_truncation_bootstraps_from_final_state_value():
    r, v, nv = _rollout(T=4, N=1)
    trunc = np.zeros_like(r)
    trunc[1] = 1
    z = np.zeros_like(r)
    adv, _ = rm.gae(r, v, nv, z, trunc, 0.9, 0.95)
    assert np.isclose(adv[1, 0], r[1, 0] + 0.9 * nv[1, 0] - v[1, 0])
    G = rm.discounted_returns(r, z, trunc, nv, 0.9)
    assert np.isclose(G[1, 0], r[1, 0] + 0.9 * nv[1, 0])


def test_truncated_vs_terminal_differ_exactly_by_bootstrap():
    r, v, nv = _rollout(T=3, N=1)
    flag = np.zeros_like(r)
    flag[2] = 1
    z = np.zeros_like(r)
    a_trunc, _ = rm.gae(r, v, nv, z, flag, 0.9, 1.0)
    a_term, _ = rm.gae(r, v, nv, flag, z, 0.9, 1.0)
    assert np.isclose(a_trunc[2, 0] - a_term[2, 0], 0.9 * nv[2, 0])


def test_rsl_rl_variant_matches_exact_without_timeouts_and_differs_with():
    r, v, nv = _rollout()
    z = np.zeros_like(r)
    a_exact, _ = rm.gae(r, v, nv, z, z, 0.99, 0.95)
    a_rsl, _ = rm.gae_rsl_rl(r, v, nv[-1], z, z, 0.99, 0.95)
    assert np.allclose(a_exact, a_rsl)
    to = z.copy()
    to[3] = 1
    a_rsl2, _ = rm.gae_rsl_rl(r, v, nv[-1], to, to, 0.99, 0.95)
    # rsl_rl bootstraps the time-out with V(s_t) instead of V(s_{t+1})
    assert np.allclose(a_rsl2[3], r[3] + 0.99 * v[3] - v[3])


def test_shapes_and_flags_validated():
    r, v, nv = _rollout()
    z = np.zeros_like(r)
    with pytest.raises(ValueError):
        rm.gae(r, v[:-1], nv, z, z, 0.9, 0.9)
    with pytest.raises(ValueError):
        rm.gae(r, v, nv, z + 0.5, z, 0.9, 0.9)
    with pytest.raises(ValueError):
        rm.gae(r, v, nv, z, z, 1.5, 0.9)


def test_kl_estimators_against_closed_form_gaussian():
    rng = np.random.default_rng(1)
    mu0, mu1, s = 0.0, 0.3, 1.0
    x = rng.normal(mu0, s, size=400_000)
    lp0 = -0.5 * ((x - mu0) / s) ** 2
    lp1 = -0.5 * ((x - mu1) / s) ** 2
    true = (mu1 - mu0) ** 2 / (2 * s**2)
    k = rm.kl_estimators(lp0, lp1)
    assert abs(k["k1"] - true) < 3e-3
    assert abs(k["k3"] - true) < 3e-3
    assert k["k3"] >= 0


def test_k3_nonnegative_per_sample():
    rng = np.random.default_rng(2)
    l = rng.normal(scale=2.0, size=1000)
    assert np.all(np.exp(l) - 1 - l >= 0)


def test_clip_fraction_and_explained_variance():
    assert rm.clip_fraction(np.log([1, 1, 1, 1]), np.log([1.0, 1.1, 1.3, 0.7]), 0.2) == 0.5
    R = np.array([1.0, 2.0, 3.0])
    assert rm.explained_variance(R, R) == 1.0
    assert rm.explained_variance(np.zeros(3), R) == pytest.approx(0.0)
    assert rm.explained_variance(R, np.ones(3)) is None


def test_ppo_clip_gradient_mask():
    rho = np.array([0.5, 0.5, 1.0, 1.5, 1.5])
    A = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    obj, active = rm.ppo_clip_objective(rho, A, 0.2)
    # rho=0.5, A>0: unclipped 0.5 < clipped 0.8 -> active (policy may still raise the prob)
    # rho=0.5, A<0: clipped branch -0.8 is smaller -> no gradient (already pushed down enough)
    # rho=1.5, A>0: clipped 1.2 smaller -> no gradient; rho=1.5, A<0: unclipped -1.5 -> active
    assert active.tolist() == [True, False, True, False, True]
    assert np.allclose(obj, [0.5, -0.8, 1.0, 1.2, -1.5])


def test_tanh_log_prob_matches_change_of_variables():
    u = np.array([[0.3, -1.2]])
    mean, log_std = np.zeros(2), np.zeros(2)
    a = np.tanh(u)
    expected = rm.gaussian_log_prob(u, mean, log_std) - np.sum(np.log(1 - a**2), axis=-1)
    assert np.allclose(rm.tanh_gaussian_log_prob(u, mean, log_std), expected)


def test_truncation_bias_numbers():
    b = rm.truncation_bias(1.0, 0.99, 1000)
    assert b["true_value"] == pytest.approx(100.0)
    assert b["terminal_last_step"] == 1.0
    # brute-force mean over the episode's states
    k = np.arange(1, 1001)
    mean = np.mean((1 - 0.99**k) / (1 - 0.99))
    assert b["terminal_mean_over_episode"] == pytest.approx(mean)
    assert 0.09 < b["relative_error_mean"] < 0.11


def test_ppo_clip_mask_zero_advantage_has_no_gradient():
    _, active = rm.ppo_clip_objective(np.array([0.5, 1.0, 1.5]), np.zeros(3), 0.2)
    assert not active.any()
