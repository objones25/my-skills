"""Reference implementations of the small RL formulas that silently bias training when wrong.

numpy only, batched over parallel envs. Arrays are time-major: shape (T, N) for T rollout steps
and N envs, matching rsl_rl's RolloutStorage layout.

Use them as oracles: run your implementation and these on the same arrays and compare.

    python rl_math.py        # self-check, including the truncation-bias demo

Contents
    discounted_returns       reward-to-go with terminated / truncated flags and bootstrap values
    gae                      generalized advantage estimation, correct time-out bootstrapping
    gae_rsl_rl               rsl_rl 5.4.2's variant: r += gamma*V(s_t) on time-outs, then cut
    kl_estimators            Schulman's k1, k2, k3 from log-probs of samples drawn from the old policy
    clip_fraction, explained_variance, normalize
    ppo_clip_objective       per-sample clipped surrogate and whether its gradient is zero
    gaussian_log_prob, tanh_gaussian_log_prob
    truncation_bias          value of a constant-reward chain with and without bootstrapping
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_2d(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _check_flags(*flags: Array) -> None:
    for f in flags:
        if not np.isin(np.asarray(f), (0, 1, False, True)).all():
            raise ValueError("done flags must be 0/1 or bool")


def discounted_returns(
    rewards: Array, terminated: Array, truncated: Array, next_values: Array, gamma: float
) -> Array:
    """Reward-to-go G_t for a (T, N) rollout.

    terminated[t]: the episode really ended at step t (fall): no future value.
    truncated[t]:  the episode was cut at t (time limit): bootstrap with next_values[t], the critic's
                   value of the true final state s_{t+1} (before any auto-reset).
    next_values[T-1] also bootstraps the last step of a rollout that did not end.
    """
    r, term, trunc, nv = map(_as_2d, (rewards, terminated, truncated, next_values))
    _check_flags(term, trunc)
    if not (r.shape == term.shape == trunc.shape == nv.shape):
        raise ValueError(f"shape mismatch: {r.shape} {term.shape} {trunc.shape} {nv.shape}")
    T = r.shape[0]
    G = np.zeros_like(r)
    running = nv[T - 1] * (1.0 - term[T - 1])
    for t in reversed(range(T)):
        if t < T - 1:
            # a cut at t means the carried return belongs to another episode: restart from bootstrap
            running = np.where(trunc[t] > 0, nv[t], running)
            running = np.where(term[t] > 0, 0.0, running)
        G[t] = r[t] + gamma * running
        running = G[t]
    return G


def gae(
    rewards: Array,
    values: Array,
    next_values: Array,
    terminated: Array,
    truncated: Array,
    gamma: float,
    lam: float,
) -> tuple[Array, Array]:
    """GAE(gamma, lambda) advantages and value targets for a (T, N) rollout.

    values[t]      = V(s_t)
    next_values[t] = V(s_{t+1}) of the true next state, including the final state of a truncated
                     episode (store it before the env auto-resets).
    delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1},  done = terminated or truncated
    Returns (advantages, value_targets = advantages + values).
    """
    r, v, nv, term, trunc = map(_as_2d, (rewards, values, next_values, terminated, truncated))
    _check_flags(term, trunc)
    if not (r.shape == v.shape == nv.shape == term.shape == trunc.shape):
        raise ValueError("all inputs must share shape (T, N)")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError("gamma and lam must be in [0, 1]")
    done = np.maximum(term, trunc)
    adv = np.zeros_like(r)
    last = np.zeros(r.shape[1])
    for t in reversed(range(r.shape[0])):
        delta = r[t] + gamma * (1.0 - term[t]) * nv[t] - v[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + v


def gae_rsl_rl(
    rewards: Array, values: Array, last_values: Array, dones: Array, time_outs: Array, gamma: float, lam: float
) -> tuple[Array, Array]:
    """rsl_rl 5.4.2's approximation (algorithms/ppo.py process_env_step + compute_returns).

    On a time-out the env has already reset, so the true V(s_{t+1}) is unavailable; rsl_rl adds
    gamma * V(s_t) to that step's reward, then treats every done (time-out or fall) as terminal.
    last_values (N,) bootstraps the end of the rollout. Compare with gae() to see the size of the
    approximation on your own data.
    """
    r, v, d, to = map(_as_2d, (rewards, values, dones, time_outs))
    _check_flags(d, to)
    r = r + gamma * v * to
    T, N = r.shape
    nv = np.vstack([v[1:], np.asarray(last_values, float).reshape(1, N)])
    return gae(r, v, nv, d, np.zeros_like(d), gamma, lam)


def kl_estimators(logp_old: Array, logp_new: Array) -> dict[str, float]:
    """Estimate KL(old || new) from samples x ~ old (Schulman, "Approximating KL Divergence", 2020).

    With log ratio l = log(new(x)/old(x)) and ratio rho = exp(l):
      k1 = -l                 unbiased, high variance, can be negative
      k2 = l**2 / 2           biased, low variance (accurate when the policies are close)
      k3 = (rho - 1) - l      unbiased, low variance, always >= 0  (CleanRL, SB3, the G1 project)
    """
    l = np.asarray(logp_new, float) - np.asarray(logp_old, float)
    rho = np.exp(l)
    return {"k1": float(np.mean(-l)), "k2": float(np.mean(0.5 * l**2)), "k3": float(np.mean(rho - 1.0 - l))}


def clip_fraction(logp_old: Array, logp_new: Array, clip: float) -> float:
    """Fraction of samples whose probability ratio left [1 - clip, 1 + clip]."""
    if clip <= 0:
        raise ValueError("clip must be positive")
    rho = np.exp(np.asarray(logp_new, float) - np.asarray(logp_old, float))
    return float(np.mean(np.abs(rho - 1.0) > clip))


def explained_variance(values: Array, returns: Array) -> float | None:
    """1 - Var(returns - values) / Var(returns); None when returns are constant (undefined)."""
    v, R = np.asarray(values, float).ravel(), np.asarray(returns, float).ravel()
    var = R.var()
    if var == 0.0:
        return None
    return float(1.0 - (R - v).var() / var)


def normalize(x: Array, eps: float = 1e-8) -> Array:
    """(x - mean) / (std + eps) over all elements: rsl_rl's batch advantage normalization."""
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + eps)


def ppo_clip_objective(ratio: Array, adv: Array, clip: float) -> tuple[Array, Array]:
    """Per-sample min(rho*A, clip(rho)*A) and a mask that is True where d/d rho is nonzero.

    The gradient vanishes exactly when the clipped branch is the smaller one, i.e. when rho has
    already moved past 1 +/- clip in the direction that improves the objective.
    """
    rho, A = np.asarray(ratio, float), np.asarray(adv, float)
    unclipped = rho * A
    clipped = np.clip(rho, 1.0 - clip, 1.0 + clip) * A
    obj = np.minimum(unclipped, clipped)
    active = ((unclipped <= clipped) | ((rho >= 1.0 - clip) & (rho <= 1.0 + clip))) & (A != 0)
    return obj, active


def gaussian_log_prob(a: Array, mean: Array, log_std: Array) -> Array:
    """Log-density of a diagonal Gaussian, summed over the last axis."""
    a, mean, log_std = (np.asarray(x, float) for x in (a, mean, log_std))
    z = (a - mean) / np.exp(log_std)
    return np.sum(-0.5 * z**2 - log_std - 0.5 * np.log(2 * np.pi), axis=-1)


def tanh_gaussian_log_prob(u: Array, mean: Array, log_std: Array) -> Array:
    """Log-density of a = tanh(u), u ~ N(mean, std): subtract sum log(1 - tanh(u)^2) (SAC App. C).

    Uses the stable identity log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)).
    """
    u = np.asarray(u, float)
    corr = 2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))
    return gaussian_log_prob(u, mean, log_std) - np.sum(corr, axis=-1)


def truncation_bias(reward: float, gamma: float, horizon: int) -> dict[str, float]:
    """A robot that earns `reward` every step forever, observed in episodes cut at `horizon`.

    true value         reward / (1 - gamma)
    time-out treated as terminal: the value at the start of the episode is
                       reward * (1 - gamma**horizon) / (1 - gamma), and states near the cut are
                       valued near `reward`: the critic learns that the world ends.
    mean over the episode's states (k = 1..horizon steps left):
                       reward/(1-gamma) * (1 - gamma*(1-gamma**horizon) / (horizon*(1-gamma)))
    The terminal-treated value also depends on time left, which the policy usually cannot observe,
    so the critic is fitting a non-Markov target.
    """
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    true = reward / (1 - gamma)
    start = reward * (1 - gamma**horizon) / (1 - gamma)
    mean = true * (1 - gamma * (1 - gamma**horizon) / (horizon * (1 - gamma)))
    return {"true_value": true, "terminal_at_start": start, "terminal_last_step": reward,
            "terminal_mean_over_episode": mean,
            "relative_error_at_start": (true - start) / true,
            "relative_error_mean": (true - mean) / true}


def _self_check() -> None:
    rng = np.random.default_rng(0)
    T, N, g, lam = 6, 3, 0.9, 0.95
    r = rng.normal(size=(T, N))
    v = rng.normal(size=(T, N))
    nv = np.vstack([v[1:], rng.normal(size=(1, N))])
    z = np.zeros((T, N))
    # lam = 1 and no dones: GAE advantage equals the bootstrapped return minus V
    adv, _ = gae(r, v, nv, z, z, g, 1.0)
    G = discounted_returns(r, z, z, nv, g)
    assert np.allclose(adv, G - v)
    # lam = 0: advantage is the one-step TD error
    adv0, _ = gae(r, v, nv, z, z, g, 0.0)
    assert np.allclose(adv0, r + g * nv - v)
    # rsl_rl variant equals the exact one when no time-outs occur
    a1, _ = gae_rsl_rl(r, v, nv[-1], z, z, g, lam)
    a2, _ = gae(r, v, nv, z, z, g, lam)
    assert np.allclose(a1, a2)
    k = kl_estimators(np.log([0.5, 0.5]), np.log([0.5, 0.5]))
    assert all(abs(x) < 1e-12 for x in k.values())
    b = truncation_bias(1.0, 0.99, 1000)
    assert abs(b["true_value"] - 100.0) < 1e-9 and abs(b["terminal_at_start"] - 100 * (1 - 0.99**1000)) < 1e-9
    print("rl_math self-check passed")
    print("truncation demo (r=1, gamma=0.99, 20 s at 50 Hz = 1000 steps):")
    print(f"  true value {b['true_value']:.1f}; terminal-treated: start {b['terminal_at_start']:.1f}, "
          f"episode mean {b['terminal_mean_over_episode']:.1f} ({100 * b['relative_error_mean']:.1f}% low), "
          f"last step {b['terminal_last_step']:.1f}")
    b2 = truncation_bias(1.0, 0.99, 24)
    print(f"if the 24-step rollout boundary were treated as terminal: start value {b2['terminal_at_start']:.2f} vs {b2['true_value']:.0f}")


if __name__ == "__main__":
    _self_check()
