"""Tests for the massively-parallel-rl scripts. Run from the skill root:

    python -m pytest tests/ -q

The paper-reproduction tests pin the skill's claims about Zheng et al. (2026):
if one fails, a statement in references/scaling-laws.md is wrong.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

import elo_league as el  # noqa: E402
import fit_scaling_laws as fsl  # noqa: E402


def run(script, *args, timeout=900):
    return subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)],
                          capture_output=True, text=True, timeout=timeout, cwd=SCRIPTS)


# ------------------------------------------------------------------ Elo league

@pytest.mark.parametrize("pred,actual,paper", [
    (1151.5, 1110.7, 5.84), (2043.0, 2087.5, -6.37), (1392.5, 1312.3, 11.34),
    (2266.8, 2185.9, 11.44), (1102.6, 1110.7, -1.17), (2062.8, 2087.5, -3.55)])
def test_win_rate_error_matches_paper_tables_2_and_3(pred, actual, paper):
    assert round(100 * el.winrate_deviation(pred - actual), 2) == paper


def test_bradley_terry_recovers_known_ratings():
    # Gumbel noise: the difference of two Gumbels is logistic, so pairwise wins
    # follow the Bradley-Terry model exactly.
    rng = np.random.default_rng(0)
    true = np.array([0.0, 200.0, 400.0, 600.0])
    returns = true[:, None] / el.ELO_SCALE + rng.gumbel(size=(4, 20000))
    elo = el.elo_league(returns, reference=0)
    assert np.allclose(elo, true, atol=15)


def test_ties_count_half():
    w = el.win_rate_matrix(np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert w[0, 1] == pytest.approx(0.75)
    assert w[1, 0] == pytest.approx(0.25)


def test_undefeated_agent_is_flagged_and_finite():
    returns = np.array([[0.0] * 50, [1.0] * 50])
    w = el.win_rate_matrix(returns)
    assert el.undefeated_or_winless(w) == [0, 1]
    assert np.all(np.isfinite(el.elo_league(returns, 0)))


def test_elo_cli_roundtrip(tmp_path):
    path = tmp_path / "r.csv"
    rng = np.random.default_rng(1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", *[f"ep{i}" for i in range(200)]])
        for name, mu in [("random", 0.0), ("a", 1.0), ("b", 2.0)]:
            w.writerow([name, *(mu + rng.gumbel(size=200)).tolist()])
    out = run("elo_league.py", path, "--reference", "random", "--json", tmp_path / "o.json")
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[1].startswith("b") and lines[-1].startswith("random")


# ------------------------------------------------------------- scaling laws

TABLE1 = {"starpilot": (dict(aB=7.14e-4, dB=0.39, a_lr=4.08e-3, d_lr=-0.023),
                        dict(aM=5.3, dM=0.44, a_score=-3186, d_score=240.97)),
          "craftax": (dict(aB=0.28, dB=0.26, a_lr=1.57e-2, d_lr=-0.082),
                      dict(aM=0.53, dM=0.43, a_score=-1987, d_score=223.86))}
TABLES_6_7 = {"starpilot": {1e15: (9, -9.12), 3e15: (10, -8.97), 1e16: (10, -9.35), 3e16: (11, -9.19),
                            1e17: (12, -9.07), 1e18: (13, -9.30), 1e19: (14, -9.53)},
              "craftax": {1e15: (11, -10.17), 3e15: (11, -10.51), 1e16: (12, -10.38), 3e16: (12, -10.71),
                          1e17: (13, -10.58), 1e18: (14, -10.79), 1e19: (15, -11.00)}}


def test_predict_reproduces_appendix_configs():
    """13 of 14 batch sizes exact; learning rates within 0.06 in log2 except the one
    budget where the rounded batch differs (Table 1 coefficients are rounded)."""
    batch_hits, lr_errs = 0, []
    for env, rows in TABLES_6_7.items():
        hp, iso = TABLE1[env]
        for c, (log2_b, log2_lr) in rows.items():
            r = fsl.predict(hp, iso, c)
            if int(math.log2(r["batch"])) == log2_b:
                batch_hits += 1
                lr_errs.append(abs(r["log2_lr"] - log2_lr))
    assert batch_hits == 13
    assert max(lr_errs) < 0.07


def test_predict_reproduces_table3_elo():
    hp, iso = TABLE1["starpilot"]
    assert fsl.predict(hp, iso, 1e18)["score"] == pytest.approx(1151.5, abs=0.2)
    assert fsl.predict(hp, iso, 1e19)["score"] == pytest.approx(1392.5, abs=0.2)


def _synthetic_iso(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for c in [1e15, 3e15, 1e16, 3e16, 1e17, 1e18]:
        ms, es = 0.5 * c ** 0.43, 224 * np.log10(c) - 1987
        for m in ms * np.logspace(-0.6, 0.6, 7):
            for s in range(10):
                rows.append(dict(flops=c, fpf=m, seed=s,
                                 score=es - 300 * np.log10(m / ms) ** 2 + rng.normal(0, 25)))
    path = tmp_path / "iso.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_isoflops_recovers_exponents(tmp_path):
    path = _synthetic_iso(tmp_path)
    out = run("fit_scaling_laws.py", "isoflops", path, "--holdout", "1e18", "--json", tmp_path / "iso.json")
    assert out.returncode == 0, out.stderr
    import json
    fit = json.load(open(tmp_path / "iso.json"))
    assert fit["dM"] == pytest.approx(0.43, abs=0.03)
    assert fit["d_score"] == pytest.approx(224, abs=6)


def test_isoflops_warns_when_grid_misses_optimum(tmp_path):
    rows = [dict(flops=c, fpf=m, seed=s, score=-(np.log10(m) - 9) ** 2 + 0.01 * s)
            for c in [1e15, 1e16, 1e17] for m in np.logspace(5, 7, 5) for s in range(3)]
    path = tmp_path / "bad.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run("fit_scaling_laws.py", "isoflops", path)
    assert "outside the tested sizes" in out.stderr


def test_hparams_recovers_exponents(tmp_path):
    """Across six synthetic datasets, batch exponent error was <=0.025 and learning-rate
    exponent error up to 0.04 (the lr optimum moves less than one grid step over the
    whole range), so the tolerances are 0.05."""
    rng = np.random.default_rng(2)
    rows = []
    for c in [3e14, 1e15, 3e15, 1e16, 3e16, 1e17]:
        b0, l0 = np.log2(0.28 * c ** 0.26), np.log2(0.0157 * c ** -0.082)
        for b in range(int(round(b0)) - 3, int(round(b0)) + 4):
            for lg in np.arange(-12, -8.9, 0.5):
                for s in range(5):
                    rows.append(dict(flops=c, model=1, batch_size=2 ** b, lr=2.0 ** lg, seed=s,
                                     score=60 - 2 * (b - b0) ** 2 - 3 * (lg - l0) ** 2 + rng.normal(0, 1.0)))
    path = tmp_path / "grid.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run("fit_scaling_laws.py", "hparams", path, "--boot", "100", "--json", tmp_path / "hp.json")
    assert out.returncode == 0, out.stderr
    import json
    fit = json.load(open(tmp_path / "hp.json"))
    assert fit["d_lr"] == pytest.approx(-0.082, abs=0.05)
    assert fit["dB"] == pytest.approx(0.26, abs=0.05)


# ------------------------------------------------------------ JAX utilities

def test_flops_per_frame_matches_hand_count():
    import jax.numpy as jnp
    from flops_per_frame import forward_flops, mlp_apply
    f = forward_flops(mlp_apply(100, [64, 64], 4), jnp.ones((100,)))
    assert f == pytest.approx(2 * (100 * 64 + 64 * 64 + 64 * 4), rel=0.01)


def test_autoreset_separates_truncation_from_termination():
    import jax
    import jax.numpy as jnp
    from mprl_common import AutoReset, make_env

    env, params = make_env("Pendulum-v1")  # never terminates: every end is a time limit
    w = AutoReset(env, params)
    obs, state = w.reset(jax.random.key(0))
    step = jax.jit(w.step)
    for t in range(params.max_steps_in_episode):
        out = step(jax.random.key(t + 1), state, jnp.zeros((1,)))
        state = out.state
    assert bool(out.truncated) and not bool(out.terminated)
    assert not np.allclose(out.final_obs, out.obs)  # pre-reset obs differs from the new episode's
    assert int(out.episode_length) == params.max_steps_in_episode

    env, params = make_env("CartPole-v1")  # falls over long before the 500-step limit
    w = AutoReset(env, params)
    obs, state = w.reset(jax.random.key(0))
    step = jax.jit(w.step)
    for t in range(200):
        out = step(jax.random.key(t), state, jnp.int32(1))  # always push right
        state = out.state
        if bool(out.terminated):
            break
    assert bool(out.terminated) and not bool(out.truncated)


def test_evaluate_uses_identical_instances():
    import jax
    import jax.numpy as jnp
    from mprl_common import AutoReset, evaluate, make_env
    env, params = make_env("CartPole-v1")
    w = AutoReset(env, params)
    act = lambda p, obs, k: jnp.zeros(obs.shape[0], jnp.int32)  # noqa: E731
    a = evaluate(w, act, None, 16, eval_seed=7)
    b = evaluate(w, act, None, 16, eval_seed=7)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_throughput_bench_runs():
    out = run("throughput_bench.py", "--n", "16", "32", "--steps", "8", "--reps", "1")
    assert out.returncode == 0, out.stderr
    assert "steps/s" in out.stdout


# ---------------------------------------------------------------- scaffolds

def test_ppo_learns_cartpole(tmp_path):
    out = run("ppo_jax.py", "--env", "CartPole-v1", "--total-timesteps", "500000", "--num-seeds", "2",
              "--results-csv", tmp_path / "res.csv", "--eval-csv", tmp_path / "ev.csv")
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path / "res.csv")
    assert len(df) == 2 and df["score"].min() > 400, out.stdout
    assert {"flops", "fpf", "batch_size", "lr", "seed", "score", "model"} <= set(df.columns)
    assert sum(1 for _ in open(tmp_path / "ev.csv")) == 3  # header + 2 agents


def test_a2c_mode_sets_single_pass():
    out = run("ppo_jax.py", "--algo", "a2c", "--env", "CartPole-v1", "--total-timesteps", "20000",
              "--num-envs", "16", "--num-steps", "16")
    assert out.returncode == 0, out.stderr
    assert "minibatches=1 epochs=1" in out.stdout


def test_minibatch_count_scales_with_envs():
    small = run("ppo_jax.py", "--total-timesteps", "20000", "--num-envs", "16", "--num-steps", "64",
                "--minibatch-size", "256")
    big = run("ppo_jax.py", "--total-timesteps", "40000", "--num-envs", "64", "--num-steps", "64",
              "--minibatch-size", "256")
    assert "minibatches=4" in small.stdout and "minibatches=16" in big.stdout


def test_pqn_learns_cartpole(tmp_path):
    # Single seeds vary at this budget: see "Reference runs" in references/debugging.md.
    # So this checks the typical seed, not every seed.
    out = run("pqn_jax.py", "--env", "CartPole-v1", "--total-timesteps", "1000000", "--num-seeds", "8",
              "--results-csv", tmp_path / "res.csv")
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path / "res.csv")
    assert df["score"].median() > 400, out.stdout


def test_elo_warns_on_duplicate_instances(tmp_path):
    path = tmp_path / "dup.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", *[f"ep{i}" for i in range(100)]])
        w.writerow(["random", *([0.0, 1.0] * 50)])
        w.writerow(["a", *([2.0, 3.0] * 50)])
    out = run("elo_league.py", path, "--reference", "random")
    assert "only 2 of 100 evaluation instances differ" in out.stderr


def test_minatar_eval_diversity_and_sticky_actions():
    from mprl_common import AutoReset, instance_diversity, make_env
    env, params = make_env("Breakout-MinAtar")
    assert instance_diversity(AutoReset(env, params), 64, 0) == (2, False)
    assert instance_diversity(AutoReset(env, params, sticky_prob=0.1), 64, 0)[1] is True
