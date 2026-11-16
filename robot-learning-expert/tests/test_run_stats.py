import numpy as np
import pytest

import run_stats as rs


@pytest.mark.parametrize("n", [4, 5, 10, 11, 37])
def test_iqm_matches_scipy_trim_mean(n):
    stats = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(n).normal(size=n)
    assert rs.iqm(x) == pytest.approx(stats.trim_mean(x, 0.25))


def test_ci_contains_point_and_shrinks_with_runs():
    rng = np.random.default_rng(0)
    small = {"t": rng.normal(size=5)}
    big = {"t": rng.normal(size=200)}
    p1, lo1, hi1 = rs.stratified_bootstrap_ci(small, reps=500)
    p2, lo2, hi2 = rs.stratified_bootstrap_ci(big, reps=500)
    assert lo1 <= p1 <= hi1 and lo2 <= p2 <= hi2
    assert hi2 - lo2 < hi1 - lo1


def test_prob_improvement_extremes():
    x = {"a": np.array([2.0, 3.0]), "b": np.array([5.0])}
    y = {"a": np.array([0.0, 1.0]), "b": np.array([5.0])}
    assert rs.prob_improvement(x, y) == pytest.approx((1.0 + 0.5) / 2)


def test_csv_roundtrip(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("method,seed,score\nA,0,1\nA,1,2\nB,0,0\n")
    d = rs.load_csv(str(p))
    assert d["A"]["all"] == [1.0, 2.0] and d["B"]["all"] == [0.0]


def test_prob_improvement_ci_is_one_when_x_always_wins():
    x = {"a": np.array([5.0, 6.0, 7.0]), "b": np.array([9.0, 10.0])}
    y = {"a": np.array([0.0, 1.0, 2.0]), "b": np.array([3.0, 4.0])}
    assert rs.prob_improvement_ci(x, y, reps=200) == (1.0, 1.0, 1.0)


def test_prob_improvement_ci_brackets_point_and_is_seeded():
    rng = np.random.default_rng(1)
    x, y = {"t": rng.normal(0.3, 1, 20)}, {"t": rng.normal(0.0, 1, 20)}
    p, lo, hi = rs.prob_improvement_ci(x, y, reps=500, seed=7)
    assert p == pytest.approx(rs.prob_improvement(x, y))
    assert 0.0 <= lo < p < hi <= 1.0
    assert rs.prob_improvement_ci(x, y, reps=500, seed=7) == (p, lo, hi)


def test_curve_band_mean_and_bounds():
    curves = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 9.0]])
    mean, lo, hi = rs.curve_band(curves, reps=500)
    assert np.allclose(mean, [2.0, 3.0, 5.0])
    assert mean.shape == lo.shape == hi.shape == (3,)
    assert np.all(lo <= mean) and np.all(mean <= hi) and np.all(lo < hi)
    # every bootstrap mean lies between the smallest and largest seed
    assert np.all(lo >= curves.min(axis=0)) and np.all(hi <= curves.max(axis=0))


def test_curve_band_collapses_when_seeds_agree():
    mean, lo, hi = rs.curve_band(np.tile([1.0, 2.0, 3.0], (4, 1)), reps=100)
    assert np.allclose(lo, mean) and np.allclose(hi, mean)


def _percentiles(boots, alpha):
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)


@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_curve_band_is_percentile_bootstrap_over_seeds(alpha):
    curves = np.random.default_rng(2).normal(size=(6, 4))
    rng = np.random.default_rng(11)
    means = np.stack([curves[rng.integers(6, size=6)].mean(axis=0) for _ in range(300)])
    lo_ref, hi_ref = _percentiles(means, alpha)
    _, lo, hi = rs.curve_band(curves, reps=300, alpha=alpha, seed=11)
    assert np.allclose(lo, lo_ref) and np.allclose(hi, hi_ref)


@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_prob_improvement_ci_is_percentile_bootstrap(alpha):
    x, y = np.random.default_rng(3).normal(0.2, 1, 9), np.random.default_rng(4).normal(0.0, 1, 7)
    rng = np.random.default_rng(5)
    boots = []
    for _ in range(300):
        xs, ys = x[rng.integers(9, size=9)], y[rng.integers(7, size=7)]
        boots.append(((xs[:, None] > ys[None, :]) + 0.5 * (xs[:, None] == ys[None, :])).mean())
    lo_ref, hi_ref = _percentiles(np.array(boots), alpha)
    _, lo, hi = rs.prob_improvement_ci({"t": x}, {"t": y}, reps=300, alpha=alpha, seed=5)
    assert lo == pytest.approx(lo_ref) and hi == pytest.approx(hi_ref)
