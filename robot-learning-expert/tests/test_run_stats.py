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
