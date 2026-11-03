import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_analyze_sweep_reproduces_reference_table(capsys):
    out = _load("analyze_sweep").main([])
    assert out["standing_still"] == pytest.approx(0.5)
    rows = {r["num_envs"]: r for r in out["rows"]}
    assert rows[4096]["45 min"].startswith("0.082")
    assert rows[16384]["24 min"].startswith("0.081")
    assert rows[64]["45 min"].startswith("0.503")
    ns, M = out["pairwise"]
    assert M[ns.index(4096), ns.index(1024)] == 1.0


def test_exploration_chain_matches_theory(capsys):
    rows = _load("exploration_chain").main(["--A", "3", "--H", "2", "5", "--seeds", "30"])
    for r in rows:
        assert r["p_goal"] == pytest.approx(3.0 ** -r["H"])
        assert abs(r["reset_steps"] - r["H"] * 2) < 0.6 * r["H"]  # H(A+1)/2 = 2H for A = 3


def test_ppo_pendulum_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    mod = _load("ppo_pendulum")
    args = mod.parse(["--steps", "4096", "--num-envs", "2", "--rollout", "256", "--epochs", "2", "--minibatches", "4"])
    res = mod.train(args)
    assert np.isfinite(res["eval_return"])
