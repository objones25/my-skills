"""spike_monitor.py: the OLMo 2 spike rule, growth, and attention-logit flags."""
from __future__ import annotations

import json
import subprocess
import sys


def test_demo_log_spikes_found_at_planted_steps(spikes, skill_root):
    cols = spikes.load(skill_root / "assets" / "example_training_log.csv")
    r = spikes.analyse(cols)
    assert r["series"]["grad_norm"]["spike_steps"] == [3500, 3501, 3502, 3503]
    assert r["series"]["loss"]["spike_steps"] == [4300]


def test_early_spike_needs_a_shorter_window(spikes, skill_root):
    cols = spikes.load(skill_root / "assets" / "example_training_log.csv")
    assert 2100 not in spikes.analyse(cols)["series"]["loss"]["spike_steps"]
    assert 2100 in spikes.analyse(cols, window=200)["series"]["loss"]["spike_steps"]


def test_constant_series_has_no_spikes(spikes):
    assert spikes.spikes([1.0] * 500) == []


def test_single_outlier_in_noise_is_found(spikes):
    import random
    rng = random.Random(1)
    series = [rng.gauss(0, 1) for _ in range(300)]
    series[250] = 20.0
    assert spikes.spikes(series) == [250]


def test_growth_factor_of_exponential(spikes):
    steps = [float(s) for s in range(0, 20000, 10)]
    values = [2 ** (s / 10000) for s in steps]
    assert abs(spikes.log_slope_per_10k(steps, values) - 2.0) < 1e-6


def test_attention_logit_flag(spikes):
    r = spikes.analyse({"step": [0.0, 1.0, 2.0], "max_attn_logit": [10.0, 2e4, 3e4]})
    assert r["series"]["max_attn_logit"]["steps_over_1e4"] == [1, 2]


def test_jsonl_input_and_json_output(spikes, skill_root, tmp_path):
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps({"step": i, "loss": 3.0 - i * 1e-3}) for i in range(100)))
    out = subprocess.run([sys.executable, str(skill_root / "scripts" / "spike_monitor.py"), str(log), "--json"],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["series"]["loss"]["spike_score_pct"] == 0.0
