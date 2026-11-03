"""memory_budget.py: the arithmetic matches the hand calculation, and --measure agrees with PyTorch."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def run(skill_root, *args):
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "memory_budget.py"), *args,
                           "--json"], capture_output=True, text=True, timeout=300, check=True)
    return json.loads(proc.stdout)


def test_adamw_fp32_is_sixteen_bytes_per_param(skill_root):
    r = run(skill_root, "--params", "7e9", "--optimizer", "adamw")
    assert sum(r["bytes_per_param"].values()) == 16


def test_7b_adamw_ddp_matches_the_reference_walkthrough(skill_root):
    r = run(skill_root, "--params", "7e9", "--optimizer", "adamw")
    assert round(r["total_gib"]) == 104


def test_fsdp_divides_by_world_size(skill_root):
    r = run(skill_root, "--params", "7e9", "--optimizer", "adamw", "--world", "8")
    assert r["total_gib"] == pytest.approx(104.308 / 8, abs=1e-3)


def test_7b_adamw_does_not_fit_one_80gib_device(skill_root):
    assert run(skill_root, "--params", "7e9")["fits"] is False


def test_bf16_master_layout_is_also_sixteen_bytes(skill_root):
    r = run(skill_root, "--params", "1e9", "--precision", "bf16-master")
    assert sum(r["bytes_per_param"].values()) == 16


@pytest.mark.parametrize("optimizer", ["adamw", "adam", "sgd-momentum", "muon", "adafactor"])
def test_formula_matches_measured_optimizer_state(skill_root, optimizer):
    r = run(skill_root, "--model", str(skill_root / "examples" / "models" / "healthy.py"),
            "--optimizer", optimizer, "--measure")
    assert r["measured_optimizer_bytes_fp32"] == pytest.approx(r["predicted_optimizer_bytes_fp32"])


def test_muon_without_a_model_is_refused(skill_root):
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "memory_budget.py"),
                           "--params", "1e9", "--optimizer", "muon"], capture_output=True, text=True)
    assert "needs --model" in proc.stderr


EDGE_MODELS = {
    "conv": "nn.Sequential(nn.Conv2d(3, 16, 3), nn.Flatten(), nn.Linear(16, 10))",
    "one-element-bias": "nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))",
    "frozen-first-layer": "_frozen()",
    "embedding-and-head": "nn.Sequential(nn.Embedding(1000, 64), nn.Linear(64, 1000))",
}
EDGE_SOURCE = """import torch.nn as nn
def _frozen():
    m = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    m[0].requires_grad_(False)
    return m
def make_model():
    return {expr}
"""


@pytest.mark.parametrize("optimizer", ["adamw", "adafactor", "muon"])
@pytest.mark.parametrize("model", sorted(EDGE_MODELS))
def test_formula_matches_measurement_on_edge_case_models(skill_root, tmp_path, model, optimizer):
    path = tmp_path / "m.py"
    path.write_text(EDGE_SOURCE.format(expr=EDGE_MODELS[model]))
    r = run(skill_root, "--model", str(path), "--optimizer", optimizer, "--measure")
    assert r["measured_optimizer_bytes_fp32"] == pytest.approx(r["predicted_optimizer_bytes_fp32"])


def test_frozen_parameters_get_no_gradient_bytes(skill_root, tmp_path):
    path = tmp_path / "m.py"
    path.write_text(EDGE_SOURCE.format(expr="_frozen()"))
    r = run(skill_root, "--model", str(path))
    assert r["trainable"] == 650
