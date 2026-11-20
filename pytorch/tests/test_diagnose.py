"""diagnose.py classifies each planted-bug model in examples/models/ correctly, and passes the healthy one."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def run(skill_root, command, model):
    proc = subprocess.run(
        [sys.executable, str(skill_root / "scripts" / "diagnose.py"), command,
         str(skill_root / "examples" / "models" / f"{model}.py"), "--json"],
        capture_output=True, text=True, timeout=300)
    return proc.returncode, json.loads(proc.stdout)


def flags(report):
    return {row["name"]: row["flag"] for row in report["grads"]["params"]}


def test_healthy_model_passes_every_check(skill_root):
    code, _ = run(skill_root, "all", "healthy")
    assert code == 0


def test_healthy_model_memorizes_one_batch(skill_root):
    _, r = run(skill_root, "overfit", "healthy")
    assert r["overfit"]["verdict"] == "memorized"


@pytest.mark.parametrize("model", ["plain_list", "detached", "softmax_head"])
def test_buggy_model_fails_to_memorize_one_batch(skill_root, model):
    _, r = run(skill_root, "overfit", model)
    assert r["overfit"]["verdict"] == "stuck"


def test_plain_list_of_layers_is_reported_as_hidden(skill_root):
    _, r = run(skill_root, "grads", "plain_list")
    assert r["grads"]["hidden"][0].startswith("hidden: modules in a plain list")


def test_detached_encoder_has_no_gradient(skill_root):
    _, r = run(skill_root, "grads", "detached")
    assert flags(r)["encoder.0.weight"] == "NONE"


def test_detached_model_head_still_gets_gradient(skill_root):
    _, r = run(skill_root, "grads", "detached")
    assert flags(r)["head.weight"] == "ok"


def test_softmax_head_is_named_in_a_warning(skill_root):
    _, r = run(skill_root, "grads", "softmax_head")
    assert "applies softmax" in r["grads"]["warnings"][0]


def test_nan_is_traced_to_the_module_that_produced_it(skill_root):
    _, r = run(skill_root, "nan", "nan_log")
    assert r["nan"]["first_nonfinite_module"] == "1 (LogFeatures)"


def test_nan_model_overfit_reports_divergence(skill_root):
    _, r = run(skill_root, "overfit", "nan_log")
    assert r["overfit"]["verdict"] == "diverged"


def test_buggy_model_exits_nonzero(skill_root):
    code, _ = run(skill_root, "grads", "detached")
    assert code == 1


def test_hidden_state_finds_a_plain_tensor_attribute(skill_root):
    import importlib.util

    import torch
    import torch.nn as nn

    spec = importlib.util.spec_from_file_location("diag", skill_root / "scripts" / "diagnose.py")
    assert spec is not None and spec.loader is not None
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.scale = torch.ones(2)          # not a Parameter, not a buffer

    assert diag.hidden_state(Scaled())[0].startswith("scale: plain tensor attribute (2,)")


LABEL_SMOOTHING = """import torch, torch.nn as nn
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
def make_model(): return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
def make_batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(32, 20, generator=g), torch.randint(0, 10, (32,), generator=g)
"""
MEAN_ONLY = """import torch, torch.nn as nn
loss_fn = nn.MSELoss()
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)
    def forward(self, x):
        return self.head(self.body(x).detach()).squeeze(-1)
def make_model(): return M()
def make_batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 8, generator=g), 6.0 + torch.randn(64, generator=g)
"""


def run_source(skill_root, tmp_path, source):
    path = tmp_path / "m.py"
    path.write_text(source)
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "diagnose.py"), "overfit",
                           str(path), "--json"], capture_output=True, text=True, timeout=300)
    return json.loads(proc.stdout)["overfit"]


def test_label_smoothing_floor_is_not_mistaken_for_stuck(skill_root, tmp_path):
    assert run_source(skill_root, tmp_path, LABEL_SMOOTHING)["verdict"] == "memorized"


def test_label_smoothing_floor_is_computed(skill_root, tmp_path):
    assert run_source(skill_root, tmp_path, LABEL_SMOOTHING)["floor"] == pytest.approx(0.5003, abs=1e-4)


def test_model_that_only_learns_the_mean_is_not_memorized(skill_root, tmp_path):
    assert run_source(skill_root, tmp_path, MEAN_ONLY)["verdict"] == "stuck"


# --------------------------------------------------------------------------- #
# In-process checks of the overfit arithmetic (fast, and each one can fail on its own)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def diag(skill_root):
    import importlib.util

    spec = importlib.util.spec_from_file_location("diag", skill_root / "scripts" / "diagnose.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smoothed_entropy(c: int, eps: float) -> float:
    import math

    hi, lo = 1 - eps + eps / c, eps / c
    return -(hi * math.log(hi) + (c - 1) * lo * math.log(lo))


def test_loss_floor_reads_classes_from_dim_1_for_k_dimensional_logits(diag):
    import torch
    import torch.nn as nn

    out = torch.zeros(4, 200, 2)                     # (N, C, d1): CrossEntropyLoss's layout
    y = torch.zeros(4, 2, dtype=torch.long)
    floor = diag.loss_floor(nn.CrossEntropyLoss(label_smoothing=0.1), out, y)
    assert floor == pytest.approx(_smoothed_entropy(200, 0.1), abs=1e-6)


def test_loss_floor_for_plain_logits_uses_the_last_dim(diag):
    import torch
    import torch.nn as nn

    floor = diag.loss_floor(nn.CrossEntropyLoss(label_smoothing=0.1),
                            torch.zeros(32, 10), torch.zeros(32, dtype=torch.long))
    assert floor == pytest.approx(_smoothed_entropy(10, 0.1), abs=1e-6)


def test_loss_floor_for_unbatched_logits_uses_dim_0(diag):
    import torch
    import torch.nn as nn

    floor = diag.loss_floor(nn.CrossEntropyLoss(label_smoothing=0.1),
                            torch.zeros(10), torch.tensor(3))
    assert floor == pytest.approx(_smoothed_entropy(10, 0.1), abs=1e-6)


@pytest.mark.parametrize("case", ["no-smoothing", "soft-targets", "mse"])
def test_loss_floor_is_zero_when_no_floor_is_known(diag, case):
    import torch
    import torch.nn as nn

    out = torch.zeros(4, 10)
    loss_fn, y = {
        "no-smoothing": (nn.CrossEntropyLoss(), torch.zeros(4, dtype=torch.long)),
        "soft-targets": (nn.CrossEntropyLoss(label_smoothing=0.1), torch.full((4, 10), 0.1)),
        "mse": (nn.MSELoss(), torch.zeros(4, 10)),
    }[case]
    floor = diag.loss_floor(loss_fn, out, y)
    assert floor == 0.0 and isinstance(floor, float)


def test_overfit_with_no_trainable_parameter_is_stuck_without_training(diag):
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 3).requires_grad_(False)
    batch = (torch.randn(8, 4), torch.randint(0, 3, (8,)))
    report = diag.overfit(model, batch, nn.CrossEntropyLoss(), steps=5, lr=1e-3)
    assert (report["verdict"], report["reason"]) == ("stuck", "no parameter requires grad")


# judge(first, last, floor, baseline) -> verdict, at the exact thresholds:
# near the floor means last - floor < 0.1 * (first - floor) or < 1e-2;
# beating the baseline means last < baseline - 0.5 * (baseline - floor).
@pytest.mark.parametrize("first, last, floor, baseline, verdict", [
    (2.0, 0.05, 0.0, None, "memorized"),        # below 10% of the drop and no baseline
    (2.0, 0.2, 0.0, None, "stuck"),             # exactly 10%: not strictly below
    (2.0, 0.19, 0.0, None, "memorized"),        # just under 10%
    (0.05, 0.0099, 0.0, None, "memorized"),     # absolute 1e-2 rule
    (0.05, 0.01, 0.0, None, "stuck"),           # exactly 1e-2: not strictly below
    (1.05, 1.009, 1.0, None, "memorized"),      # 1e-2 rule measured from the floor
    (3.0, 1.19, 1.0, None, "memorized"),        # floor subtracted: 0.19 < 0.1 * (3.0 - 1.0)
    (3.0, 1.3, 1.0, None, "stuck"),             # 0.3 is not within 10% of a 2.0 gap
    (2.0, 0.05, 0.0, 1.0, "memorized"),         # beats baseline: 0.05 < 1.0 - 0.5
    (2.0, 0.009, 0.0, 0.018, "stuck"),          # near floor but 0.009 is not < 0.018 - 0.009
    (2.0, 0.0089, 0.0, 0.018, "memorized"),     # just beats it
    (2.0, 1.05, 1.0, 1.1, "stuck"),             # 1.05 is not < 1.1 - 0.05
    (2.0, 1.04, 1.0, 1.1, "memorized"),         # floor-relative baseline margin
])
def test_judge_applies_the_documented_thresholds(diag, first, last, floor, baseline, verdict):
    assert diag.judge(first, last, floor, baseline) == verdict
