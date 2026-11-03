"""reference_block.py: the property tests pass, and each one can fail on its bug."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import textwrap

import pytest

DETECTORS = ["rope-split-order", "rmsnorm-eps", "cache-causal-mask",
             "cache-position-restart", "pad-equals-eos", "gqa-repeat"]


def test_verify_passes_on_the_reference_implementation(block):
    with contextlib.redirect_stdout(io.StringIO()):
        assert block.verify() == 0


@pytest.mark.parametrize("name", DETECTORS)
def test_bug_detector_catches_its_bug(block, name):
    caught, _detail = block.BUG_DETECTORS[name]()
    assert caught


def test_qk_norm_parameter_cost_matches_config_budget(block, budget):
    extra = (sum(p.numel() for p in block.Attention(block.Cfg(qk_norm=True)).parameters())
             - sum(p.numel() for p in block.Attention(block.Cfg()).parameters()))
    c = budget.Config(d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab=128, d_ff=176,
                      qk_norm=True)
    plain = budget.Config(d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab=128, d_ff=176)
    budget_extra = (budget.param_breakdown(c)["attn_per_layer"]
                    - budget.param_breakdown(plain)["attn_per_layer"])
    assert extra == budget_extra == 2 * 16


def run_verify_with_impl(skill_root, tmp_path, source):
    impl = tmp_path / "impl.py"
    impl.write_text(textwrap.dedent(source))
    return subprocess.run([sys.executable, str(skill_root / "scripts" / "reference_block.py"),
                           "--verify", "--impl", str(impl)],
                          capture_output=True, text=True, timeout=300)


def test_impl_hook_passes_a_correct_user_repeat_kv(skill_root, tmp_path):
    proc = run_verify_with_impl(skill_root, tmp_path, """
        def repeat_kv(x, n_rep):
            return x.repeat_interleave(n_rep, dim=1)
    """)
    assert proc.returncode == 0, proc.stdout[-2000:]


def test_impl_hook_fails_a_buggy_user_repeat_kv(skill_root, tmp_path):
    proc = run_verify_with_impl(skill_root, tmp_path, """
        def repeat_kv(x, n_rep):
            return x.repeat(1, n_rep, 1, 1)
    """)
    assert proc.returncode != 0


def test_impl_hook_fails_eps_added_to_the_sqrt_not_inside_it(skill_root, tmp_path):
    proc = run_verify_with_impl(skill_root, tmp_path, """
        import torch
        import torch.nn as nn

        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

            def forward(self, x):
                return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps) * self.weight
    """)
    assert proc.returncode != 0
    assert "[FAIL] ...and gets a finite gradient" in proc.stdout


def test_shipped_buggy_port_fails_exactly_two_checks(skill_root):
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "reference_block.py"), "--verify",
                           "--impl", str(skill_root / "examples" / "buggy_port.py")],
                          capture_output=True, text=True, timeout=300)
    assert proc.stdout.count("[FAIL]") == 2
