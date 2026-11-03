"""verify_api.py must be able to fail: it reports planted drift and passes a clean install."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def test_matching_signature_reports_no_drift(verify_api):
    path = "torch.nn.Flatten"
    assert verify_api.signature_drift(path, verify_api.actual_params(path)) is None


def test_added_parameter_is_reported(verify_api):
    path = "torch.nn.Flatten"
    trimmed = verify_api.actual_params(path)[:-1]
    assert "added ['end_dim']" in verify_api.signature_drift(path, trimmed)


def test_removed_parameter_is_reported(verify_api):
    path = "torch.nn.Flatten"
    padded = verify_api.actual_params(path) + ("ghost",)
    assert "removed ['ghost']" in verify_api.signature_drift(path, padded)


def test_every_pinned_signature_resolves(verify_api):
    unresolved = []
    for path in verify_api.SIGNATURES:
        try:
            verify_api.actual_params(path)
        except (ImportError, AttributeError) as exc:
            unresolved.append((path, type(exc).__name__))
    assert unresolved == []


def test_resolve_prefers_the_function_over_the_same_named_module(verify_api):
    assert callable(verify_api.resolve("torch.autograd.gradcheck"))


def test_resolve_reaches_a_submodule_that_is_not_an_attribute(verify_api):
    assert verify_api.resolve("torch.nn.attention.varlen").__name__ == "torch.nn.attention.varlen"


def test_signature_claim_with_a_wrong_default_is_reported(verify_api):
    args = "params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False"
    found = verify_api._check_claim("x.md", "torch.optim.AdamW", args)
    assert [f["detail"] for f in found] == ["weight_decay: doc says 0, installed has 0.01"]


def test_signature_claim_with_a_renamed_parameter_is_reported(verify_api):
    args = "model=None, *, fullgraph=False, dynamic=None, backend=None, shapes_spec=None"
    found = verify_api._check_claim("x.md", "torch.compile", args)
    assert [f["kind"] for f in found] == ["unknown-param"]


def test_usage_example_is_not_treated_as_a_signature(verify_api):
    assert verify_api._check_claim("x.md", "torch.optim.AdamW", "params, lr=3e-4") == []


def test_unresolvable_name_in_a_reference_is_reported(verify_api, tmp_path):
    (tmp_path / "r.md").write_text("Use `torch.nn.NoSuchLayer` here.\n")
    found = verify_api.scan_docs(tmp_path)
    assert [(f["kind"], f["name"]) for f in found] == [("unresolved", "torch.nn.NoSuchLayer")]


@pytest.mark.skipif(tuple(int(v) for v in __import__("torch").__version__.split(".")[:2]) < (2, 15),
                    reason="the references name 2.15-only APIs (torch.optim.functional, CuspyConfig)")
def test_shipped_references_have_no_doc_findings(verify_api, skill_root):
    assert verify_api.scan_docs(skill_root / "references") == []


def test_cli_json_reports_versions(skill_root):
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "verify_api.py"), "--json"],
                          capture_output=True, text=True, timeout=300)
    assert json.loads(proc.stdout)["reference"].startswith("torch 2.15.0")


def test_two_keyword_misquote_of_adamw_weight_decay_is_reported(verify_api):
    found = verify_api._check_claim("x.md", "torch.optim.AdamW", "params, lr=0.001, weight_decay=0")
    assert [f["detail"] for f in found] == ["weight_decay: doc says 0, installed has 0.01"]


def test_misspelled_keyword_is_reported_even_in_a_usage_example(verify_api):
    found = verify_api._check_claim("x.md", "torch.nn.Dropout", "p=0.3, inplce=False")
    assert [f["kind"] for f in found] == ["unknown-param"]


def test_callable_taking_kwargs_is_not_checked_for_unknown_keywords(verify_api):
    assert verify_api._check_claim("x.md", "torch.load", "f, weights_only=True, mmapp=None") == []


def test_positional_only_parameter_written_as_keyword_is_reported(verify_api):
    found = verify_api._check_claim("x.md", "torch.accelerator.max_memory_allocated", "device=None")
    assert [f["kind"] for f in found] == ["unknown-param"]


def test_missing_attribute_is_reported_as_attribute_error(verify_api, tmp_path):
    (tmp_path / "r.md").write_text("`torch.optim.NoSuchOptimizer`\n")
    assert verify_api.scan_docs(tmp_path)[0]["detail"] == "AttributeError"


@pytest.mark.skipif(__import__("importlib.util").util.find_spec("tensorboard") is not None,
                    reason="needs tensorboard to be absent")
def test_name_needing_an_absent_optional_package_is_skipped(verify_api, tmp_path):
    (tmp_path / "r.md").write_text("`torch.utils.tensorboard.SummaryWriter`\n")
    assert verify_api.scan_docs(tmp_path) == []
