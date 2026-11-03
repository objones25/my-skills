"""Tests for the skill's scripts. Run from the skill root: python -m pytest tests -q

The Inspect smoke test is skipped when inspect_ai is not installed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))

import reliability as rel  # noqa: E402
import tool_budget as tb  # noqa: E402
import tool_lint  # noqa: E402
import rule_of_two_audit as r2  # noqa: E402
import triage_transcripts as tri  # noqa: E402
import cache_prefix_check as cpc  # noqa: E402
import agent_loop as al  # noqa: E402


# ---------------------------------------------------------------- reliability
class TestReliability:
    def test_pass_at_k_matches_closed_form(self):
        # 1 - C(3,2)/C(8,2) = 1 - 3/28
        assert rel.pass_at_k(8, 5, 2) == pytest.approx(1 - 3 / 28)

    def test_pass_hat_k_matches_closed_form(self):
        # C(7,4)/C(8,4) = 35/70
        assert rel.pass_hat_k(8, 7, 4) == pytest.approx(0.5)

    def test_k1_both_equal_raw_rate(self):
        assert rel.pass_at_k(10, 6, 1) == pytest.approx(0.6)
        assert rel.pass_hat_k(10, 6, 1) == pytest.approx(0.6)

    def test_pass_hat_k_never_exceeds_pass_at_k(self):
        for n in range(1, 9):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    assert rel.pass_hat_k(n, c, k) <= rel.pass_at_k(n, c, k) + 1e-12

    def test_rejects_k_above_n(self):
        with pytest.raises(ValueError, match="run more trials"):
            rel.pass_hat_k(3, 3, 5)

    def test_matches_inspect_estimators(self):
        """Same formulas Inspect's pass_at / pass_k reducers use (read from its source)."""
        n, c, k = 6, 4, 2
        inspect_pass_at = 1.0 - math.prod(1.0 - k / x for x in range(n - c + 1, n + 1))
        assert rel.pass_at_k(n, c, k) == pytest.approx(inspect_pass_at)
        assert rel.pass_hat_k(n, c, k) == pytest.approx(math.comb(c, k) / math.comb(n, k))

    def test_compare_detects_clear_gain_and_ignores_noise(self):
        old = {f"t{i}": (5, 2) for i in range(30)}
        new = {f"t{i}": (5, 5) for i in range(30)}
        assert rel.compare(old, new).distinguishable
        noisy_new = {f"t{i}": (5, 3 if i % 2 else 1) for i in range(30)}
        assert not rel.compare(old, noisy_new).distinguishable

    def test_budget(self):
        assert rel.run_budget(40, 5, 2, 0.30) == {"runs": 400, "cost": 120.0}

    def test_per_trial_rate(self):
        assert rel.per_trial_rate_needed(0.9, 8) ** 8 == pytest.approx(0.9)


# ---------------------------------------------------------------- tool budget
class TestToolBudget:
    TEXT = "\n".join(f"line {i} " + "x" * 36 for i in range(1000))

    def test_small_text_untouched(self):
        b = tb.budget_text("a\nb", max_tokens=100)
        assert b.text == "a\nb" and not b.truncated and b.next_offset is None

    def test_truncation_is_visible_and_recoverable(self):
        b = tb.budget_text(self.TEXT, max_tokens=200)
        assert b.truncated and b.next_offset is not None
        assert f"offset={b.next_offset}" in b.text.splitlines()[-1]
        assert "of 1,000" in b.text

    def test_paging_covers_everything_once(self):
        seen, off = [], 0
        while off is not None:
            b = tb.budget_text(self.TEXT, offset=off, max_tokens=500)
            seen += [l for l in b.text.splitlines() if l.startswith("line ")]
            off = b.next_offset
        assert len(seen) == 1000 and len(set(seen)) == 1000

    def test_records_drop_plumbing_fields(self):
        recs = [{"name": f"f{i}", "uuid": "9f2c-...", "mime_type": "x"} for i in range(50)]
        b = tb.budget_records(recs, fields=["name"], limit=10)
        assert "uuid" not in b.text and b.next_offset == 10
        assert "of 50" in b.text


# ---------------------------------------------------------------- tool lint
class TestToolLint:
    def test_flags_overlap_boundary_vague_and_unmarked_write(self):
        tools = json.loads((FIX / "tools_overlapping.json").read_text())
        findings, stats = tool_lint.lint(tools)
        checks = {(f.check, tuple(f.tools)) for f in findings}
        assert ("overlap", ("search_customers", "find_customer")) in checks
        assert ("boundary", ("find_customer",)) in checks
        assert ("vague-param", ("find_customer",)) in checks
        assert ("write-unmarked", ("delete_customer",)) in checks
        assert not any(f.tools == ["get_customer_context"] and f.check == "thin-desc" for f in findings)
        assert stats["tools"] == 4

    def test_catalog_size_warning(self):
        tools = [{"name": f"tool_{i}", "description": "Does one specific distinct thing number " + str(i) * 5,
                  "input_schema": {}} for i in range(12)]
        findings, _ = tool_lint.lint(tools, overlap=0.99)
        assert any(f.check == "catalog-size" for f in findings)

    def test_accepts_openai_and_mcp_shapes(self):
        tools = [{"type": "function", "function": {"name": "a_b", "description": "x",
                                                   "parameters": {"properties": {"id": {}}}}},
                 {"name": "c_d", "description": "y", "inputSchema": {"properties": {}}}]
        findings, _ = tool_lint.lint(tools)
        assert any(f.check == "vague-param" and f.tools == ["a_b"] for f in findings)


# ---------------------------------------------------------------- rule of two
class TestRuleOfTwo:
    def reports(self):
        return {r.name: r for r in r2.audit(json.loads((FIX / "email_agent_manifest.json").read_text()))}

    def test_single_agent_violates(self):
        assert self.reports()["single_agent"].violation

    def test_human_gate_on_c_clears_it(self):
        assert not self.reports()["gated_agent"].violation

    def test_free_text_handoff_inherits_untrusted_input(self):
        r = self.reports()
        assert r["sender_free_text"].violation and r["sender_free_text"].inherited_a_from == ["reader"]

    def test_structured_handoff_breaks_the_chain(self):
        assert not self.reports()["sender_structured"].violation

    def test_unknown_tool_is_an_error(self):
        with pytest.raises(ValueError):
            r2.audit({"tools": {}, "sessions": {"s": {"tools": ["ghost"]}}})


# ---------------------------------------------------------------- triage
class TestTriage:
    def test_buckets(self):
        runs = [json.loads(l) for l in (FIX / "false_success_runs.jsonl").read_text().splitlines()]
        labels = {r["run_id"]: tri.classify(r) for r in runs}
        assert labels == {"r1": "never_called", "r2": "silent_failure", "r3": "wrong_id",
                          "r4": "partial_retry", "r5": "empty_result", "r6": "verified"}


# ---------------------------------------------------------------- cache prefix
class TestCachePrefix:
    def test_flags_timestamp_and_tool_reorder(self):
        reqs = [json.loads(l) for l in (FIX / "requests_cache_break.jsonl").read_text().splitlines()]
        rows = cpc.check(reqs)
        assert rows[0]["ok"]
        assert not rows[1]["ok"] and "timestamp" in rows[1]["likely_cause"]
        assert not rows[2]["ok"] and "tool order" in rows[2]["likely_cause"]

    def test_append_only_passes(self):
        base = {"system": "s", "messages": [{"role": "user", "content": "q"}]}
        nxt = {"system": "s", "messages": base["messages"] + [{"role": "assistant", "content": "a"}]}
        assert cpc.check([base, nxt])[0]["ok"]


# ---------------------------------------------------------------- agent loop
def _order_tools(calls_seen: list | None = None):
    def get_order(order_id: str) -> dict:
        if calls_seen is not None:
            calls_seen.append(order_id)
        return {"A1": {"status": "shipped"}}.get(order_id, {"found": False})

    schema = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    return [al.Tool("get_order", "Look up an order by id.", schema, get_order),
            al.Tool("refund", "Refund an order.", schema, lambda order_id: "refunded", needs_approval=True)]


T = al.ModelTurn
C = al.ToolCall


class TestAgentLoop:
    def test_verified_done(self, tmp_path):
        m = al.ScriptedModel([T("", [C("c1", "get_order", {"order_id": "A1"})]),
                              T("", [C("c2", "finish", {"status": "done", "summary": "shipped",
                                                        "evidence": ["c1"]})])])
        out = al.Agent(m, _order_tools(), "s", verify=al.evidence_exists).run("A1?", tmp_path / "l.jsonl")
        assert out.status == "done" and out.steps == 2

    def test_unsupported_done_is_rejected_then_sanctioned_failure_accepted(self, tmp_path):
        m = al.ScriptedModel([
            T("", [C("c1", "get_order", {"order_id": "Z9"})]),
            T("", [C("c2", "finish", {"status": "done", "summary": "shipped yesterday"})]),
            T("", [C("c3", "finish", {"status": "could_not_complete", "summary": "no such order"})]),
        ])
        out = al.Agent(m, _order_tools(), "s", verify=al.evidence_exists).run("Z9?", tmp_path / "l.jsonl")
        assert out.status == "could_not_complete"
        kinds = [e["kind"] for e in al.EventLog(tmp_path / "l.jsonl").events()]
        assert "verify_failed" in kinds

    def test_repeated_false_done_ends_in_verify_failed(self, tmp_path):
        bad = lambda i: T("", [C(f"f{i}", "finish", {"status": "done", "summary": "ok"})])
        m = al.ScriptedModel([bad(i) for i in range(5)])
        out = al.Agent(m, _order_tools(), "s", verify=al.evidence_exists,
                       max_verify_retries=2).run("x", tmp_path / "l.jsonl")
        assert out.status == "verify_failed"

    def test_step_limit(self, tmp_path):
        m = al.ScriptedModel([T("", [C(f"c{i}", "get_order", {"order_id": "A1"})]) for i in range(10)])
        out = al.Agent(m, _order_tools(), "s", max_steps=3).run("loop", tmp_path / "l.jsonl")
        assert out.status == "step_limit" and out.steps == 3

    def test_approval_denied_by_default(self, tmp_path):
        m = al.ScriptedModel([T("", [C("c1", "refund", {"order_id": "A1"})]),
                              T("", [C("c2", "finish", {"status": "could_not_complete", "summary": "needs human"})])])
        al.Agent(m, _order_tools(), "s").run("refund A1", tmp_path / "l.jsonl")
        res = [e for e in al.EventLog(tmp_path / "l.jsonl").events() if e["kind"] == "tool_result"][0]
        assert res["is_error"] and "approval" in res["content"]

    def test_bad_arguments_are_instructive(self, tmp_path):
        m = al.ScriptedModel([T("", [C("c1", "get_order", {"id": "A1"})]),
                              T("", [C("c2", "finish", {"status": "could_not_complete", "summary": "x"})])])
        al.Agent(m, _order_tools(), "s").run("x", tmp_path / "l.jsonl")
        res = [e for e in al.EventLog(tmp_path / "l.jsonl").events() if e["kind"] == "tool_result"][0]
        assert res["is_error"] and "Schema" in res["content"]

    def test_resume_after_crash_does_not_redo_finished_calls(self, tmp_path):
        log = tmp_path / "l.jsonl"
        seen: list = []

        class Crash(Exception):
            pass

        def crashing(messages, tools):
            if len(messages) >= 3:   # after the first tool result
                raise Crash()
            return T("", [C("c1", "get_order", {"order_id": "A1"})])

        with pytest.raises(Crash):
            al.Agent(crashing, _order_tools(seen), "s").run("A1?", log)
        assert seen == ["A1"]
        m = al.ScriptedModel([T("", [C("c2", "finish", {"status": "done", "summary": "shipped",
                                                        "evidence": ["c1"]})])])
        out = al.Agent(m, _order_tools(seen), "s", verify=al.evidence_exists).resume(log)
        assert out.status == "done" and seen == ["A1"]   # tool not re-run
        assert any(msg["role"] == "tool" for msg in m.seen[0])  # context rebuilt from the log

    def test_resume_answers_dangling_call(self, tmp_path):
        log = al.EventLog(tmp_path / "l.jsonl")
        log.append("task", task="A1?", system="s")
        log.append("assistant", text="", calls=[{"id": "c1", "name": "get_order", "args": {"order_id": "A1"}}])
        seen: list = []
        m = al.ScriptedModel([T("", [C("c2", "finish", {"status": "done", "summary": "ok", "evidence": ["c1"]})])])
        out = al.Agent(m, _order_tools(seen), "s", verify=al.evidence_exists).resume(log.path)
        assert seen == ["A1"] and out.status == "done"


# ---------------------------------------------------------------- inspect
def test_inspect_template_smoke():
    pytest.importorskip("inspect_ai")
    import inspect_eval_template as tmpl
    log = tmpl._smoke()
    assert log.status == "success"
    by_reducer = {s.reducer: s.metrics["accuracy"].value for s in log.results.scores}
    # 2 of 3 samples correct; the fabricated answer for the missing order fails
    assert by_reducer["pass_at_1"] == pytest.approx(2 / 3)
    assert by_reducer["pass_k_2"] == pytest.approx(2 / 3)
