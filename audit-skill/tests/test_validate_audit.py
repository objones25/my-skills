"""validate_audit.py: every mechanical rule a finished audit must meet, one fixture per code."""

from __future__ import annotations

import json

import pytest
from conftest import good_items, good_searches, read_jsonl, run_script, write_jsonl


def codes(run) -> set[str]:
    proc = run_script("validate_audit.py", run, "--json")
    assert proc.returncode in (0, 1), proc.stderr
    return {f["code"] for f in json.loads(proc.stdout)["findings"]}


def set_items(run, mutate) -> None:
    items = good_items()
    mutate(items)
    write_jsonl(run / "items.jsonl", items)


def test_complete_run_is_clean(complete_run):
    proc = run_script("validate_audit.py", complete_run)
    assert proc.returncode == 0, proc.stdout
    assert "0 finding" in proc.stdout


def test_missing_meta_exits_2(tmp_path):
    assert run_script("validate_audit.py", tmp_path).returncode == 2


def test_changed_skill_is_as002(complete_run):
    meta = json.loads((complete_run / "meta.json").read_text())
    with open(meta["skill_path"] + "/SKILL.md", "a", encoding="utf-8") as fh:
        fh.write("edited\n")
    assert "AS002" in codes(complete_run)


def test_candidates_not_run_is_as003(complete_run):
    (complete_run / "candidates.jsonl").unlink()
    assert "AS003" in codes(complete_run)


def test_uncovered_candidate_is_as010(complete_run):
    set_items(complete_run, lambda items: items.pop(2))
    assert "AS010" in codes(complete_run)


def test_dismissal_with_reason_covers_a_candidate(complete_run):
    set_items(complete_run, lambda items: items.pop(2))
    write_jsonl(complete_run / "dismissed.jsonl", [{"file": "SKILL.md", "lines": [9, 15], "reason": "style advice"}])
    assert "AS010" not in codes(complete_run)


def test_dismissal_without_reason_is_as011(complete_run):
    write_jsonl(complete_run / "dismissed.jsonl", [{"file": "SKILL.md", "lines": [9], "reason": ""}])
    assert "AS011" in codes(complete_run)


@pytest.mark.parametrize("field,value", [("kind", "vibe"), ("verdict", "fine"), ("id", "I002"), ("line", "x")])
def test_bad_schema_is_as011(complete_run, field, value):
    set_items(complete_run, lambda items: items[0].update({field: value}))
    assert "AS011" in codes(complete_run)


def test_in_scope_item_not_checked_is_as012(complete_run):
    set_items(complete_run, lambda items: items[0].update({"verdict": "not-checked"}))
    assert "AS012" in codes(complete_run)


def test_out_of_scope_item_with_verdict_is_as012(complete_run):
    set_items(complete_run, lambda items: items[2].update({"verdict": "correct", "strength": "D"}))
    assert "AS012" in codes(complete_run)


def test_definite_verdict_without_evidence_is_as013(complete_run):
    set_items(complete_run, lambda items: items[0].update({"evidence": []}))
    assert "AS013" in codes(complete_run)


def test_unknown_source_is_as014(complete_run):
    set_items(complete_run, lambda items: items[1]["evidence"][0].update({"source_id": "S999"}))
    assert "AS014" in codes(complete_run)


def test_quote_not_in_source_is_as015(complete_run):
    set_items(complete_run, lambda items: items[1]["evidence"][0].update(
        {"quote": "It supports sound by default and multi-line editing."}))
    assert "AS015" in codes(complete_run)


def test_short_quote_is_as016(complete_run):
    set_items(complete_run, lambda items: items[1]["evidence"][0].update({"quote": "colour by default"}))
    assert "AS016" in codes(complete_run)


@pytest.mark.parametrize("tier", [5, 6])
def test_weak_or_excluded_tier_as_support_is_as017(complete_run, tier):
    set_items(complete_run, lambda items: items[1]["evidence"][0].update({"tier": tier}))
    assert "AS017" in codes(complete_run)


def test_a_trivial_command_does_not_switch_off_source_rules(complete_run):
    def change(items):
        items[1]["evidence"][0]["tier"] = 5
        items[1]["searches"] = []
        items[1]["evidence"].append({"type": "command", "command": "true", "output": "ok", "exit_code": 0})
    set_items(complete_run, change)
    assert {"AS017", "AS019"} <= codes(complete_run)


@pytest.mark.parametrize("verdict,strength", [("couldnt-verify", "C"), ("correct", "E"), ("correct", None)])
def test_verdict_strength_mismatch_is_as018(complete_run, verdict, strength):
    def change(items):
        items[1].update({"verdict": verdict})
        if strength is None:
            items[1].pop("strength")
        else:
            items[1]["strength"] = strength
    set_items(complete_run, change)
    assert "AS018" in codes(complete_run)


def test_source_verdict_without_search_log_is_as019(complete_run):
    set_items(complete_run, lambda items: items[1].update({"searches": ["Q404"]}))
    assert "AS019" in codes(complete_run)


def test_strength_a_needs_execution_or_two_strong_sources_as020(complete_run):
    set_items(complete_run, lambda items: items[1].update({"strength": "A"}))
    assert "AS020" in codes(complete_run)


def test_command_without_output_is_as021(complete_run):
    set_items(complete_run, lambda items: items[0]["evidence"][0].update({"output": ""}))
    assert "AS021" in codes(complete_run)


def test_no_disconfirming_search_is_as023(complete_run):
    searches = good_searches()
    searches[1]["purpose"] = "support"
    write_jsonl(complete_run / "search_log.jsonl", searches)
    assert "AS023" in codes(complete_run)


def test_disputed_without_both_positions_is_as024(complete_run):
    set_items(complete_run, lambda items: items[0].update({"verdict": "disputed"}))
    assert "AS024" in codes(complete_run)


def test_bad_search_entry_is_as025(complete_run):
    searches = good_searches()
    searches[0]["hits"] = "many"
    write_jsonl(complete_run / "search_log.jsonl", searches)
    assert "AS025" in codes(complete_run)


def test_targeted_scope_marks_files_outside_paths_as012(complete_run):
    meta = json.loads((complete_run / "meta.json").read_text())
    meta["scope"] = {"mode": "targeted", "paths": ["scripts"]}
    (complete_run / "meta.json").write_text(json.dumps(meta))
    assert "AS012" in codes(complete_run)
    items = read_jsonl(complete_run / "items.jsonl")
    for item in items:
        if not item["file"].startswith("scripts/"):
            item.update({"in_scope": False, "verdict": "not-checked", "evidence": [], "searches": []})
            item.pop("strength", None)
    write_jsonl(complete_run / "items.jsonl", items)
    assert "AS012" not in codes(complete_run)


def test_one_fake_search_row_does_not_satisfy_every_item(complete_run):
    write_jsonl(complete_run / "search_log.jsonl", [
        {"id": "Q001", "engine": "WebSearch", "query": "x", "date": "2026-11-07", "hits": 0, "opened": [],
         "purpose": "disconfirm", "item_ids": []}])
    set_items(complete_run, lambda items: items[1].update({"searches": ["Q001"]}))
    assert "AS019" in codes(complete_run)


def test_duplicate_search_ids_are_as025(complete_run):
    searches = good_searches()
    searches[1]["id"] = "Q001"
    write_jsonl(complete_run / "search_log.jsonl", searches)
    assert "AS025" in codes(complete_run)


def test_couldnt_verify_needs_logged_searches_as019(complete_run):
    set_items(complete_run, lambda items: items[1].update(
        {"verdict": "couldnt-verify", "strength": "E", "evidence": [], "searches": []}))
    assert "AS019" in codes(complete_run)


@pytest.mark.parametrize("tier,strength,ok", [(4, "D", True), (4, "C", False), (3, "C", True), (3, "B", False),
                                               (2, "B", True), (1, "B", True)])
def test_strength_ceiling_by_best_tier_as027(complete_run, tier, strength, ok):
    set_items(complete_run, lambda items: items[1].update({"strength": strength})
              or items[1]["evidence"][0].update({"tier": tier}))
    assert ("AS027" not in codes(complete_run)) == ok


def test_single_non_tier_1_source_caps_empirical_claims_at_c(complete_run):
    set_items(complete_run, lambda items: items[1].update({"kind": "empirical", "strength": "B"})
              or items[1]["evidence"][0].update({"tier": 2}))
    assert "AS027" in codes(complete_run)


def test_whole_file_dismissal_of_markdown_is_as026(complete_run):
    write_jsonl(complete_run / "dismissed.jsonl", [{"file": "SKILL.md", "all": True, "reason": "x"}])
    assert "AS026" in codes(complete_run)


def test_dismissing_a_script_is_as026(complete_run):
    set_items(complete_run, lambda items: items.pop(4))
    write_jsonl(complete_run / "dismissed.jsonl", [{"file": "SKILL.md", "lines": [15], "reason": "x"},
                                                   {"file": "scripts/tool.py", "lines": [1], "reason": "trivial"}])
    assert "AS026" in codes(complete_run)


def test_dismissal_must_include_the_first_line_of_a_code_block(complete_run):
    set_items(complete_run, lambda items: items.pop(3))
    write_jsonl(complete_run / "dismissed.jsonl", [{"file": "SKILL.md", "lines": [12, 15], "reason": "x"}])
    assert "AS010" in codes(complete_run)


@pytest.mark.parametrize("update", [{"end_line": 2}, {"end_line": "9"}, {"file": "nope.md"}, {"line": 99999},
                                    {"line": 1, "end_line": 10**6}])
def test_item_location_must_be_real_as011(complete_run, update):
    set_items(complete_run, lambda items: items[1].update(update))
    assert "AS011" in codes(complete_run)


def test_malformed_evidence_is_as011_not_a_crash(complete_run):
    set_items(complete_run, lambda items: items[0].update({"evidence": ["ran it"]}))
    assert "AS011" in codes(complete_run)


def test_auditor_cannot_mark_an_item_disputed(complete_run):
    set_items(complete_run, lambda items: items[1].update(
        {"verdict": "disputed", "dispute": {"auditor": {"verdict": "correct"}, "reviewer": {"verdict": "wrong"}}}))
    assert "AS024" in codes(complete_run)


def test_same_source_twice_does_not_lift_the_cap(complete_run):
    def change(items):
        items[1].update({"kind": "empirical", "strength": "A"})
        items[1]["evidence"][0]["tier"] = 2
        items[1]["evidence"].append(dict(items[1]["evidence"][0]))
    set_items(complete_run, change)
    assert {"AS020", "AS027"} & codes(complete_run)


@pytest.mark.parametrize("name,body", [("bin/deploy.sh", "echo hi\n"), ("helper.py", "x = 1\n"),
                                       ("references/defaults.yaml", "timeout: 5\n")])
def test_code_and_data_files_cannot_be_dismissed_whole(complete_run, name, body):
    meta = json.loads((complete_run / "meta.json").read_text())
    from pathlib import Path

    target = Path(meta["skill_path"]) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    import audit_common

    meta["tree_hash"] = audit_common.tree_hash(Path(meta["skill_path"]))
    (complete_run / "meta.json").write_text(json.dumps(meta))
    assert run_script("find_candidates.py", complete_run).returncode == 0
    rows = read_jsonl(complete_run / "dismissed.jsonl") + [{"file": name, "all": True, "reason": "trivial"}]
    write_jsonl(complete_run / "dismissed.jsonl", rows)
    assert "AS026" in codes(complete_run)


def test_dismissal_of_a_file_not_in_the_skill_is_as011(complete_run):
    rows = read_jsonl(complete_run / "dismissed.jsonl") + [{"file": "## Injected", "lines": [1], "reason": "x"}]
    write_jsonl(complete_run / "dismissed.jsonl", rows)
    assert "AS011" in codes(complete_run)


def add_second_source(run, tier_a, tier_b, strength, kind="tool-behavior"):
    from conftest import SOURCE_TEXT, add_source

    add_source(run, text=SOURCE_TEXT, url="https://example.org/second", source_id="S002")

    def change(items):
        items[1].update({"strength": strength, "kind": kind})
        items[1]["evidence"][0]["tier"] = tier_a
        items[1]["evidence"].append(dict(items[1]["evidence"][0], source_id="S002", tier=tier_b))
    set_items(run, change)


def test_two_distinct_strong_sources_allow_a(complete_run):
    add_second_source(complete_run, 1, 2, "A")
    assert not {"AS020", "AS027"} & codes(complete_run)


def test_two_tier_3_sources_do_not_allow_a(complete_run):
    add_second_source(complete_run, 3, 3, "A")
    assert {"AS020", "AS027"} <= codes(complete_run)


def test_best_tier_sets_the_ceiling(complete_run):
    add_second_source(complete_run, 1, 4, "B")
    assert "AS027" not in codes(complete_run)


@pytest.mark.parametrize("kind,tier,strength", [("empirical", 1, "B"), ("recommendation", 2, "C")])
def test_single_source_cap_edges(complete_run, kind, tier, strength):
    set_items(complete_run, lambda items: items[1].update({"kind": kind, "strength": strength})
              or items[1]["evidence"][0].update({"tier": tier}))
    assert "AS027" not in codes(complete_run)


def test_targeted_scope_with_two_paths(complete_run):
    meta = json.loads((complete_run / "meta.json").read_text())
    meta["scope"] = {"mode": "targeted", "paths": ["references", "scripts"]}
    (complete_run / "meta.json").write_text(json.dumps(meta))
    items = read_jsonl(complete_run / "items.jsonl")
    for item in items:
        if not item["file"].startswith("scripts/"):
            item.update({"in_scope": False, "verdict": "not-checked", "evidence": [], "searches": []})
            item.pop("strength", None)
    write_jsonl(complete_run / "items.jsonl", items)
    assert "AS012" not in codes(complete_run)


@pytest.mark.parametrize("update", [{"text": "   "}, {"searches": [1, 2]}])
def test_more_schema_edges_are_as011(complete_run, update):
    set_items(complete_run, lambda items: items[0].update(update))
    assert "AS011" in codes(complete_run)


def test_merge_disputed_item_still_needs_both_positions(complete_run):
    (complete_run / "review").mkdir(exist_ok=True)
    (complete_run / "review" / "state.json").write_text(json.dumps({"disputed_ids": ["I001"]}))
    set_items(complete_run, lambda items: items[0].update({"verdict": "disputed", "dispute": {"auditor": {"v": 1}}}))
    assert "AS024" in codes(complete_run)
    set_items(complete_run, lambda items: items[0].update(
        {"verdict": "disputed", "dispute": {"auditor": {"v": 1}, "reviewer": {"v": 2}}}))
    assert "AS024" not in codes(complete_run)


def test_text_item_span_boundary(complete_run):
    from pathlib import Path

    import audit_common

    meta = json.loads((complete_run / "meta.json").read_text())
    skill = Path(meta["skill_path"])
    (skill / "long.md").write_text("".join(f"line {n}\n" for n in range(1, 81)))
    meta["tree_hash"] = audit_common.tree_hash(skill)
    (complete_run / "meta.json").write_text(json.dumps(meta))
    base = {"kind": "recommendation", "text": "x", "in_scope": False, "verdict": "not-checked",
            "evidence": [], "searches": []}
    items = read_jsonl(complete_run / "items.jsonl")
    write_jsonl(complete_run / "items.jsonl", [*items, {**base, "id": "L1", "file": "long.md", "line": 1, "end_line": 61}])
    ok = [f for f in json.loads(run_script("validate_audit.py", complete_run, "--json").stdout)["findings"]
          if f["code"] == "AS011" and "span" in f["message"]]
    assert not ok
    write_jsonl(complete_run / "items.jsonl", [*items, {**base, "id": "L1", "file": "long.md", "line": 1, "end_line": 62}])
    bad = [f for f in json.loads(run_script("validate_audit.py", complete_run, "--json").stdout)["findings"]
           if f["code"] == "AS011" and "span" in f["message"]]
    assert bad
