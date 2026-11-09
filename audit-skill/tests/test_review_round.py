"""review_round.py: a blind packet for the reviewer, a mechanical merge, and a 3-round cap."""

from __future__ import annotations

import json

from conftest import SOURCE_TEXT, add_source, read_jsonl, run_script, write_jsonl


def packet(run):
    return run_script("review_round.py", "packet", run)


def merge(run):
    return run_script("review_round.py", "merge", run)


def agreeing_verdicts(run, round_no=1, skip=()):
    rows = []
    for entry in read_jsonl(run / "review" / f"round-{round_no}" / "packet.jsonl"):
        if entry["id"] in skip:
            continue
        verdict = {"I001": "correct", "I002": "correct", "I005": "correct"}[entry["id"]]
        strength = {"I001": "A", "I002": "B", "I005": "A"}[entry["id"]]
        rows.append({"id": entry["id"], "verdict": verdict, "strength": strength,
                     "reran": [{"command": c, "output": "same output"} for c in entry["commands"]],
                     "problems": []})
    write_jsonl(run / "review" / f"round-{round_no}" / "verdicts.jsonl", rows)


def refetch(run, round_no=1, text=SOURCE_TEXT):
    add_source(run, text=text, dest=f"review/round-{round_no}/sources")


def state(run):
    return json.loads((run / "review" / "state.json").read_text())


def test_packet_hides_verdicts_reasoning_tiers_and_outputs(complete_run):
    proc = packet(complete_run)
    assert proc.returncode == 0, proc.stderr
    entries = read_jsonl(complete_run / "review" / "round-1" / "packet.jsonl")
    assert {e["id"] for e in entries} == {"I001", "I002", "I005"}
    text = json.dumps(entries)
    for hidden in ("verdict", "strength", "note", "output", "tier", "Official what's-new page"):
        assert hidden not in text
    i002 = next(e for e in entries if e["id"] == "I002")
    assert i002["sources"] == [{"url": "https://docs.python.org/3/whatsnew/3.13.html",
                                "quote": "It supports colour by default and multi-line editing."}]


def test_packet_refuses_an_invalid_audit(complete_run):
    write_jsonl(complete_run / "items.jsonl", [])
    assert packet(complete_run).returncode == 1


def test_agreeing_review_is_clean(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    refetch(complete_run)
    proc = merge(complete_run)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert state(complete_run)["status"] == "clean"


def test_merge_refuses_when_a_source_was_not_refetched(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    proc = merge(complete_run)
    assert proc.returncode == 2
    assert "not refetched" in proc.stderr


def test_merge_refuses_when_a_command_was_not_rerun(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows = [dict(r, reran=[]) if r["id"] == "I001" else r for r in rows]
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_merge_refuses_a_missing_verdict(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run, skip={"I005"})
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_quote_missing_from_reviewer_copy_becomes_feedback(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    refetch(complete_run, text="A different page entirely, with nothing about the shell in it at all.")
    proc = merge(complete_run)
    assert proc.returncode == 1
    feedback = read_jsonl(complete_run / "review" / "round-1" / "feedback.jsonl")
    assert feedback[0]["id"] == "I002"
    assert feedback[0]["problems"][0]["code"] == "quote-not-in-reviewer-copy"


def test_verdict_disagreement_becomes_feedback_and_next_packet_has_only_open_items(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows[0].update({"verdict": "wrong", "problems": [{"code": "output-differs", "detail": "head shows otherwise"}]})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 1
    assert packet(complete_run).returncode == 0
    assert [e["id"] for e in read_jsonl(complete_run / "review" / "round-2" / "packet.jsonl")] == [rows[0]["id"]]


def test_missing_item_report_becomes_feedback(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows.append({"id": "new", "file": "SKILL.md", "line": 15, "text": "a missed claim",
                 "problems": [{"code": "missing-item", "detail": "line 15 states a fact"}]})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 1
    assert any(f["id"] is None for f in read_jsonl(complete_run / "review" / "round-1" / "feedback.jsonl"))


def test_after_three_rounds_open_items_become_disputed(complete_run):
    for round_no in (1, 2, 3):
        assert packet(complete_run).returncode == 0
        agreeing_verdicts(complete_run, round_no)
        rows = read_jsonl(complete_run / "review" / f"round-{round_no}" / "verdicts.jsonl")
        for row in rows:
            if row["id"] == "I001":
                row.update({"verdict": "wrong", "problems": [{"code": "output-differs", "detail": "still wrong"}]})
        write_jsonl(complete_run / "review" / f"round-{round_no}" / "verdicts.jsonl", rows)
        if round_no == 1:
            refetch(complete_run, round_no)
        result = merge(complete_run)
    assert result.returncode == 0
    assert state(complete_run)["status"] == "capped"
    item = next(i for i in read_jsonl(complete_run / "items.jsonl") if i["id"] == "I001")
    assert item["verdict"] == "disputed"
    assert item["dispute"]["auditor"]["verdict"] == "correct"
    assert item["dispute"]["reviewer"]["verdict"] == "wrong"
    assert packet(complete_run).returncode == 2
    assert run_script("validate_audit.py", complete_run).returncode == 0


def test_merge_refuses_a_new_row_without_location_or_text(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows.append({"id": "new", "problems": [{"code": "not-atomic", "detail": "bundles facts"}]})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    proc = merge(complete_run)
    assert proc.returncode == 2
    assert "new" in proc.stderr


def test_unchanged_item_after_feedback_is_reviewed_again_blind(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows[0]["problems"] = [{"code": "weak-search", "detail": "one query only"}]
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 1
    assert packet(complete_run).returncode == 0
    assert [e["id"] for e in read_jsonl(complete_run / "review" / "round-2" / "packet.jsonl")] == [rows[0]["id"]]


def run_dir_packet(run, round_no=1):
    return run / "review" / f"round-{round_no}" / "packet.jsonl"


def test_merge_refuses_rerun_without_output(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    for row in rows:
        row["reran"] = [r["command"] for r in row["reran"]]
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_merge_refuses_a_verdict_without_strength(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows[0].pop("strength")
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_merge_refuses_a_refetch_older_than_the_packet(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    add_source(complete_run, dest="review/round-1/sources", fetched_at="2000-01-01T00:00:00Z")
    proc = merge(complete_run)
    assert proc.returncode == 2
    assert "before the packet" in proc.stderr


def test_merge_refuses_a_new_row_for_a_file_not_in_the_skill(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows.append({"id": "new", "file": "nope.md", "line": 1, "text": "x", "problems": [{"code": "missing-item"}]})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_missed_item_outside_targeted_scope_at_cap_does_not_wedge_the_run(complete_run):
    meta = json.loads((complete_run / "meta.json").read_text())
    meta["scope"] = {"mode": "targeted", "paths": ["scripts"]}
    (complete_run / "meta.json").write_text(json.dumps(meta))
    items = read_jsonl(complete_run / "items.jsonl")
    for item in items:
        if not item["file"].startswith("scripts/"):
            item.update({"in_scope": False, "verdict": "not-checked", "evidence": [], "searches": []})
            item.pop("strength", None)
    write_jsonl(complete_run / "items.jsonl", items)
    for round_no in (1, 2, 3):
        assert packet(complete_run).returncode == 0
        rows = [{"id": "I005", "verdict": "wrong", "strength": "A",
                 "reran": [{"command": "python scripts/tool.py", "output": "boom"}],
                 "problems": [{"code": "output-differs", "detail": "boom"}]}]
        if round_no == 3:
            rows.append({"id": "new", "file": "SKILL.md", "line": 15, "kind": "vibe", "text": "missed claim",
                         "problems": [{"code": "missing-item", "detail": "line 15"}]})
        write_jsonl(complete_run / "review" / f"round-{round_no}" / "verdicts.jsonl", rows)
        merge(complete_run)
    assert state(complete_run)["status"] == "capped"
    assert run_script("validate_audit.py", complete_run).returncode == 0, run_script(
        "validate_audit.py", complete_run).stdout
    assert run_script("render_report.py", complete_run).returncode == 0


def test_new_row_line_must_exist_in_the_file(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows.append({"id": "new", "file": "SKILL.md", "line": 99999, "text": "x", "problems": [{"code": "missing-item"}]})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_reviewer_cannot_answer_disputed_or_not_checked(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows[0]["verdict"] = "disputed"
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_refetch_path_must_stay_in_the_reviewer_folder(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    refetch(complete_run)
    index = complete_run / "review" / "round-1" / "sources" / "index.jsonl"
    rows = read_jsonl(index)
    rows[0]["path"] = "../../../sources/S001.txt"
    write_jsonl(index, rows)
    assert merge(complete_run).returncode == 2


def test_malformed_reviewer_row_exits_2(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows[0]["verdict"] = ["correct"]
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_strength_only_disagreement_takes_the_weaker_strength_without_a_round(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    for row in rows:
        if row["id"] == "I001":
            row["strength"] = "D"
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 0
    item = next(i for i in read_jsonl(complete_run / "items.jsonl") if i["id"] == "I001")
    assert item["strength"] == "D"
    assert run_script("render_report.py", complete_run).returncode == 0


def test_reviewer_strength_must_fit_its_verdict(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows[0]["strength"] = "E"
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2


def test_reran_may_record_the_adapted_command(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    for row in rows:
        for r in row["reran"]:
            r["ran_as"] = r["command"].replace("scripts/", "./scripts/")
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 0


def test_packet_lists_every_inventoried_location_without_verdicts(complete_run):
    packet(complete_run)
    covered = read_jsonl(complete_run / "review" / "round-1" / "covered.jsonl")
    assert {c["id"] for c in covered} == {"I001", "I002", "I003", "I004", "I005"}
    assert all(set(c) == {"id", "file", "line", "end_line", "text"} for c in covered)


def test_merge_refuses_rows_with_unknown_ids_or_codes(complete_run):
    packet(complete_run)
    agreeing_verdicts(complete_run)
    rows = read_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl")
    rows.append({"id": None, "file": "SKILL.md", "line": 8, "text": "x", "problems": [{"code": "new"}]})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    refetch(complete_run)
    assert merge(complete_run).returncode == 2
    rows[-1] = {"id": "I001", **{k: v for k, v in rows[0].items() if k != "id"}}
    rows[0]["problems"] = [{"code": "made-up", "detail": "x"}]
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows[:-1])
    assert merge(complete_run).returncode == 2
