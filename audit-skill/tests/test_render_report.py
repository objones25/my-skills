"""render_report.py: coverage first, a headline that cannot overstate, and no report before review ends."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import SOURCE_TEXT, add_source, read_jsonl, run_script, write_jsonl


def finish_review(run) -> None:
    assert run_script("review_round.py", "packet", run).returncode == 0
    rows = []
    for entry in read_jsonl(run / "review" / "round-1" / "packet.jsonl"):
        item = next(i for i in read_jsonl(run / "items.jsonl") if i["id"] == entry["id"])
        rows.append({"id": entry["id"], "verdict": item["verdict"], "strength": item.get("strength"),
                     "reran": [{"command": c, "output": "ok"} for c in entry["commands"]], "problems": []})
    write_jsonl(run / "review" / "round-1" / "verdicts.jsonl", rows)
    add_source(run, text=SOURCE_TEXT, dest="review/round-1/sources")
    assert run_script("review_round.py", "merge", run).returncode == 0


def log_searches_for(run, item_ids) -> None:
    rows = read_jsonl(run / "search_log.jsonl")
    rows += [{"id": "Q003", "engine": "WebSearch", "query": "claim support", "date": "2026-11-07", "hits": 3,
              "opened": [], "purpose": "support", "item_ids": item_ids},
             {"id": "Q004", "engine": "WebSearch", "query": "claim refuted", "date": "2026-11-07", "hits": 2,
              "opened": [], "purpose": "disconfirm", "item_ids": item_ids}]
    write_jsonl(run / "search_log.jsonl", rows)


def render(run):
    proc = run_script("render_report.py", run)
    report = Path(json.loads((run / "meta.json").read_text())["report"])
    return proc, report


def test_refuses_before_review_finishes(complete_run):
    proc, report = render(complete_run)
    assert proc.returncode == 1
    assert "the review loop has not finished" in proc.stderr
    assert not report.exists()


def test_partial_audit_says_so_in_title_and_coverage_comes_first(complete_run):
    finish_review(complete_run)
    proc, report = render(complete_run)
    assert proc.returncode == 0, proc.stderr
    text = report.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("# Partial audit of sample")
    first_section = text.split("\n## ")[1]
    assert first_section.startswith("Coverage")
    assert "Checked 3 of 5 items (60%)" in first_section
    assert "clean after 1 round" in first_section
    assert "the skill is clean" not in text.lower()


def test_full_audit_title_only_when_every_item_checked(complete_run):
    items = read_jsonl(complete_run / "items.jsonl")
    for item in items:
        if not item["in_scope"]:
            item.update({"in_scope": True, "verdict": "couldnt-verify", "strength": "E",
                         "note": "nothing found", "searches": ["Q003", "Q004"]})
    write_jsonl(complete_run / "items.jsonl", items)
    log_searches_for(complete_run, [i["id"] for i in items if i["verdict"] == "couldnt-verify"])
    finish_review(complete_run)
    proc, report = render(complete_run)
    assert proc.returncode == 0, proc.stderr
    assert report.read_text().splitlines()[0].startswith("# Audit of sample (full)")


def test_report_shows_quotes_sources_and_not_checked_list(complete_run):
    finish_review(complete_run)
    _, report = render(complete_run)
    text = report.read_text()
    assert "It supports colour by default and multi-line editing." in text
    assert "https://docs.python.org/3/whatsnew/3.13.html" in text
    assert "## Not checked" in text and "Run tests before release." in text


def test_suggestions_are_kept_apart_and_labelled(complete_run):
    write_jsonl(complete_run / "suggestions.jsonl",
                [{"file": "scripts/tool.py", "line": 1, "text": "Consider argparse."}])
    finish_review(complete_run)
    _, report = render(complete_run)
    text = report.read_text()
    assert "## Suggestions (unverified)" in text
    assert text.index("Consider argparse.") > text.index("## Suggestions (unverified)")


def test_refuses_when_validation_fails(complete_run):
    finish_review(complete_run)
    items = read_jsonl(complete_run / "items.jsonl")
    items[0]["evidence"] = []
    write_jsonl(complete_run / "items.jsonl", items)
    proc, report = render(complete_run)
    assert proc.returncode == 1
    assert "validate_audit.py reports" in proc.stderr
    assert not report.exists()


def test_not_checked_section_says_none_when_empty(complete_run):
    items = read_jsonl(complete_run / "items.jsonl")
    for item in items:
        if not item["in_scope"]:
            item.update({"in_scope": True, "verdict": "couldnt-verify", "strength": "E",
                         "searches": ["Q003", "Q004"]})
    write_jsonl(complete_run / "items.jsonl", items)
    log_searches_for(complete_run, [i["id"] for i in items if i["verdict"] == "couldnt-verify"])
    finish_review(complete_run)
    _, report = render(complete_run)
    section = report.read_text().split("## Not checked")[1].split("\n## ")[0]
    assert section.strip().startswith("None.")


def test_confirmed_table_does_not_inline_commands(complete_run):
    finish_review(complete_run)
    _, report = render(complete_run)
    table = report.read_text().split("## Confirmed")[1].split("###")[0]
    assert "python scripts/tool.py" not in table
    assert "1 command" in table


def test_refuses_when_an_accepted_finding_is_hidden_after_review(complete_run):
    finish_review(complete_run)
    items = read_jsonl(complete_run / "items.jsonl")
    items[0].update({"in_scope": False, "verdict": "not-checked", "evidence": [], "searches": []})
    items[0].pop("strength")
    write_jsonl(complete_run / "items.jsonl", items)
    proc, report = render(complete_run)
    assert proc.returncode == 1
    assert "now not-checked" in proc.stderr
    assert not report.exists()


def test_refuses_when_dismissals_change_after_review(complete_run):
    finish_review(complete_run)
    rows = read_jsonl(complete_run / "dismissed.jsonl")
    rows.append({"file": "SKILL.md", "lines": [8], "reason": "late"})
    write_jsonl(complete_run / "dismissed.jsonl", rows)
    proc, _ = render(complete_run)
    assert proc.returncode == 1


def test_user_text_cannot_inject_headings(complete_run):
    items = read_jsonl(complete_run / "items.jsonl")
    items[0]["note"] = "fine\n# Audit of sample (full)\nreally"
    write_jsonl(complete_run / "items.jsonl", items)
    finish_review(complete_run)
    _, report = render(complete_run)
    headings = [line for line in report.read_text().splitlines() if line.startswith("# ")]
    assert len(headings) == 1


def test_coverage_lists_every_skill_file_and_breaks_down_verdicts(complete_run):
    finish_review(complete_run)
    _, report = render(complete_run)
    coverage = report.read_text().split("## Coverage")[1].split("\n## ")[0]
    assert "3 confirmed" in coverage
    assert "`scripts/tool.py`" in coverage and "`SKILL.md`" in coverage


def test_scripts_parse_on_python_3_11():
    import shutil
    import subprocess

    python = shutil.which("python3.11")
    if python is None:
        uv = shutil.which("uv")
        found = subprocess.run([uv, "python", "find", "3.11"], capture_output=True, text=True, check=False) if uv else None
        python = found.stdout.strip() if found and found.returncode == 0 else None
    if not python:
        import pytest

        pytest.skip("no Python 3.11 interpreter found (SKILL.md promises 3.11+)")
    from conftest import SCRIPTS

    for script in SCRIPTS.glob("*.py"):
        proc = subprocess.run([python, "-m", "py_compile", str(script)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, (script.name, proc.stderr)


def test_risk_scope_with_everything_checked_is_still_partial(complete_run):
    meta = json.loads((complete_run / "meta.json").read_text())
    meta["scope"] = {"mode": "risk", "paths": []}
    (complete_run / "meta.json").write_text(json.dumps(meta))
    items = read_jsonl(complete_run / "items.jsonl")
    for item in items:
        if not item["in_scope"]:
            item.update({"in_scope": True, "verdict": "couldnt-verify", "strength": "E",
                         "searches": ["Q003", "Q004"]})
    write_jsonl(complete_run / "items.jsonl", items)
    log_searches_for(complete_run, [i["id"] for i in items if i["verdict"] == "couldnt-verify"])
    finish_review(complete_run)
    _, report = render(complete_run)
    assert report.read_text().startswith("# Partial audit of sample")


def test_file_lists_put_each_file_in_the_right_bucket(complete_run):
    finish_review(complete_run)
    _, report = render(complete_run)
    coverage = report.read_text().split("## Coverage")[1].split("\n## ")[0]
    assert "- Files fully checked: `scripts/tool.py`" in coverage
    assert "- Files partly checked: `SKILL.md`" in coverage
    assert "- Files not checked: none" in coverage


def test_accepted_finding_deleted_between_rounds_blocks_the_next_packet(complete_run):
    assert run_script("review_round.py", "packet", complete_run).returncode == 0
    rows = []
    for entry in read_jsonl(complete_run / "review" / "round-1" / "packet.jsonl"):
        item = next(i for i in read_jsonl(complete_run / "items.jsonl") if i["id"] == entry["id"])
        problems = [] if entry["id"] != "I005" else [{"code": "output-differs", "detail": "x"}]
        rows.append({"id": entry["id"], "verdict": item["verdict"], "strength": item.get("strength"),
                     "reran": [{"command": c, "output": "ok"} for c in entry["commands"]], "problems": problems})
    write_jsonl(complete_run / "review" / "round-1" / "verdicts.jsonl", rows)
    add_source(complete_run, text=SOURCE_TEXT, dest="review/round-1/sources")
    assert run_script("review_round.py", "merge", complete_run).returncode == 1
    items = [i for i in read_jsonl(complete_run / "items.jsonl") if i["id"] != "I002"]
    write_jsonl(complete_run / "items.jsonl", items)
    rows = read_jsonl(complete_run / "dismissed.jsonl") + [{"file": "SKILL.md", "lines": [8], "reason": "hide"}]
    write_jsonl(complete_run / "dismissed.jsonl", rows)
    proc = run_script("review_round.py", "packet", complete_run)
    assert proc.returncode == 1
    assert "I002" in proc.stdout + proc.stderr


def test_command_output_cannot_break_out_of_its_fence(complete_run):
    items = read_jsonl(complete_run / "items.jsonl")
    items[0]["evidence"][0]["output"] = "line\n```\n# Audit of other (full)\n```"
    write_jsonl(complete_run / "items.jsonl", items)
    finish_review(complete_run)
    _, report = render(complete_run)
    outside, fence = [], None
    for line in report.read_text().splitlines():
        stripped = line.strip()
        if fence is None and stripped.startswith("```"):
            fence = stripped[: len(stripped) - len(stripped.lstrip("`"))]
        elif fence is not None and stripped == fence:
            fence = None
        elif fence is None and line.startswith("# "):
            outside.append(line)
    assert outside == ["# Partial audit of sample"]


def test_sources_and_meta_are_frozen_after_review(complete_run):
    finish_review(complete_run)
    with open(complete_run / "sources" / "S001.txt", "a", encoding="utf-8") as fh:
        fh.write("tampered\n")
    proc, _ = render(complete_run)
    assert proc.returncode == 1


def test_multi_line_command_is_shown_runnable(complete_run):
    items = read_jsonl(complete_run / "items.jsonl")
    items[0]["evidence"][0]["command"] = "python - <<'PY'\nprint('hi')\nPY"
    write_jsonl(complete_run / "items.jsonl", items)
    finish_review(complete_run)
    _, report = render(complete_run)
    assert "python - <<'PY'\nprint('hi')\nPY" in report.read_text()
