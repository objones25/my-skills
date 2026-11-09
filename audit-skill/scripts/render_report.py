#!/usr/bin/env python3
"""Render the audit report from the run's files. The report is generated, never hand-written.

Usage:
    python render_report.py RUN_DIR

Refuses (exit 1) unless validate_audit.py is clean, the review loop has finished
(clean, or capped at 3 rounds), and no checked item changed after the reviewer
accepted it. Writes the report to meta.json's "report" path (audits/<skill>/<date>.md).

The first section is always Coverage. The title says "Audit of X (full)" only when
the scope was full, every inventory item has a verdict, and every file in the skill has
an item or was dismissed whole (allowed only for non-text files); otherwise it says
"Partial audit of X". Everything the report is built from is frozen when the review
ends, and the report refuses to render if any of it changed afterwards. Exit 0 written, 1 refused, 2 cannot read the run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_common import (
    NOT_COPIED_DIRS,
    item_hash,
    iter_skill_files,
    load_meta,
    read_jsonl,
    run_snapshot,
)
from review_round import load_state, reviewed_but_hidden
from validate_audit import validate

SCOPE_WORDS = {"risk": "risk-based", "full": "full", "targeted": "targeted"}
FINDING_ORDER = ("wrong", "stale", "version-specific", "disputed")
OUTPUT_LINES = 15


def inline(text: Any) -> str:
    """One line of user-supplied text, unable to start a heading, list or quote block."""
    flat = " ".join(str(text).split())
    return "\\" + flat if flat[:1] in "#>-*+|=" and flat else flat


def code_span(text: Any) -> str:
    """Inline code that a backtick inside the text cannot close early."""
    flat = " ".join(str(text).split())
    ticks = "`" * (1 + max((len(run) for run in re.findall(r"`+", flat)), default=0))
    return f"{ticks} {flat} {ticks}"


def cell(text: Any) -> str:
    return inline(text).replace("|", "\\|")


def evidence_lines(item: dict[str, Any], sources: dict[str, dict]) -> list[str]:
    lines = []
    for ev in item.get("evidence", []):
        if ev.get("type") == "command":
            output = str(ev["output"]).splitlines()
            shown = output[:OUTPUT_LINES]
            more = f"\n... {len(output) - OUTPUT_LINES} more line(s) in items.jsonl" if len(output) > OUTPUT_LINES else ""
            body = "\n".join(shown) + more
            fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", body)), default=0))
            command = str(ev["command"])
            if "\n" in command:
                cfence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", command)), default=0))
                shown_cmd = f"this command (exit {inline(ev.get('exit_code', '?'))}):\n\n{cfence}bash\n{command}\n{cfence}"
                lines.append(f"- Ran {shown_cmd}\n\n  Output:\n\n{fence}text\n{body}\n{fence}")
            else:
                lines.append(f"- Ran {code_span(command)} (exit {inline(ev.get('exit_code', '?'))}):\n\n"
                             f"{fence}text\n{body}\n{fence}")
        else:
            src = sources[ev["source_id"]]
            interest = f"; interest: {ev['interest']}" if ev.get("interest") and ev["interest"] != "none" else ""
            lines.append(f"- [{inline(src['id'])}] tier {ev['tier']}{inline(interest)}, <{inline(src['url'])}>\n"
                         f"  > {inline(ev['quote'])}")
    return lines


def item_block(item: dict[str, Any], sources: dict[str, dict]) -> str:
    head = (f"### {inline(item['id'])} {code_span(str(item['file']) + ':' + str(item['line']))}: {item['verdict']}"
            + (f", strength {item['strength']}" if item.get("strength") else ""))
    parts = [head, "", f"**Claim:** {inline(item['text'])}"]
    if item.get("note"):
        parts.append(f"**Why:** {inline(item['note'])}")
    if item.get("dispute"):
        d = item["dispute"]
        parts.append(f"**Auditor:** {json.dumps(d['auditor'])}  \n**Reviewer:** {json.dumps(d['reviewer'])}")
    parts += [""] + evidence_lines(item, sources)
    return "\n".join(parts) + "\n"


def render(run: Path) -> str:
    meta = load_meta(run)
    state = load_state(run)
    items = read_jsonl(run / "items.jsonl")
    sources = {s["id"]: s for s in read_jsonl(run / "sources" / "index.jsonl")}
    searches = read_jsonl(run / "search_log.jsonl")
    dismissed = read_jsonl(run / "dismissed.jsonl")
    suggestions = read_jsonl(run / "suggestions.jsonl")

    total = len(items)
    checked = [i for i in items if i["verdict"] != "not-checked"]
    skill_files = [r.as_posix() for r in iter_skill_files(Path(meta["skill_path"]))]
    per_file: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        per_file[item["file"]].append(item)
    whole = {d["file"] for d in dismissed if d.get("all") is True}
    fully = [f for f in skill_files if per_file[f] and all(r["verdict"] != "not-checked" for r in per_file[f])]
    partly = [f for f in skill_files if per_file[f] and f not in fully
              and any(r["verdict"] != "not-checked" for r in per_file[f])]
    untouched = [f for f in skill_files if per_file[f] and f not in fully and f not in partly]
    no_items = [f for f in skill_files if not per_file[f]]
    skill_root = Path(meta["skill_path"])
    vendored = sorted({p.relative_to(skill_root).parts[0] for d in NOT_COPIED_DIRS for p in skill_root.rglob(d)
                       if p.is_dir()})
    full = (meta["scope"]["mode"] == "full" and total > 0 and len(checked) == total
            and all(f in whole for f in no_items) and not vendored)
    in_scope = [i for i in items if i.get("in_scope")]
    scope = SCOPE_WORDS[meta["scope"]["mode"]]
    if meta["scope"]["paths"]:
        scope += " (" + ", ".join(f"`{p}`" for p in meta["scope"]["paths"]) + ")"
    title = f"# Audit of {meta['skill']} (full)" if full else f"# Partial audit of {meta['skill']}"
    rounds = state["rounds_done"]
    disputed = sum(i["verdict"] == "disputed" for i in items)
    if state["status"] == "clean":
        review = f"Independent review: clean after {rounds} round{'s' if rounds != 1 else ''}."
    else:
        review = f"Independent review: stopped at the {rounds}-round cap; {disputed} item(s) disputed."
    pct = round(100 * len(checked) / total) if total else 0
    counts = Counter(i["verdict"] for i in items)
    words = {"correct": "confirmed", "couldnt-verify": "could not be verified"}
    breakdown = ", ".join(f"{counts[v]} {words.get(v, v)}" for v in
                          ("correct", "wrong", "stale", "version-specific", "disputed", "couldnt-verify") if counts[v])

    def names(files: list[str]) -> str:
        return ", ".join(f"`{f}`" for f in files) or "none"

    out = [title, "", f"Date: {meta['date']}. Skill: `{meta['skill_path']}`. Scope: {scope}.", "",
           "## Coverage", "",
           f"Checked {len(checked)} of {total} items ({pct}%){': ' + breakdown if breakdown else ''}. {review}", "",
           (f"- In scope: {len(in_scope)} items, of which {sum(i['verdict'] != 'not-checked' for i in in_scope)} "
           f"checked. The rest of the inventory is listed under Not checked."),
           f"- Files fully checked: {names(fully)}",
           f"- Files partly checked: {names(partly)}",
           f"- Files not checked: {names(untouched)}",
           (f"- Files with no inventoried claim (every candidate line dismissed): "
           f"{names([f for f in no_items if f not in whole])}"),
           f"- Files dismissed whole: {names([f for f in no_items if f in whole])}",
           f"- Dismissals: {len(dismissed)}, each listed with its reason under Not checked.",
           f"- Not inventoried (virtualenv or vendored folders): {names(vendored)}", ""]
    if not full:
        out += ["This is a partial audit. Nothing here describes the items that were not checked.", ""]

    out += ["## Summary", "", "| Verdict | Items |", "|---|---|"]
    out += [f"| {v} | {counts[v]} |" for v in ("wrong", "stale", "version-specific", "disputed", "couldnt-verify",
                                                 "correct", "not-checked") if counts[v]]
    out.append("")

    out += ["## Findings", ""]
    findings = sorted((i for i in items if i["verdict"] in FINDING_ORDER),
                      key=lambda i: (FINDING_ORDER.index(i["verdict"]), i["file"], i["line"]))
    out += [item_block(i, sources) for i in findings] or ["No item was found wrong, stale or version-specific.", ""]

    unverified = [i for i in items if i["verdict"] == "couldnt-verify"]
    out += ["## Could not verify", ""]
    out += [item_block(i, sources) for i in unverified] or ["None.", ""]

    correct = [i for i in items if i["verdict"] == "correct"]
    out += ["## Confirmed", "", "| Item | Location | Claim | Strength | Evidence |", "|---|---|---|---|---|"]
    for i in correct:
        n_cmd = sum(ev["type"] == "command" for ev in i["evidence"])
        cited = [ev["source_id"] for ev in i["evidence"] if ev["type"] == "source"]
        refs = ", ".join(([f"{n_cmd} command{'s' if n_cmd != 1 else ''}"] if n_cmd else []) + cited)
        out.append(f"| {cell(i['id'])} | {cell(code_span(str(i['file']) + ':' + str(i['line'])))} | {cell(i['text'])} "
                   f"| {i['strength']} | {cell(refs)} |")
    out.append("")
    if correct:
        out += ["Evidence for each confirmed item:", ""]
    out += [item_block(i, sources) for i in correct]

    out += ["## Suggestions (unverified)", "",
            "Concerns the auditor could not prove with a failing input or a source. Not findings.", ""]
    out += [f"- {code_span(str(s.get('file', '?')) + ':' + str(s.get('line', '?')))} {inline(s['text'])}"
            for s in suggestions] or ["None."]
    out.append("")

    out += ["## Not checked", ""]
    out += [f"- {inline(i['id'])} {code_span(str(i['file']) + ':' + str(i['line']))} {inline(i['text'])}"
            for i in items if i["verdict"] == "not-checked"] \
        or ["None."]
    out += ["", "Dismissed candidate lines:", ""]
    out += [f"- {code_span(d['file'])} {'(whole file)' if d.get('all') else inline(d.get('lines'))}: {inline(d['reason'])}"
            for d in dismissed] or ["None."]
    out.append("")

    purposes = Counter(s.get("purpose") for s in searches)
    out += ["## How things were checked", "",
            (f"- {len(searches)} logged searches ({', '.join(f'{k} {v}' for k, v in sorted(purposes.items()))}); "
            "every query is in `search_log.jsonl` in the run folder."),
            (f"- {len(sources)} sources fetched raw by `fetch_source.py`. Every quote above was string-matched "
             "against that saved text by `validate_audit.py`; for every item the reviewer accepted, the quote was "
             "also found in the reviewer's own fetch. Disputed items show the reviewer's problems."),
            "- Review: " + ", ".join("round {round}: {reviewed} reviewed, {open} open".format(**h)
                                     for h in state["history"]) + ".",
            "- Tools:"]
    out += [f"  - {t.get('name', '?')} {t.get('version', '?')} ({t.get('how', '?')})" for t in meta.get("tools", [])] \
        or ["  - none recorded"]
    out += ["", "## Sources", "", "| Id | URL | Fetched | SHA-256 |", "|---|---|---|---|"]
    out += [f"| {cell(s['id'])} | {cell(s['url'])} | {cell(s['fetched_at'])} | `{cell(s['sha256'][:12])}` |"
            for s in sources.values()]
    return "\n".join(out) + "\n"


def refusal(run: Path) -> str | None:
    findings = validate(run)
    if findings:
        return f"validate_audit.py reports {len(findings)} finding(s); fix them first"
    state = load_state(run)
    if state["status"] not in ("clean", "capped"):
        return "the review loop has not finished; run review_round.py"
    for item in read_jsonl(run / "items.jsonl"):
        if item["verdict"] not in ("not-checked", "disputed") and state["accepted"].get(item["id"]) != item_hash(item):
            return f"{item['id']} changed after the reviewer accepted it, or was never reviewed; run another round"
    hidden = reviewed_but_hidden(state, read_jsonl(run / "items.jsonl"))
    if hidden:
        return hidden
    if state.get("final") != run_snapshot(run):
        return ("the items, dismissals, candidates, suggestions or search log changed after the review finished; "
                "restore them, or start a new audit run")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        reason = refusal(args.run_dir)
        if reason:
            print(f"render_report: {reason}", file=sys.stderr)
            return 1
        meta = load_meta(args.run_dir)
        report = Path(meta["report"])
        report.write_text(render(args.run_dir), encoding="utf-8")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"render_report: cannot read the run: {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
