#!/usr/bin/env python3
"""Run the blind review loop: build a packet for the reviewer, then merge its verdicts.

Usage:
    python review_round.py packet RUN_DIR
    python review_round.py merge RUN_DIR

packet  Validates the audit (validate_audit.py must be clean), then writes
        review/round-N/packet.jsonl with every checked item not yet accepted. The packet
        is blind: claim, location, commands and cited URLs with their quotes, but no
        verdict, strength, note, command output or tier. Exit 0 written, 1 audit not
        valid, 2 review already finished (clean or capped at 3 rounds).

merge   Reads review/round-N/verdicts.jsonl (one row per packet item, plus rows with
        id "new" for missed items) and the reviewer's own fetches in
        review/round-N/sources/. Refuses (exit 2) unless every item has a verdict, every
        command was rerun (listed in "reran") and every cited URL was refetched. Then it
        adds mechanical problems: quote missing from the reviewer's copy, and verdict
        mismatch. When the verdicts agree and only the strengths are more than one level
        apart, the item takes the weaker strength (noted in review_note) and is accepted.
        Items with no problems are accepted. Problems go to review/round-N/feedback.jsonl for the auditor (exit 1).
        No problems: status clean (exit 0). After round 3, open items become "disputed"
        with both positions recorded, missed items are added as disputed (or not-checked
        when outside the targeted paths), and the status is capped (exit 0). Only this
        script marks items disputed. An item once sent to the reviewer may be fixed or
        downgraded but never deleted or taken out of scope; packet refuses if it is.

State lives in review/state.json, so the loop can resume in a later session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from audit_common import (
    KINDS,
    MAX_REVIEW_ROUNDS,
    STRENGTHS,
    item_hash,
    iter_skill_files,
    load_meta,
    read_jsonl,
    run_snapshot,
    write_jsonl,
)
from quote_match import quote_found
from validate_audit import in_scope_by_meta, validate


def load_state(run: Path) -> dict[str, Any]:
    path = run / "review" / "state.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"rounds_done": 0, "status": "open", "pending_round": None, "accepted": {}, "history": []}


def save_state(run: Path, state: dict[str, Any]) -> None:
    (run / "review").mkdir(exist_ok=True)
    (run / "review" / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def blind_entry(item: dict[str, Any], sources: dict[str, dict]) -> dict[str, Any]:
    commands = [ev["command"] for ev in item.get("evidence", []) if ev.get("type") == "command"]
    cited = [{"url": sources[ev["source_id"]]["url"], "quote": ev["quote"]}
             for ev in item.get("evidence", []) if ev.get("type") == "source"]
    return {"id": item["id"], "file": item["file"], "line": item["line"], "end_line": item.get("end_line"),
            "kind": item["kind"], "text": item["text"], "commands": commands, "sources": cited}


def cmd_packet(run: Path) -> int:
    state = load_state(run)
    if state["status"] in ("clean", "capped") or state["rounds_done"] >= MAX_REVIEW_ROUNDS:
        print(f"review_round: review already finished ({state['status']})", file=sys.stderr)
        return 2
    findings = validate(run)
    if findings:
        for f in findings:
            print(f"{f.code}  {f.where}: {f.message}")
        print(f"review_round: fix the {len(findings)} validation finding(s) before review", file=sys.stderr)
        return 1
    items = read_jsonl(run / "items.jsonl")
    hidden = reviewed_but_hidden(state, items)
    if hidden:
        print(f"review_round: {hidden}", file=sys.stderr)
        return 1
    round_no = state["rounds_done"] + 1
    sources = {row["id"]: row for row in read_jsonl(run / "sources" / "index.jsonl")}
    todo = [i for i in items if i["verdict"] != "not-checked" and state["accepted"].get(i["id"]) != item_hash(i)]
    folder = run / "review" / f"round-{round_no}"
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "packet.jsonl", [blind_entry(i, sources) for i in todo])
    write_jsonl(folder / "covered.jsonl", [{"id": i["id"], "file": i["file"], "line": i["line"],
                                            "end_line": i.get("end_line", i["line"]), "text": i["text"]}
                                           for i in items])
    (folder / "hashes.json").write_text(json.dumps({i["id"]: item_hash(i) for i in todo}, indent=2) + "\n",
                                        encoding="utf-8")
    state["pending_round"] = round_no
    state.setdefault("reviewed_ids", [])
    state["reviewed_ids"] = sorted(set(state["reviewed_ids"]) | {i["id"] for i in todo})
    state["packet_created"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_state(run, state)
    print(f"round {round_no}: {len(todo)} item(s) for the reviewer -> {folder / 'packet.jsonl'}")
    return 0


PROBLEM_CODES = frozenset({"does-not-support", "tier-too-low", "vested-interest", "output-differs", "not-atomic",
                           "weak-search", "missing-item", "bad-dismissal"})
REVIEWER_VERDICTS = frozenset({"correct", "wrong", "stale", "version-specific", "couldnt-verify"})
SAFE_NAME = re.compile(r"^S\d{3,}\.txt$")
ISO_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def reviewed_but_hidden(state: dict[str, Any], items: list[dict]) -> str | None:
    """A reviewed item may be fixed or downgraded, never deleted or taken out of scope."""
    by_id = {i.get("id"): i for i in items}
    for item_id in state.get("reviewed_ids", []):
        item = by_id.get(item_id)
        if item is None:
            return f"{item_id} was sent to the reviewer and is now missing from items.jsonl; restore it"
        if item.get("verdict") == "not-checked":
            return f"{item_id} was sent to the reviewer and is now not-checked; a reviewed item stays in scope"
    return None


def strength_gap(a: str | None, b: str | None) -> int:
    if a not in STRENGTHS or b not in STRENGTHS:
        return 0
    return abs(STRENGTHS.index(a) - STRENGTHS.index(b))


def cmd_merge(run: Path) -> int:
    state = load_state(run)
    round_no = state.get("pending_round")
    if not round_no:
        print("review_round: no pending round; run packet first", file=sys.stderr)
        return 2
    folder = run / "review" / f"round-{round_no}"
    packet = read_jsonl(folder / "packet.jsonl")
    hashes = json.loads((folder / "hashes.json").read_text(encoding="utf-8"))
    rows = read_jsonl(folder / "verdicts.jsonl")
    verdicts = {r["id"]: r for r in rows if r.get("id") != "new"}
    missed = [r for r in rows if r.get("id") == "new"]
    refetched = {r["url"]: r for r in read_jsonl(folder / "sources" / "index.jsonl")}

    created = state.get("packet_created", "")
    packet_ids = {e["id"] for e in packet}
    stray = [r for r in rows if r.get("id") != "new" and r.get("id") not in packet_ids]
    bad_codes = sorted({str(p.get("code")) for r in rows for p in (r.get("problems") or [])
                        if not isinstance(p, dict) or p.get("code") not in PROBLEM_CODES})
    skill_root = Path(load_meta(run)["skill_path"])
    skill_lines = {r.as_posix(): len((skill_root / r).read_text(encoding="utf-8", errors="replace").splitlines())
                   for r in iter_skill_files(skill_root)}
    incomplete = []
    for entry in packet:
        row = verdicts.get(entry["id"])
        if row is None or not isinstance(row.get("verdict"), str) or row["verdict"] not in REVIEWER_VERDICTS \
                or row.get("strength") not in STRENGTHS or not isinstance(row.get("problems", []), list):
            incomplete.append(f"{entry['id']}: needs a verdict ({', '.join(sorted(REVIEWER_VERDICTS))}), "
                              "a strength A to E, and problems as a list")
            continue
        if (row["verdict"] == "couldnt-verify") != (row["strength"] == "E"):
            incomplete.append(f"{entry['id']}: couldnt-verify goes with strength E, and only it")
        reran = {r.get("command") for r in row.get("reran", [])
                 if isinstance(r, dict) and str(r.get("output", "")).strip()}
        not_rerun = set(entry["commands"]) - reran
        if not_rerun:
            incomplete.append(f"{entry['id']}: commands not rerun with their output recorded: {sorted(not_rerun)}")
        for cited in entry["sources"]:
            copy = refetched.get(cited["url"])
            if copy is None:
                incomplete.append(f"{entry['id']}: source not refetched: {cited['url']}")
            elif not SAFE_NAME.match(str(copy.get("path", ""))) or not (folder / "sources" / copy["path"]).is_file():
                incomplete.append(f"{entry['id']}: the reviewer's copy of {cited['url']} is not a file fetched into "
                                  "its own folder")
            elif not ISO_TIME.match(str(copy.get("fetched_at", ""))) or copy["fetched_at"] < created:
                incomplete.append(f"{entry['id']}: {cited['url']} was fetched before the packet was made")
    if stray:
        incomplete.append(f"rows whose id is neither a packet id nor \"new\": {[r.get('id') for r in stray]}")
    if bad_codes:
        incomplete.append(f"unknown problem codes {bad_codes}; use {sorted(PROBLEM_CODES)}")
    for row in missed:
        line = row.get("line")
        if row.get("file") not in skill_lines or not isinstance(line, int) \
                or not 1 <= line <= max(skill_lines[row["file"]], 1) or not str(row.get("text", "")).strip():
            incomplete.append(f"a row with id \"new\" needs file (a file in the skill), a line inside it and text: "
                              f"{row}")
    if incomplete:
        print("review_round: review incomplete, send the reviewer back:\n  " + "\n  ".join(incomplete),
              file=sys.stderr)
        return 2

    item_list = read_jsonl(run / "items.jsonl")
    items = {i["id"]: i for i in item_list}
    feedback: list[dict[str, Any]] = []
    weakened = False
    for entry in packet:
        row, item = verdicts[entry["id"]], items[entry["id"]]
        problems = list(row.get("problems", []))
        for cited in entry["sources"]:
            copy = folder / "sources" / refetched[cited["url"]]["path"]
            if not quote_found(cited["quote"], copy.read_text(encoding="utf-8")):
                problems.append({"code": "quote-not-in-reviewer-copy", "detail": cited["url"]})
        if row["verdict"] != item["verdict"]:
            problems.append({"code": "verdict-mismatch",
                             "detail": f"auditor {item['verdict']}, reviewer {row['verdict']}"})
        gap = strength_gap(item.get("strength"), row.get("strength"))
        if gap > 1 and not problems:
            # Same verdict, no other problem: settle on the weaker strength instead of spending a round.
            weaker = max(item["strength"], row["strength"])
            item["review_note"] = (f"strength set to {weaker} in review round {round_no} "
                                   f"(auditor {item['strength']}, reviewer {row['strength']})")
            item["strength"] = weaker
            hashes[entry["id"]] = item_hash(item)
            weakened = True
        if problems:
            feedback.append({"id": entry["id"], "reviewer_verdict": row["verdict"],
                             "reviewer_strength": row.get("strength"), "problems": problems})
        else:
            state["accepted"][entry["id"]] = hashes[entry["id"]]
    if weakened:
        write_jsonl(run / "items.jsonl", item_list)
    for row in missed:
        feedback.append({"id": None, "file": row.get("file"), "line": row.get("line"), "text": row.get("text"),
                         "kind": row.get("kind"), "problems": row.get("problems") or [{"code": "missing-item"}]})
    write_jsonl(folder / "feedback.jsonl", feedback)

    state["rounds_done"] = round_no
    state["pending_round"] = None
    state["history"].append({"round": round_no, "reviewed": len(packet), "open": len(feedback)})
    if not feedback:
        state["status"] = "clean"
        state["final"] = run_snapshot(run)
        save_state(run, state)
        print(f"round {round_no}: clean, {len(packet)} item(s) reviewed")
        return 0
    if round_no < MAX_REVIEW_ROUNDS:
        save_state(run, state)
        print(f"round {round_no}: {len(feedback)} open problem(s) -> {folder / 'feedback.jsonl'}; fix each item or "
              f"downgrade it to couldnt-verify, then run packet again")
        return 1

    meta = load_meta(run)
    all_items = read_jsonl(run / "items.jsonl")
    by_id = {i["id"]: i for i in all_items}
    for fb in feedback:
        reviewer = {"verdict": fb.get("reviewer_verdict"), "strength": fb.get("reviewer_strength"),
                    "problems": fb["problems"]}
        if fb["id"] is None:
            in_scope = in_scope_by_meta(fb["file"], meta)
            new_id = f"R{round_no}-{len(all_items) + 1:04d}"
            while new_id in by_id:
                new_id += "x"
            new = {"id": new_id, "file": fb["file"], "line": fb["line"],
                   "kind": fb.get("kind") if fb.get("kind") in KINDS else "recommendation", "text": fb["text"],
                   "in_scope": in_scope is not False, "evidence": [], "searches": []}
            if in_scope is False:
                new.update({"verdict": "not-checked", "note": "reported missing by the reviewer in round "
                                                              f"{round_no}; outside the targeted paths"})
            else:
                new.update({"verdict": "disputed",
                            "dispute": {"auditor": {"verdict": "not inventoried"}, "reviewer": reviewer}})
                state.setdefault("disputed_ids", []).append(new_id)
            all_items.append(new)
            continue
        item = by_id[fb["id"]]
        item["dispute"] = {"auditor": {"verdict": item["verdict"], "strength": item.get("strength")},
                           "reviewer": reviewer}
        item["verdict"] = "disputed"
        state.setdefault("disputed_ids", []).append(item["id"])
    write_jsonl(run / "items.jsonl", all_items)
    save_state(run, state)
    state["status"] = "capped"
    state["final"] = run_snapshot(run)
    save_state(run, state)
    print(f"round {round_no}: cap reached; {len(feedback)} item(s) marked disputed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("packet", "merge"))
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        return cmd_packet(args.run_dir) if args.action == "packet" else cmd_merge(args.run_dir)
    except (FileNotFoundError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"review_round: cannot continue: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
