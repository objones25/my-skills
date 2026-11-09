#!/usr/bin/env python3
"""Check every rule of a finished audit that a script can check, before review and before the report.

Usage:
    python validate_audit.py RUN_DIR [--json]

These rules are necessary, not sufficient: a script cannot tell whether a command
really tests the claim or a quote really entails it. The blind reviewer judges that.

Codes (references/formats.md explains each and its fix):
    AS002  the audited skill changed since start_audit.py (the audit must be read-only)
    AS003  candidates.jsonl missing (find_candidates.py not run)
    AS010  a candidate line is covered by no item and no dismissal
    AS011  an item, dismissal or evidence entry breaks the schema, or points at a place not in the skill
    AS012  scope mismatch: in-scope item left not-checked, or out-of-scope item given a verdict
    AS013  definite verdict (correct, wrong, stale, version-specific) with no evidence
    AS014  evidence cites a source id that is not in sources/index.jsonl, or its file is missing
    AS015  quoted passage not found in the saved source text
    AS016  quote too short to be specific
    AS017  a tier 5 (lead only) or tier 6 (excluded) source cited as evidence
    AS018  verdict and strength disagree (couldnt-verify needs E; definite verdicts need A to D)
    AS019  cited sources or couldnt-verify without at least two logged searches that name the item
    AS020  strength A without a command, or without two distinct sources of tier 1 or 2
    AS021  command evidence with no command or no captured output
    AS023  cited sources or couldnt-verify with no disconfirming search
    AS024  disputed item without both positions, or not marked disputed by review_round.py
    AS025  search_log.jsonl entry breaks the schema, or reuses an id
    AS026  a dismissal hides a script, test, example, asset or data file, or a whole text
           file (only binary files, licences and lockfiles may be dismissed whole)
    AS027  strength higher than the best source allows (tier 1-2: B, tier 3: C, tier 4: D; one
           non-tier-1 source caps an empirical claim or recommendation at C)

Exit 0 no findings, 1 findings, 2 cannot check (no meta.json or unreadable JSONL).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from audit_common import (
    DEFINITE_VERDICTS,
    KINDS,
    SEARCH_PURPOSES,
    STRENGTHS,
    TIERS,
    VERDICTS,
    iter_skill_files,
    load_meta,
    read_jsonl,
    tree_hash,
)
from quote_match import quote_found, quote_problem

TEXT_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".rst", ".txt"})
UNDISMISSABLE = frozenset({"script", "test", "example", "asset", "data"})
WHOLE_FILE_OK = re.compile(r"^(?:LICEN[CS]E|COPYING|NOTICE)(?:\.\w+)?$|\.lock$|^package-lock\.json$", re.IGNORECASE)
TIER_CEILING = {1: "B", 2: "B", 3: "C", 4: "D"}
MAX_TEXT_SPAN = 60


@dataclass(frozen=True)
class Finding:
    code: str
    where: str
    message: str


def in_scope_by_meta(file: str, meta: dict[str, Any]) -> bool | None:
    """For targeted scope, whether a file is inside the named paths; None when scope does not decide."""
    scope = meta["scope"]
    if scope["mode"] != "targeted":
        return None
    return any(file == p.rstrip("/") or file.startswith(p.rstrip("/") + "/") for p in scope["paths"])


def weaker(strength: str, ceiling: str) -> bool:
    """True if strength is above the ceiling (A is strongest)."""
    return STRENGTHS.index(strength) < STRENGTHS.index(ceiling)


def line_counts(skill: Path) -> dict[str, int | None]:
    """Line count per skill file, None for files that are not text."""
    counts: dict[str, int | None] = {}
    for rel in iter_skill_files(skill):
        try:
            counts[rel.as_posix()] = len((skill / rel).read_text(encoding="utf-8").splitlines())
        except (UnicodeDecodeError, OSError):
            counts[rel.as_posix()] = None
    return counts


def schema_problems(item: dict[str, Any], files: dict[str, int | None]) -> list[str]:
    problems = []
    if not isinstance(item.get("file"), str) or not isinstance(item.get("line"), int):
        return ["file must be a string and line an integer"]
    if item["file"] not in files:
        problems.append(f"{item['file']} is not a file in the skill")
    end = item.get("end_line", item["line"])
    if not isinstance(end, int) or end < item["line"] or item["line"] < 1:
        problems.append("end_line must be an integer no smaller than line, and line at least 1")
    elif item["file"] in files:
        count = files[item["file"]]
        if count is not None and end > max(count, 1):
            problems.append(f"line range {item['line']}-{end} is past the end of the file ({count} lines)")
        if Path(item["file"]).suffix.lower() in TEXT_SUFFIXES and end - item["line"] > MAX_TEXT_SPAN:
            problems.append(f"an item may span at most {MAX_TEXT_SPAN} lines of a text file; split it")
    if item.get("kind") not in KINDS:
        problems.append(f"kind must be one of {sorted(KINDS)}")
    if item.get("verdict") not in VERDICTS:
        problems.append(f"verdict must be one of {sorted(VERDICTS)}")
    if not isinstance(item.get("text"), str) or not item["text"].strip():
        problems.append("text (the atomic claim) is required")
    if not isinstance(item.get("in_scope"), bool):
        problems.append("in_scope must be true or false")
    evidence, searches = item.get("evidence", []), item.get("searches", [])
    if not isinstance(evidence, list) or not all(isinstance(ev, dict) for ev in evidence):
        problems.append("evidence must be a list of objects")
    if not isinstance(searches, list) or not all(isinstance(s, str) for s in searches):
        problems.append("searches must be a list of search ids")
    return problems


def check_evidence(run: Path, item: dict[str, Any], where: str,
                   sources: dict[str, dict]) -> tuple[list[Finding], int, list[int]]:
    """Findings, the number of commands, and the best tier per distinct cited source."""
    out: list[Finding] = []
    commands, tiers = 0, []
    best: dict[str, int] = {}
    for ev in item.get("evidence", []):
        kind = ev.get("type")
        if kind == "command":
            if not str(ev.get("command", "")).strip() or not str(ev.get("output", "")).strip():
                out.append(Finding("AS021", where, "command evidence needs command and output"))
            else:
                commands += 1
        elif kind == "source":
            tier = ev.get("tier")
            if tier not in TIERS:
                out.append(Finding("AS011", where, "source evidence needs tier 1 to 6"))
                continue
            if tier >= 5:
                out.append(Finding("AS017", where, f"tier {tier} sources are {'leads only' if tier == 5 else 'excluded'}"
                                   " and cannot be cited as evidence"))
            source = sources.get(str(ev.get("source_id")))
            if source is None or not (run / "sources" / source["path"]).is_file():
                out.append(Finding("AS014", where, f"source {ev.get('source_id')} not fetched into sources/"))
                continue
            quote = str(ev.get("quote", ""))
            weak = quote_problem(quote)
            if weak:
                out.append(Finding("AS016", where, weak))
            elif not quote_found(quote, (run / "sources" / source["path"]).read_text(encoding="utf-8")):
                out.append(Finding("AS015", where, f"quote not in {source['id']}: {quote[:80]!r}"))
            best[source["id"]] = min(tier, best.get(source["id"], tier))
        else:
            out.append(Finding("AS011", where, "evidence type must be command or source"))
    tiers = list(best.values())
    return out, commands, tiers


def check_items(run: Path, meta: dict[str, Any], items: list[dict], sources: dict[str, dict],
                searches: dict[str, dict], files: dict[str, int | None]) -> list[Finding]:
    out: list[Finding] = []
    state_path = run / "review" / "state.json"
    disputed_ids = set(json.loads(state_path.read_text(encoding="utf-8")).get("disputed_ids", [])) \
        if state_path.is_file() else set()
    seen: set[str] = set()
    for item in items:
        where = f"{item.get('id', '?')} {item.get('file', '?')}:{item.get('line', '?')}"
        problems = schema_problems(item, files)
        if not isinstance(item.get("id"), str) or item["id"] in seen:
            problems.append("id missing or duplicated")
        seen.add(str(item.get("id")))
        if problems:
            out.extend(Finding("AS011", where, p) for p in problems)
            continue

        verdict, strength = item["verdict"], item.get("strength")
        scoped = in_scope_by_meta(item["file"], meta)
        if scoped is not None and scoped != item["in_scope"]:
            out.append(Finding("AS012", where, f"targeted scope says in_scope={scoped}"))
        if item["in_scope"] and verdict == "not-checked":
            out.append(Finding("AS012", where, "in scope but not checked"))
        if not item["in_scope"] and verdict != "not-checked":
            out.append(Finding("AS012", where, "out of scope, so the verdict must be not-checked"))

        if verdict == "couldnt-verify" and strength != "E":
            out.append(Finding("AS018", where, "couldnt-verify needs strength E"))
        if verdict in DEFINITE_VERDICTS and strength not in STRENGTHS[:4]:
            out.append(Finding("AS018", where, f"{verdict} needs strength A, B, C or D"))
        if verdict == "not-checked" and strength is not None:
            out.append(Finding("AS018", where, "not-checked items carry no strength"))
        if verdict in DEFINITE_VERDICTS and not item.get("evidence"):
            out.append(Finding("AS013", where, f"{verdict} with no evidence"))
        if verdict == "disputed":
            dispute = item.get("dispute")
            if not isinstance(dispute, dict) or not dispute.get("auditor") or not dispute.get("reviewer"):
                out.append(Finding("AS024", where, "disputed needs dispute.auditor and dispute.reviewer"))
            elif item["id"] not in disputed_ids:
                out.append(Finding("AS024", where, "only review_round.py merge may mark an item disputed"))
            continue

        found, commands, tiers = check_evidence(run, item, where, sources)
        out += found
        if strength == "A" and not commands and sum(t <= 2 for t in tiers) < 2:
            out.append(Finding("AS020", where, "A needs execution or two tier 1-2 sources"))
        if strength in STRENGTHS[:4] and tiers and not commands:
            ceiling = TIER_CEILING.get(min(tiers), "D")
            if item["kind"] in ("empirical", "recommendation") and len(tiers) == 1 and tiers[0] != 1:
                ceiling = max(ceiling, "C")
            if weaker(strength, ceiling) and not (strength == "A" and sum(t <= 2 for t in tiers) >= 2):
                out.append(Finding("AS027", where, f"strength {strength} is above {ceiling}, the most this "
                                                   f"evidence allows"))

        needs_search = bool(tiers) or (verdict == "couldnt-verify" and not commands)
        if needs_search and item["in_scope"]:
            ids = item.get("searches", [])
            known = [searches[i] for i in ids if i in searches]
            naming = [s for s in known if item["id"] in s.get("item_ids", [])]
            queries = {s["query"].strip().casefold() for s in naming}
            if len(known) != len(ids) or len(naming) < 2 or len(queries) < 2:
                out.append(Finding("AS019", where, "needs at least two logged searches with different queries, "
                                                   "each listing this item in item_ids"))
            elif not any(s.get("purpose") == "disconfirm" for s in naming):
                out.append(Finding("AS023", where, "no disconfirming search logged"))
    return out


def check_coverage(candidates: list[dict], items: list[dict], dismissed: list[dict],
                   files: dict[str, int | None]) -> list[Finding]:
    out: list[Finding] = []
    item_lines: dict[str, set[int]] = {}
    for item in items:
        line, end = item.get("line"), item.get("end_line", item.get("line"))
        if isinstance(line, int) and isinstance(end, int) and isinstance(item.get("file"), str) and end >= line:
            item_lines.setdefault(item["file"], set()).update(range(line, end + 1))
    dismissed_lines: dict[str, set[int]] = {}
    whole: set[str] = set()
    for n, row in enumerate(dismissed, start=1):
        where = f"dismissed.jsonl:{n}"
        if not str(row.get("reason", "")).strip() or not isinstance(row.get("file"), str):
            out.append(Finding("AS011", where, "a dismissal needs file and a reason"))
        elif row["file"] not in files:
            out.append(Finding("AS011", where, f"{row['file']} is not a file in the skill"))
        elif row.get("all") is True:
            if not (files[row["file"]] is None or WHOLE_FILE_OK.search(Path(row["file"]).name)):
                out.append(Finding("AS026", where, f"{row['file']} holds text; only binary files, licences and "
                                                   "lockfiles may be dismissed whole"))
            whole.add(row["file"])
        elif isinstance(row.get("lines"), list) and all(isinstance(x, int) for x in row["lines"]):
            dismissed_lines.setdefault(row["file"], set()).update(row["lines"])
        else:
            out.append(Finding("AS011", where, "a dismissal needs lines (integers) or all: true"))
    for cand in candidates:
        where = f"{cand['id']} {cand['file']}:{cand['line']}"
        span = set(range(cand["line"], cand.get("end_line", cand["line"]) + 1))
        by_item = bool(span & item_lines.get(cand["file"], set()))
        by_dismissal = cand["file"] in whole or cand["line"] in dismissed_lines.get(cand["file"], set())
        if by_dismissal and not by_item and UNDISMISSABLE & set(cand["reasons"]):
            out.append(Finding("AS026", where, f"a {cand['reasons'][0]} must be inventoried, not dismissed"))
        if not by_item and not by_dismissal:
            out.append(Finding("AS010", where, f"not covered by an item or dismissal ({', '.join(cand['reasons'])}): "
                                               f"{cand['text'][:80]}"))
    return out


def check_searches(searches: list[dict]) -> list[Finding]:
    out = []
    seen: set[str] = set()
    for n, row in enumerate(searches, start=1):
        ok = (isinstance(row.get("id"), str) and isinstance(row.get("query"), str) and row["query"].strip()
              and isinstance(row.get("engine"), str) and isinstance(row.get("date"), str)
              and isinstance(row.get("hits"), int) and isinstance(row.get("opened"), list)
              and isinstance(row.get("item_ids", []), list) and row.get("purpose") in SEARCH_PURPOSES)
        if not ok:
            out.append(Finding("AS025", f"search_log.jsonl:{n}",
                               f"needs id, engine, query, date, hits (int), opened (list), item_ids (list), "
                               f"purpose in {sorted(SEARCH_PURPOSES)}"))
        elif row["id"] in seen:
            out.append(Finding("AS025", f"search_log.jsonl:{n}", f"id {row['id']} is used twice"))
        seen.add(str(row.get("id")))
    return out


def validate(run: Path) -> list[Finding]:
    meta = load_meta(run)
    findings: list[Finding] = []
    skill = Path(meta["skill_path"])
    if not skill.is_dir() or tree_hash(skill) != meta["tree_hash"]:
        findings.append(Finding("AS002", str(skill), "the audited skill changed or vanished during the audit"))
    if not (run / "candidates.jsonl").is_file():
        findings.append(Finding("AS003", str(run), "run find_candidates.py first"))
    items = read_jsonl(run / "items.jsonl")
    searches = read_jsonl(run / "search_log.jsonl")
    sources = {row["id"]: row for row in read_jsonl(run / "sources" / "index.jsonl")}
    files = line_counts(skill) if skill.is_dir() else {}
    findings += check_searches(searches)
    findings += check_coverage(read_jsonl(run / "candidates.jsonl"), items, read_jsonl(run / "dismissed.jsonl"), files)
    findings += check_items(run, meta, items, sources,
                            {s["id"]: s for s in searches if isinstance(s.get("id"), str)}, files)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        findings = validate(args.run_dir)
    except (FileNotFoundError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"validate_audit: cannot check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"findings": [asdict(f) for f in findings]}, indent=2))
    else:
        for f in findings:
            print(f"{f.code}  {f.where}: {f.message}")
        print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
