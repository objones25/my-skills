#!/usr/bin/env python3
"""List every line or file the inventory must account for, as RUN/candidates.jsonl.

Usage:
    python find_candidates.py RUN_DIR

This is a recall aid, not a claim extractor, so it over-collects: every non-blank
line of prose in a text file (.md, .mdx, .rst, .txt) is a candidate, tagged with what
it holds (url, version, number, code, citation, absolute, self-claim) or "prose";
the frontmatter description is one candidate across all its lines; each fenced code
block is one candidate spanning its lines; headings and table rules are skipped. Every other file is one file-level candidate,
tagged script, test, example, asset or eval by folder, else script (code suffix or
shebang), data (other text, such as YAML or JSON) or binary. validate_audit.py fails until each
candidate is covered by an item or dismissed with a reason.

Exit 0 on success, 2 if RUN_DIR is not a started audit.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from audit_common import is_text, iter_skill_files, load_meta, write_jsonl

PATTERNS = {
    "url": re.compile(r"https?://"),
    "version": re.compile(r"\b\d+\.\d+(?:\.\d+)?\b"),
    "number": re.compile(r"\d"),
    "code": re.compile(r"`[^`]+`"),
    "citation": re.compile(r"\bet al\.|\(\d{4}\)|\b(?:19|20)\d{2}\b|\[[A-Z][A-Za-z]+\d{2}"),
    "absolute": re.compile(
        r"\b(?:always|never|must|should|only|every|none|faster|slower|better|best|worse|worst|"
        r"recommend\w*|avoid|default\w*|prefer\w*|deprecated|removed|requires?)\b", re.IGNORECASE),
    "self-claim": re.compile(r"\b(?:tested|verified|passes|passed|clean|checks|audited)\b", re.IGNORECASE),
}
TOP_DIR_KIND = {"scripts": "script", "tests": "test", "selftest": "test", "assets": "asset",
                "examples": "example", "evals": "eval"}
TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


TEXT_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".rst", ".txt"})
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def markdown_candidates(rel: Path, text: str) -> list[dict]:
    out: list[dict] = []
    name = rel.as_posix()
    lines = text.splitlines()
    front_end = 0
    if lines and lines[0].strip() == "---":
        front_end = next((n for n, line in enumerate(lines[1:], start=2) if line.strip() == "---"), 0)
    number = 1
    while front_end and number < front_end:
        line = lines[number - 1]
        if line.strip().startswith("description:"):
            end = number
            while end < front_end - 1 and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
                end += 1
            out.append({"file": name, "line": number, "end_line": end, "reasons": ["frontmatter"],
                        "text": " ".join(x.strip() for x in lines[number - 1:end])[:300]})
            number = end
        number += 1
    fence: tuple[str, int, int] | None = None  # (char, length, start line)
    for number, line in enumerate(lines, start=1):
        if number <= front_end:
            continue
        stripped = line.strip()
        marker = FENCE.match(line)
        if fence is not None:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= fence[1] \
                    and not stripped[len(marker.group(1)):].strip():
                out.append({"file": name, "line": fence[2], "end_line": number, "reasons": ["code-block"],
                            "text": lines[fence[2] - 1].strip()[:300]})
                fence = None
            continue
        if marker:
            fence = (marker.group(1)[0], len(marker.group(1)), number)
            continue
        if not stripped or stripped.startswith("#") or TABLE_RULE.match(stripped):
            continue
        reasons = [label for label, pattern in PATTERNS.items() if pattern.search(line)] or ["prose"]
        out.append({"file": name, "line": number, "end_line": number, "reasons": reasons, "text": stripped[:300]})
    if fence is not None:
        out.append({"file": name, "line": fence[2], "end_line": len(lines), "reasons": ["code-block"],
                    "text": "unclosed fence"})
    return out


CODE_SUFFIXES = frozenset({".py", ".sh", ".bash", ".zsh", ".fish", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".rb",
                           ".go", ".rs", ".pl", ".ps1", ".lua", ".r", ".jl", ".java", ".kt", ".c", ".h", ".cpp"})


def file_kind(skill: Path, rel: Path) -> str:
    """script, test, example, asset or eval by folder; else by content: script, data or binary."""
    if len(rel.parts) > 1 and rel.parts[0] in TOP_DIR_KIND:
        return TOP_DIR_KIND[rel.parts[0]]
    if rel.name.startswith("test_") and rel.suffix == ".py":
        return "test"
    path = skill / rel
    if rel.suffix.lower() in CODE_SUFFIXES or (is_text(path) and path.read_text(encoding="utf-8")[:2] == "#!"):
        return "script"
    return "data" if is_text(path) else "binary"


def find(skill: Path) -> list[dict]:
    rows: list[dict] = []
    for rel in iter_skill_files(skill):
        if rel.suffix.lower() in TEXT_SUFFIXES:
            rows.extend(markdown_candidates(rel, (skill / rel).read_text(encoding="utf-8", errors="replace")))
        else:
            kind = file_kind(skill, rel)
            rows.append({"file": rel.as_posix(), "line": 1, "end_line": 1, "reasons": [kind], "text": rel.name})
    for n, row in enumerate(rows, start=1):
        row["id"] = f"C{n:04d}"
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        meta = load_meta(args.run_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"find_candidates: {exc}", file=sys.stderr)
        return 2
    rows = find(Path(meta["skill_path"]))
    write_jsonl(args.run_dir / "candidates.jsonl", rows)
    per_file = Counter(row["file"] for row in rows)
    for name, count in sorted(per_file.items()):
        print(f"{count:5d}  {name}")
    print(f"{len(rows)} candidates in {len(per_file)} files -> {args.run_dir / 'candidates.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
