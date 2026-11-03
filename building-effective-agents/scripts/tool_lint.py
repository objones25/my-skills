#!/usr/bin/env python3
"""Lint a set of tool definitions for the problems that cause wrong-tool picks.

Accepts a JSON file holding a list of tool definitions in Anthropic
(`input_schema`), OpenAI function (`parameters`, optionally nested under
`function`), or MCP (`inputSchema`, `annotations`) shape.

Checks
  overlap        pairs of tools whose names or descriptions share most content words
                 (the "would two engineers pick the same tool?" test, approximated)
  boundary       overlapping tools whose descriptions never say when NOT to use them
  thin-desc      descriptions under 15 words
  vague-param    parameter names like `id`, `user`, `data` that invite the wrong value
  write-unmarked tools whose names suggest side effects but carry no MCP annotations
  catalog-size   total definition size past the point where on-demand loading
                 usually pays (roughly 10 tools or 10K tokens, Anthropic's guideline)

This is a heuristic screen, not a verdict. Confirm overlap with a tool-selection
eval on real tasks before merging or renaming tools.

Usage
  python tool_lint.py tools.json [--overlap 0.5] [--json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Sequence

STOP = set("""a an the of to for and or in on with by from at is are be this that it its
as use used using call returns return get given into via when your you can will all any
""".split())
VAGUE_PARAMS = {"id", "user", "data", "value", "input", "item", "obj", "object", "thing", "args", "info"}
WRITE_WORDS = {"delete", "remove", "send", "create", "update", "write", "post", "drop",
               "pay", "refund", "transfer", "deploy", "merge", "publish", "execute", "run", "kill"}
BOUNDARY_RE = re.compile(r"\b(do not use|don't use|not for|instead use|use .* instead|only when|only for)\b", re.I)


@dataclass
class Finding:
    check: str
    tools: list[str]
    detail: str


def normalize(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function", tool)
    schema = fn.get("input_schema") or fn.get("parameters") or fn.get("inputSchema") or {}
    return {
        "name": fn.get("name", "?"),
        "description": fn.get("description", "") or "",
        "params": list((schema.get("properties") or {}).keys()),
        "annotations": tool.get("annotations") or fn.get("annotations"),
        "raw": tool,
    }


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def name_parts(name: str) -> set[str]:
    return {p for p in re.split(r"[_\-.\s]+|(?<=[a-z])(?=[A-Z])", name.lower()) if p}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def lint(tools: Sequence[dict[str, Any]], overlap: float = 0.5) -> tuple[list[Finding], dict]:
    ts = [normalize(t) for t in tools]
    findings: list[Finding] = []
    for a, b in itertools.combinations(ts, 2):
        d = jaccard(words(a["description"]), words(b["description"]))
        n = jaccard(name_parts(a["name"]), name_parts(b["name"]))
        score = max(d, n)
        if score >= overlap:
            findings.append(Finding("overlap", [a["name"], b["name"]],
                                    f"description similarity {d:.2f}, name similarity {n:.2f}"))
            for t in (a, b):
                if not BOUNDARY_RE.search(t["description"]):
                    findings.append(Finding("boundary", [t["name"]],
                                            f"overlaps {b['name'] if t is a else a['name']} but never says when not to use it"))
    for t in ts:
        nwords = len(t["description"].split())
        if nwords < 15:
            findings.append(Finding("thin-desc", [t["name"]], f"{nwords} words"))
        vague = [p for p in t["params"] if p.lower() in VAGUE_PARAMS]
        if vague:
            findings.append(Finding("vague-param", [t["name"]], f"rename {vague} (e.g. user -> user_id)"))
        if name_parts(t["name"]) & WRITE_WORDS and not t["annotations"]:
            findings.append(Finding("write-unmarked", [t["name"]],
                                    "side-effect name with no annotations (destructiveHint / idempotentHint / openWorldHint)"))
    def_tokens = sum(len(json.dumps(t["raw"])) // 4 for t in ts)
    stats = {"tools": len(ts), "definition_tokens_est": def_tokens}
    if len(ts) > 10 or def_tokens > 10_000:
        findings.append(Finding("catalog-size", [], f"{len(ts)} tools, ~{def_tokens:,} definition tokens: "
                                "consider on-demand tool loading, keep the 3-5 most used always loaded"))
    # dedupe boundary findings
    seen, uniq = set(), []
    for f in findings:
        key = (f.check, tuple(f.tools))
        if key not in seen:
            seen.add(key); uniq.append(f)
    return uniq, stats


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file"); p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    data = json.load(open(a.file))
    tools = data["tools"] if isinstance(data, dict) and "tools" in data else data
    findings, stats = lint(tools, a.overlap)
    if a.json:
        print(json.dumps({"stats": stats, "findings": [asdict(f) for f in findings]}, indent=2))
    else:
        print(f"{stats['tools']} tools, ~{stats['definition_tokens_est']:,} definition tokens")
        for f in findings:
            print(f"[{f.check}] {', '.join(f.tools)}: {f.detail}")
        if not findings:
            print("no findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
