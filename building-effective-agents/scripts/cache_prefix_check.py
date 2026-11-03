#!/usr/bin/env python3
"""Find what breaks prompt-cache reuse across consecutive model calls in one run.

Prompt caches reuse the longest identical prefix of a request. In an agent loop
each call should extend the previous one (append-only), so call i+1 should
share almost all of call i as its prefix. When it doesn't, something early in
the request changed: a timestamp in the system prompt, tools re-ordered or
added mid-run, an edited earlier message, non-deterministic JSON key order.

Input: JSONL, one request per line, in call order, each with any of
  {"system": "...", "tools": [...], "messages": [...]}
(the exact body you send the provider, minus credentials).

Output per transition: shared-prefix fraction of the earlier request, and the
first differing section (system / tools / messages[i]) with a short diff.

Usage
  python cache_prefix_check.py requests.jsonl [--threshold 0.9]
Exit code 1 when any transition shares less than the threshold.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Sequence

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?|\b\d{10,13}\b")


def canonical(req: dict[str, Any]) -> list[tuple[str, str]]:
    """Request as ordered (section, text) pairs.

    Uses Anthropic's cache order (tools, then system, then messages); reorder for other providers.
    """
    parts: list[tuple[str, str]] = []
    if "tools" in req:
        parts.append(("tools", json.dumps(req["tools"], ensure_ascii=False)))
    if "system" in req:
        parts.append(("system", req["system"] if isinstance(req["system"], str)
                      else json.dumps(req["system"], ensure_ascii=False)))
    for i, m in enumerate(req.get("messages", [])):
        parts.append((f"messages[{i}]", json.dumps(m, ensure_ascii=False)))
    return parts


def shared_prefix(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> tuple[int, str | None, str]:
    """Characters of a's serialization that b repeats exactly, and where they diverge."""
    n = 0
    for (sa, ta), (sb, tb) in zip(a, b):
        if sa != sb or ta != tb:
            k = 0
            while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
                k += 1
            ctx = f"...{ta[max(0, k - 30):k + 40]!r} -> ...{tb[max(0, k - 30):k + 40]!r}"
            return n + k, sa, ctx
        n += len(ta)
    return n, None, ""


def diagnose(section: str | None, ctx: str, a: dict, b: dict) -> str:
    if section is None:
        return ""
    if section == "tools":
        na = [t.get("name") for t in a.get("tools", [])]
        nb = [t.get("name") for t in b.get("tools", [])]
        if sorted(na) == sorted(nb) and na != nb:
            return "tool order changed: list tools in a fixed order"
        if na != nb:
            return "tool set changed mid-run: mask tools instead of adding/removing them"
        return "tool definitions edited mid-run"
    if section == "system" and TIMESTAMP.search(ctx):
        return "timestamp in system prompt: move it to the end or drop it"
    if section.startswith("messages"):
        return "an earlier message was edited: append instead (or accept the cache miss knowingly, e.g. compaction)"
    return "early content changed"


def check(reqs: Sequence[dict[str, Any]], threshold: float = 0.9) -> list[dict[str, Any]]:
    out = []
    for i in range(len(reqs) - 1):
        a, b = canonical(reqs[i]), canonical(reqs[i + 1])
        total = sum(len(t) for _, t in a) or 1
        n, section, ctx = shared_prefix(a, b)
        frac = n / total
        out.append({"from": i, "to": i + 1, "shared": round(frac, 3), "ok": frac >= threshold,
                    "first_diff": section, "context": ctx,
                    "likely_cause": diagnose(section, ctx, reqs[i], reqs[i + 1]) if frac < threshold else ""})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file"); p.add_argument("--threshold", type=float, default=0.9)
    a = p.parse_args(argv)
    reqs = [json.loads(l) for l in open(a.file) if l.strip()]
    rows = check(reqs, a.threshold)
    for r in rows:
        flag = "ok " if r["ok"] else "LOW"
        line = f"{flag} call {r['from']}->{r['to']}: {r['shared']:.0%} shared"
        if not r["ok"]:
            line += f"; diverges in {r['first_diff']}: {r['likely_cause']}\n     {r['context']}"
        print(line)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
