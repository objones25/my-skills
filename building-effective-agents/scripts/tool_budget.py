#!/usr/bin/env python3
"""Budget a tool result before it enters the model's context.

Wrap any tool that can return large text or record lists. The wrapper:
  1. caps the result at a token budget (rough estimate, see estimate_tokens),
  2. pages by line or record with an explicit offset,
  3. appends a visible, actionable hint so the agent knows it saw a slice and
     how to get the rest: "[showing 1-200 of 1,483 lines; call again with offset=200]".

Truncation the agent can't see is worse than none: the model treats the slice
as the whole and reports that later data "doesn't exist".

Usage
  python tool_budget.py FILE [--offset N] [--max-tokens N]
  from tool_budget import budget_text, budget_records
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Sequence

CHARS_PER_TOKEN = 4  # crude English average; use the provider's token counter when it matters


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


@dataclass
class Budgeted:
    text: str
    truncated: bool
    next_offset: int | None
    total: int


def budget_text(text: str, offset: int = 0, max_tokens: int = 2_000, unit: str = "lines",
                tool_hint: str = "call again with offset={next}") -> Budgeted:
    """Return lines [offset, ...) that fit in max_tokens, plus a recovery hint."""
    lines = text.splitlines()
    total = len(lines)
    if offset < 0 or (offset > total and total):
        raise ValueError(f"offset {offset} outside 0..{total}")
    out: list[str] = []
    used = 0
    i = offset
    while i < total:
        cost = estimate_tokens(lines[i]) + 1
        if out and used + cost > max_tokens:
            break
        out.append(lines[i])
        used += cost
        i += 1
    truncated = offset > 0 or i < total
    nxt = i if i < total else None
    body = "\n".join(out)
    if truncated:
        hint = f"[showing {unit} {offset + 1}-{i} of {total:,}"
        hint += f"; {tool_hint.format(next=nxt)}]" if nxt is not None else "; end reached]"
        body = f"{body}\n{hint}"
    return Budgeted(body, truncated, nxt, total)


def budget_records(records: Sequence[dict[str, Any]], fields: Sequence[str] | None = None,
                   offset: int = 0, limit: int = 20, max_tokens: int = 2_000) -> Budgeted:
    """Page a record list, keep only meaningful fields, cap by tokens.

    Pass `fields` to drop plumbing (UUIDs, mime types, internal URLs) the model
    can't use. Records are rendered one JSON object per line, which is easy to
    scan and cheap to parse back.
    """
    total = len(records)
    page = records[offset: offset + limit]
    if fields:
        page = [{f: r.get(f) for f in fields if f in r} for r in page]
    rendered = "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in page)
    b = budget_text(rendered, 0, max_tokens, unit="records")
    shown = len(b.text.splitlines()) - (1 if b.truncated else 0)
    end = offset + shown
    truncated = offset > 0 or end < total
    nxt = end if end < total else None
    body = "\n".join(b.text.splitlines()[:shown])
    if truncated:
        hint = f"[showing records {offset + 1}-{end} of {total:,}"
        hint += f"; call again with offset={nxt}]" if nxt is not None else "; end reached]"
        body = f"{body}\n{hint}"
    return Budgeted(body, truncated, nxt, total)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=2_000)
    a = p.parse_args(argv)
    b = budget_text(open(a.file, encoding="utf-8", errors="replace").read(), a.offset, a.max_tokens)
    print(b.text)


if __name__ == "__main__":
    main()
