#!/usr/bin/env python3
"""Audit agent sessions against Meta's Agents Rule of Two.

A session should hold at most two of:
  A  processes untrustworthy input  (web pages, inbound email, user uploads, third-party tool output)
  B  reaches sensitive systems or private data
  C  changes state or communicates externally (including any outbound HTTP:
     image URLs, links, API parameters can all carry data out)

Input: a JSON manifest
{
  "tools": {
    "fetch_url":     {"A": true,  "C": true},
    "crm_lookup":    {"B": true},
    "send_reply":    {"C": true},
    "read_inbox":    {"A": true, "B": true}
  },
  "sessions": {
    "triage":   {"tools": ["read_inbox", "crm_lookup", "send_reply"], "human_approval_on": []},
    "drafter":  {"tools": ["crm_lookup"], "inputs_from": ["triage"]}
  }
}

`human_approval_on` lists tools gated by a human (or another reliable validator).
A session holding A+B+C passes only if every C tool it holds is gated.
`inputs_from` marks free-text flowing from another session; if that session
held A, this one inherits A unless "structured_handoff": true (fixed schema, no
free text), which is the point of splitting.

Usage
  python rule_of_two_audit.py manifest.json
Exit code 1 when any session violates the rule.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

PROPS = ("A", "B", "C")


@dataclass
class SessionReport:
    name: str
    props: dict[str, list[str]]       # property -> tools that confer it
    inherited_a_from: list[str]
    ungated_c: list[str]
    violation: bool

    def summary(self) -> str:
        held = "".join(p for p in PROPS if self.props[p]) or "none"
        lines = [f"{self.name}: holds {held}" + ("  VIOLATION" if self.violation else "")]
        for p in PROPS:
            if self.props[p]:
                lines.append(f"  {p} via {', '.join(self.props[p])}")
        if self.inherited_a_from:
            lines.append(f"  A inherited (free-text input) from {', '.join(self.inherited_a_from)}")
        if self.violation:
            lines.append(f"  fix: gate {', '.join(self.ungated_c)} with human approval, or split so no "
                         "session holds A, B and C; pass results between sessions with a fixed schema")
        return "\n".join(lines)


def audit(manifest: dict[str, Any]) -> list[SessionReport]:
    tools = manifest["tools"]
    sessions = manifest["sessions"]
    unknown = {t for s in sessions.values() for t in s.get("tools", []) if t not in tools}
    if unknown:
        raise ValueError(f"tools not declared in manifest: {sorted(unknown)}")

    def direct(name: str) -> dict[str, list[str]]:
        props = {p: [] for p in PROPS}
        for t in sessions[name].get("tools", []):
            for p in PROPS:
                if tools[t].get(p):
                    props[p].append(t)
        return props

    reports = []
    for name, s in sessions.items():
        props = direct(name)
        inherited = []
        if not s.get("structured_handoff"):
            for src in s.get("inputs_from", []):
                if direct(src)["A"]:
                    inherited.append(src)
        has_a = bool(props["A"] or inherited)
        gated = set(s.get("human_approval_on", []))
        ungated_c = [t for t in props["C"] if t not in gated]
        violation = has_a and bool(props["B"]) and bool(ungated_c)
        reports.append(SessionReport(name, props, inherited, ungated_c, violation))
    return reports


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    reports = audit(json.load(open(argv[0])))
    for r in reports:
        print(r.summary())
    return 1 if any(r.violation for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
