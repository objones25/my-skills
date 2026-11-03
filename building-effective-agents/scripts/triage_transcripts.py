#!/usr/bin/env python3
"""Sort "agent said it worked" runs into causes, using the tool-call record.

Each cause needs a different fix, so count them before changing anything:

  never_called     the model narrated an action it never took         -> check tool-call record in harness
  silent_failure   the call returned 200 but the body carried an error -> parse bodies, raise on error payloads
  empty_result     a lookup returned nothing, the model reported data  -> sanctioned "not found" outcome
  wrong_id         the call worked, the reported identifier is not the one returned
                                                                       -> echo IDs from responses only
  partial_retry    the same action ran more than once                  -> idempotency keys, reconcile
  verified         claim matches a successful call and its identifier

Input: JSONL, one run per line:
{"run_id": "r1",
 "claim": {"action": "create_ticket", "id": "T-88"},          # what the final answer asserted
 "tool_calls": [{"name": "create_ticket", "args": {...},
                 "status": 200, "response": {"id": "T-88"}}]}

For lookups, set "claim": {"action": "get_order", "found": true}.

Usage
  python triage_transcripts.py runs.jsonl [--json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Iterable, Sequence

ERROR_KEYS = ("error", "errors", "fault", "exception")


def body_error(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("ok") is False or resp.get("success") is False:
        return True
    for k in ERROR_KEYS:
        v = resp.get(k)
        if v:  # non-empty error / errors array (GraphQL returns 200 with "errors")
            return True
    return False


def is_empty(resp: Any) -> bool:
    if resp is None or resp == {} or resp == [] or resp == "":
        return True
    if isinstance(resp, dict):
        for k in ("results", "items", "data", "records"):
            if k in resp and not resp[k]:
                return True
        if resp.get("found") is False:
            return True
    return False


def classify(run: dict[str, Any]) -> str:
    claim = run.get("claim") or {}
    action = claim.get("action")
    calls = [c for c in run.get("tool_calls", []) if c.get("name") == action]
    if not calls:
        return "never_called"
    ok = [c for c in calls if (c.get("status") or 200) < 400 and not body_error(c.get("response"))]
    if len(calls) > 1 and len({json.dumps(c.get("args"), sort_keys=True) for c in calls}) == 1:
        return "partial_retry"
    if not ok:
        return "silent_failure"
    last = ok[-1]
    if claim.get("found") and is_empty(last.get("response")):
        return "empty_result"
    claimed_id = claim.get("id")
    if claimed_id is not None:
        resp = last.get("response") or {}
        real = resp.get("id") if isinstance(resp, dict) else None
        if real is not None and str(real) != str(claimed_id):
            return "wrong_id"
    return "verified"


def triage(runs: Iterable[dict[str, Any]]) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for r in runs:
        label = classify(r)
        counts[label] += 1
        examples.setdefault(label, []).append(r.get("run_id", "?"))
    return counts, examples


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file"); p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    runs = [json.loads(l) for l in open(a.file) if l.strip()]
    counts, ex = triage(runs)
    if a.json:
        print(json.dumps({"counts": counts, "examples": ex}, indent=2))
        return
    for label, n in counts.most_common():
        print(f"{label:15s} {n:4d}   e.g. {', '.join(ex[label][:3])}")


if __name__ == "__main__":
    main()
