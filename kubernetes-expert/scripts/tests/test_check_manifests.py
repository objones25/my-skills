#!/usr/bin/env python3
"""Regression tests for check_manifests.py. Run: python3 scripts/tests/test_check_manifests.py
(also collected by pytest). Each object must produce exactly the expected rule IDs, and the
shipped templates must stay free of errors and warnings."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import check_manifests as C  # noqa: E402


def test_fixtures() -> None:
    with open(os.path.join(HERE, "expected.json"), encoding="utf-8") as fh:
        expected = json.load(fh)
    failures = []
    for fname, per_obj in expected.items():
        findings = C.check(C.load([os.path.join(HERE, "fixtures", fname)]))
        got: dict[str, set] = {}
        for f in findings:
            got.setdefault(f.object, set()).add(f.rule)
        for obj, rules in per_obj.items():
            g, w = got.get(obj, set()), set(rules)
            if g != w:
                failures.append(f"{fname} {obj}: missing={sorted(w - g)} unexpected={sorted(g - w)}")
    assert not failures, "\n".join(failures)


def test_templates_clean() -> None:
    paths = [os.path.join(SKILL, "assets", p) for p in ("app", "job.yaml", "httproute.yaml")]
    bad = [f for f in C.check(C.load(paths)) if f.severity != "info"]
    assert not bad, "\n".join(f"{f.object} {f.rule} {f.message}" for f in bad)


def test_every_rule_is_exercised() -> None:
    with open(os.path.join(HERE, "expected.json"), encoding="utf-8") as fh:
        covered = {r for per in json.load(fh).values() for rules in per.values() for r in rules}
    covered |= {"K022"}
    untested = set(C.RULES) - covered
    assert not untested, f"rules with no fixture: {sorted(untested)}"





def test_pss_matches_recorded_apiserver_verdicts() -> None:
    with open(os.path.join(HERE, "pss_server_verdicts.json"), encoding="utf-8") as fh:
        recorded = json.load(fh)["verdicts"]
    mismatches = []
    for fname, server in recorded.items():
        objs = C.load([os.path.join(HERE, "pss_cases", fname)])
        script = "reject" if any(f.rule == "K009" for f in C.check(objs)) else "admit"
        if script != server:
            mismatches.append(f"{fname}: apiserver={server} script={script}")
    assert not mismatches, "\n".join(mismatches)


if __name__ == "__main__":
    for t in (test_fixtures, test_templates_clean, test_every_rule_is_exercised,
              test_pss_matches_recorded_apiserver_verdicts):
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}\n{e}")
            sys.exit(1)
