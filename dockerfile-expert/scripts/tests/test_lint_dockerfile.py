#!/usr/bin/env python3
"""Regression tests for lint_dockerfile.py. Run: python3 scripts/tests/test_lint_dockerfile.py
(also collected by pytest). Each fixture must produce exactly the expected set of rule IDs."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lint_dockerfile as L  # noqa: E402


def _rules_for(name: str) -> set[str]:
    path = os.path.join(HERE, "fixtures", name)
    with open(path, encoding="utf-8") as fh:
        findings = L.lint(path, fh.read(), context_dir=None)
    return {f.rule for f in findings}


def test_fixtures() -> None:
    with open(os.path.join(HERE, "expected.json"), encoding="utf-8") as fh:
        expected = json.load(fh)
    failures = []
    for name, rules in expected.items():
        got = _rules_for(name)
        want = set(rules)
        if got != want:
            failures.append(f"{name}: missing={sorted(want - got)} unexpected={sorted(got - want)}")
    assert not failures, "\n".join(failures)


def test_every_rule_is_exercised() -> None:
    with open(os.path.join(HERE, "expected.json"), encoding="utf-8") as fh:
        covered = {r for rules in json.load(fh).values() for r in rules}
    untested = set(L.RULES) - covered - {"DF021"}
    assert not untested, f"rules with no fixture: {sorted(untested)}"


if __name__ == "__main__":
    for t in (test_fixtures, test_every_rule_is_exercised):
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}\n{e}")
            sys.exit(1)


def test_verify_image_documents_only_the_flags_it_parses() -> None:
    with open(os.path.join(os.path.dirname(HERE), "verify_image.sh"), encoding="utf-8") as fh:
        source = fh.read()
    header = next(line for line in source.splitlines() if line.startswith("# Usage:"))
    documented = header.split("IMAGE", 1)[1].strip()
    assert documented == "[--stop-test]"
    assert '"${2:-}" == "--stop-test"' in source
    assert 'IMAGE [--stop-test]" >&2' in source
