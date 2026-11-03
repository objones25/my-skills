#!/usr/bin/env python3
"""Rerun a pytest selection many times to find flaky and order-dependent tests.

Each run shuffles test order with a different pytest-randomly seed. A test that
fails under some seeds and passes under others is reported with the seeds that
break it. With --repeat-seed, the same order is rerun to separate order dependence
(fails only for certain seeds, every time) from plain nondeterminism (fails at
random even with a fixed order).

Usage:
    python flake_hunt.py --runs 30 -- tests/
    python flake_hunt.py --runs 30 --repeat-seed 3 -- tests/unit -k wallet

Gruber et al. (ICST 2021) estimate ~31 random-order runs for 95% confidence that a
passing test is free of order-dependent flakiness, and ~170 reruns for flakiness that
does not depend on order, so treat a clean 30-run result as evidence, not proof.

Exit status: 1 if any test was flaky, 0 if every test behaved consistently, 2 on
usage errors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def outcomes(junit: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314, RUF100 pytest wrote this file moments ago (trusted input)
        test_id = f"{case.get('classname')}::{case.get('name')}"
        if case.find("failure") is not None or case.find("error") is not None:
            results[test_id] = "fail"
        elif case.find("skipped") is not None:
            results[test_id] = "skip"
        else:
            results[test_id] = "pass"
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pytest_args = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--repeat-seed", type=int, default=1, help="run each seed this many times")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(own)

    has_randomly = importlib.util.find_spec("pytest_randomly") is not None
    if not has_randomly:
        print("flake_hunt: pytest-randomly not installed; order is fixed, so only nondeterminism is detectable", file=sys.stderr)
    extra = ["--no-cov"] if importlib.util.find_spec("pytest_cov") else []

    history: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    with tempfile.TemporaryDirectory(prefix="flake-hunt-") as tmp:
        for seed in range(args.first_seed, args.first_seed + args.runs):
            for repeat in range(args.repeat_seed):
                junit = Path(tmp) / f"run-{seed}-{repeat}.xml"
                order = ["-p", "randomly", f"--randomly-seed={seed}"] if has_randomly else ["-p", "no:randomly"]
                cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *order, *extra, f"--junitxml={junit}", *pytest_args]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if not junit.exists():
                    print(f"flake_hunt: pytest produced no report for seed {seed}; check the arguments", file=sys.stderr)
                    return 2
                for test_id, result in outcomes(junit).items():
                    history[test_id][seed].append(result)

    flaky = []
    for test_id, per_seed in sorted(history.items()):
        results = {r for runs in per_seed.values() for r in runs} - {"skip"}
        if results != {"pass", "fail"}:
            continue
        failing = sorted(s for s, runs in per_seed.items() if "fail" in runs)
        mixed_within_seed = any(len(set(runs) - {"skip"}) > 1 for runs in per_seed.values())
        kind = "nondeterministic" if mixed_within_seed or (args.repeat_seed == 1 and not has_randomly) else "order-dependent"
        if args.repeat_seed == 1 and has_randomly:
            kind = "order-dependent or nondeterministic (use --repeat-seed 3 to tell apart)"
        flaky.append({"test": test_id, "kind": kind, "failing_seeds": failing, "runs": sum(len(r) for r in per_seed.values())})

    if args.format == "json":
        print(json.dumps({"tests": len(history), "flaky": flaky}, indent=2))
    else:
        print(f"{len(history)} tests x {args.runs} seeds x {args.repeat_seed} repeat(s)")
        for item in flaky:
            seeds = ", ".join(map(str, item["failing_seeds"][:8]))
            print(f"FLAKY {item['test']}\n      {item['kind']}; fails with --randomly-seed in [{seeds}]")
        print("no flaky tests found" if not flaky else f"\n{len(flaky)} flaky test(s)")
    return 1 if flaky else 0


if __name__ == "__main__":
    raise SystemExit(main())
