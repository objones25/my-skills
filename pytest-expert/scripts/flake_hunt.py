#!/usr/bin/env python3
"""Rerun a pytest selection many times to find flaky and order-dependent tests.

Each run shuffles test order with a different pytest-randomly seed. A test that
fails under some seeds and passes under others is reported with the seeds that
break it. With --repeat-seed, the same order is rerun, with the random number
generators reseeded from OS entropy for every test, to separate order dependence
(fails only for certain seeds, every time) from plain nondeterminism (fails at
random even with a fixed order, including a test that draws from `random`).
Without --repeat-seed, pytest-randomly reseeds `random` per test from the run's
seed, so an RNG-driven failure looks seed-dependent and is reported as
"order-dependent or nondeterministic".

Usage:
    python flake_hunt.py --runs 31 -- tests/
    python flake_hunt.py --runs 31 --repeat-seed 3 -- tests/unit -k wallet

Gruber et al. (ICST 2021) estimate ~31 random-order runs for 95% confidence that a
passing test is free of order-dependent flakiness, and ~170 reruns for flakiness that
does not depend on order, so treat a clean 31-run result as evidence, not proof.
The default --runs is 31 for that reason.

Do not pass pytest-xdist's -n: tests that land in different worker processes can no
longer interfere, so an order-dependent pair would be reported clean. The script
refuses it (exit 2). Failing and xfail-marked tests count as failures, so a flaky
xfail test is reported too.

Exit status: 0 = ran and every test behaved consistently; 1 = at least one flaky
test; 2 = could not hunt (bad arguments, a path or -k that selects no test, pytest
usage/internal/collection errors or a crash, or nothing executed).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DEFAULT_RUNS = 31  # Gruber et al. (ICST 2021): ~31 random-order runs for 95% confidence on order dependence
PYTEST_EXIT_MEANING = {2: "interrupted, usually a collection error", 3: "internal error", 4: "usage error, e.g. a path that does not exist"}

# Loaded with `-p` for --repeat-seed runs. pytest-randomly reseeds `random` before each test from the run's
# seed, which makes a test that draws from `random` look order-dependent. With --randomly-dont-reset-seed it
# reseeds only at session start, and this plugin then reseeds from OS entropy before every test (fixtures and
# body), so the repeats differ in randomness but not in order.
ENTROPY_PLUGIN = """\
import random
import sys


def pytest_runtest_logstart(nodeid, location):
    random.seed()
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed()
"""


class HuntError(Exception):
    """The hunt could not be carried out; the message says why. Exit status 2."""


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return value


def xdist_option(pytest_args: list[str]) -> str | None:
    """The pytest-xdist worker option in the arguments, if one would start worker processes."""
    for i, arg in enumerate(pytest_args):
        if arg in ("-n", "--numprocesses"):
            value = pytest_args[i + 1] if i + 1 < len(pytest_args) else ""
        elif arg.startswith("--numprocesses="):
            value = arg.split("=", 1)[1]
        elif arg.startswith("-n") and not arg.startswith("--"):
            value = arg[2:]
        else:
            continue
        if value.lstrip("=") != "0":
            return arg
    return None


def outcomes(junit: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314, RUF100 pytest wrote this file moments ago (trusted input)
        test_id = f"{case.get('classname')}::{case.get('name')}"
        skipped = case.find("skipped")
        if case.find("failure") is not None or case.find("error") is not None:
            results[test_id] = "fail"
        elif skipped is not None and skipped.get("type") == "pytest.xfail":
            results[test_id] = "fail"  # an expected failure is still a failure here; an xpass is a pass
        elif skipped is not None:
            results[test_id] = "skip"
        else:
            results[test_id] = "pass"
    return results


def output_tail(proc: subprocess.CompletedProcess[str], lines: int = 12) -> str:
    text = (proc.stdout + proc.stderr).strip().splitlines()
    return "\n".join(text[-lines:])


def run_pytest(cmd: list[str], junit: Path, seed: int, env: dict[str, str]) -> dict[str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    where = f"for seed {seed}"
    if proc.returncode == 5:
        raise HuntError(f"pytest collected no tests (exit 5) {where}; check the path and any -k or -m expression")
    if proc.returncode in PYTEST_EXIT_MEANING:
        raise HuntError(f"pytest exit {proc.returncode} ({PYTEST_EXIT_MEANING[proc.returncode]}) {where}:\n{output_tail(proc)}")
    if proc.returncode not in (0, 1):
        raise HuntError(f"pytest crashed (exit {proc.returncode}) {where}; the tests did not finish:\n{output_tail(proc)}")
    if not junit.exists():
        raise HuntError(f"pytest exited {proc.returncode} but wrote no report {where}:\n{output_tail(proc)}")
    try:
        return outcomes(junit)
    except ET.ParseError as exc:
        raise HuntError(f"pytest's report {where} is unreadable ({exc})") from exc


def hunt(args: argparse.Namespace, pytest_args: list[str], has_randomly: bool) -> dict[str, dict[int, list[str]]]:
    extra = ["--no-cov"] if importlib.util.find_spec("pytest_cov") else []
    history: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    with tempfile.TemporaryDirectory(prefix="flake-hunt-") as tmp:
        plugin_dir = Path(tmp) / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "flake_hunt_entropy.py").write_text(ENTROPY_PLUGIN)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(plugin_dir), os.environ.get("PYTHONPATH", "")]))}
        for seed in range(args.first_seed, args.first_seed + args.runs):
            for repeat in range(args.repeat_seed):
                junit = Path(tmp) / f"run-{seed}-{repeat}.xml"
                order = ["-p", "randomly", f"--randomly-seed={seed}"] if has_randomly else ["-p", "no:randomly"]
                if has_randomly and args.repeat_seed > 1:
                    order += ["--randomly-dont-reset-seed", "-p", "flake_hunt_entropy"]
                cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *order, *extra, f"--junitxml={junit}", *pytest_args]
                for test_id, result in run_pytest(cmd, junit, seed, env).items():
                    history[test_id][seed].append(result)
    if not any(r in ("pass", "fail") for per_seed in history.values() for runs in per_seed.values() for r in runs):
        raise HuntError("no test was executed (everything was skipped or deselected); there is nothing to hunt")
    return history


def classify(per_seed: dict[int, list[str]], repeat_seed: int, has_randomly: bool) -> str | None:
    """None if the test behaved consistently, else its kind of flakiness."""
    results = {r for runs in per_seed.values() for r in runs} - {"skip"}
    if results != {"pass", "fail"}:
        return None
    if not has_randomly:
        return "nondeterministic"  # nothing is shuffled, so the order cannot be the cause
    if repeat_seed == 1:
        return "order-dependent or nondeterministic (use --repeat-seed 3 to tell apart)"
    mixed_within_seed = any(len(set(runs) - {"skip"}) > 1 for runs in per_seed.values())
    return "nondeterministic" if mixed_within_seed else "order-dependent"


def report(history: dict[str, dict[int, list[str]]], args: argparse.Namespace, has_randomly: bool) -> int:
    flaky = []
    for test_id, per_seed in sorted(history.items()):
        kind = classify(per_seed, args.repeat_seed, has_randomly)
        if kind is None:
            continue
        failing = sorted(s for s, runs in per_seed.items() if "fail" in runs)
        flaky.append({"test": test_id, "kind": kind, "failing_seeds": failing, "runs": sum(len(r) for r in per_seed.values())})

    if args.format == "json":
        print(json.dumps({"tests": len(history), "flaky": flaky}, indent=2))
        return 1 if flaky else 0
    if has_randomly:
        print(f"{len(history)} tests x {args.runs} seeds x {args.repeat_seed} repeat(s)")
    else:
        print(f"{len(history)} tests x {args.runs * args.repeat_seed} runs (pytest-randomly not installed; order fixed)")
    for item in flaky:
        seeds = ", ".join(map(str, item["failing_seeds"][:8]))
        where = f"fails with --randomly-seed in [{seeds}]" if has_randomly else f"fails in runs [{seeds}] of {item['runs']}"
        print(f"FLAKY {item['test']}\n      {item['kind']}; {where}")
    print("no flaky tests found" if not flaky else f"\n{len(flaky)} flaky test(s)")
    return 1 if flaky else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pytest_args = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=positive_int, default=DEFAULT_RUNS, help=f"seeds to try (default {DEFAULT_RUNS})")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--repeat-seed", type=positive_int, default=1, help="run each seed this many times")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(own)

    try:
        option = xdist_option(pytest_args)
        if option:
            raise HuntError(f"pytest-xdist option {option!r} would hide order dependence (tests in different worker processes cannot interfere); remove it")
        has_randomly = importlib.util.find_spec("pytest_randomly") is not None
        if not has_randomly:
            print("flake_hunt: pytest-randomly not installed; order is fixed, so only nondeterminism is detectable", file=sys.stderr)
        return report(hunt(args, pytest_args, has_randomly), args, has_randomly)
    except HuntError as exc:
        print(f"flake_hunt: cannot hunt: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
