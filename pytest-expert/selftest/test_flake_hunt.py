"""flake_hunt.py: order dependence and nondeterminism are told apart; a clean suite is clean;
a hunt that examined nothing exits 2 instead of saying "no flaky tests found"."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest
from conftest import SKILL_ROOT, load_script

SCRIPT = SKILL_ROOT / "scripts" / "flake_hunt.py"

LEAKY_STATE = {
    "state.py": "RATES = {}\n",
    "tests/test_a_sets.py": "import state\n\n\ndef test_sets_rate():\n    state.RATES['EUR'] = 0.5\n    assert state.RATES['EUR'] == 0.5\n",
    "tests/test_b_reads.py": "import state\n\n\ndef test_rate_table_starts_empty():\n    assert state.RATES == {}\n",
}
ALTERNATING = {
    "tests/test_alt.py": (
        "from pathlib import Path\n\n\n"
        "def test_fails_every_other_run(tmp_path_factory):\n"
        "    counter = Path(__file__).with_name('runs.txt')\n"
        "    n = int(counter.read_text()) if counter.exists() else 0\n"
        "    counter.write_text(str(n + 1))\n"
        "    assert n % 2 == 0\n"
    ),
}
XFAIL_ALTERNATING = {
    "tests/test_alt.py": ALTERNATING["tests/test_alt.py"].replace(
        "def test_fails_every_other_run", "@pytest.mark.xfail(reason='known flaky')\ndef test_fails_every_other_run"
    ).replace("from pathlib import Path", "import pytest\nfrom pathlib import Path"),
}
GLOBAL_RNG = {
    "tests/test_rng.py": "import random\n\n\ndef test_coin_flip():\n    assert random.random() > 0.5\n",
}
RECORDS_DRAWS = {
    "tests/test_draw.py": (
        "import random\nfrom pathlib import Path\n\n\n"
        "def test_records_a_draw():\n"
        "    draw = random.random()\n"
        "    with open(Path(__file__).with_name('draws.txt'), 'a') as log:\n"
        "        log.write(f'{draw}\\n')\n"
        "    assert 0 <= draw < 1\n"
    ),
}
CLEAN = {"tests/test_ok.py": "def test_ok():\n    assert 1 + 1 == 2\n"}


@pytest.fixture
def needs_randomly():
    assert importlib.util.find_spec("pytest_randomly") is not None, (
        "these tests need pytest-randomly in the interpreter that runs the self-tests: pip install pytest-randomly"
    )


def write(tmp_path, files):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)


def run_hunt(tmp_path, files, own, pytest_args=("-p", "no:asyncio", "tests/")):
    """Run the script as a user would; returns the CompletedProcess."""
    write(tmp_path, files)
    cmd = [sys.executable, str(SCRIPT), *own, "--", *pytest_args]
    return subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=600,
                          env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=False)


def hunt(tmp_path, files, *args):
    proc = run_hunt(tmp_path, files, ["--runs", "6", "--format", "json", *args])
    return proc.returncode, json.loads(proc.stdout)


def seeds_where_b_runs_before_a(tmp_path, seeds):
    """Independent oracle: ask pytest-randomly for each seed's order and keep the seeds where the writer
    runs before the reader, because only then does the reader see the dirty table and fail."""
    failing = []
    for seed in seeds:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-p", "no:asyncio",
             "-p", "randomly", f"--randomly-seed={seed}", "tests/"],
            cwd=tmp_path, capture_output=True, text=True, check=True, env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )
        order = [line for line in proc.stdout.splitlines() if "::" in line]
        if order.index("tests/test_a_sets.py::test_sets_rate") < order.index("tests/test_b_reads.py::test_rate_table_starts_empty"):
            failing.append(seed)  # the writer ran first, so the reader sees a dirty table and fails
    return failing


# --- classification, with the real pytest-randomly -------------------------------------------------


@pytest.mark.usefixtures("needs_randomly")
def test_order_dependent_test_is_labelled_order_dependent_with_its_exact_failing_seeds(tmp_path):
    code, report = hunt(tmp_path, LEAKY_STATE, "--repeat-seed", "2")
    expected = seeds_where_b_runs_before_a(tmp_path, range(1, 7))
    assert 0 < len(expected) < 6, "the probe seeds must mix both orders or this test proves nothing"
    assert code == 1
    assert report == {
        "tests": 2,
        "flaky": [{"test": "tests.test_b_reads::test_rate_table_starts_empty", "kind": "order-dependent",
                   "failing_seeds": expected, "runs": 12}],
    }


@pytest.mark.usefixtures("needs_randomly")
def test_nondeterministic_test_is_labelled_nondeterministic_with_every_failing_seed(tmp_path):
    code, report = hunt(tmp_path, ALTERNATING, "--repeat-seed", "2")
    assert code == 1
    assert report == {
        "tests": 1,
        "flaky": [{"test": "tests.test_alt::test_fails_every_other_run", "kind": "nondeterministic",
                   "failing_seeds": [1, 2, 3, 4, 5, 6], "runs": 12}],
    }


@pytest.mark.usefixtures("needs_randomly")
def test_a_test_drawing_from_the_global_random_module_is_nondeterministic_not_order_dependent(tmp_path):
    # pytest-randomly reseeds `random` from the run's seed before every test, so with a plain rerun the
    # outcome is a function of the seed alone. The repeats must break that link.
    proc = run_hunt(tmp_path, GLOBAL_RNG, ["--runs", "10", "--repeat-seed", "3", "--format", "json"])
    report = json.loads(proc.stdout)
    assert proc.returncode == 1
    assert [(f["test"], f["kind"], f["runs"]) for f in report["flaky"]] == [("tests.test_rng::test_coin_flip", "nondeterministic", 30)]


def draws_of_a_hunt(tmp_path, *own):
    log = tmp_path / "tests" / "draws.txt"
    log.unlink(missing_ok=True)
    proc = run_hunt(tmp_path, RECORDS_DRAWS, ["--runs", "3", *own])
    assert proc.returncode == 0, proc.stderr
    return log.read_text().split()


@pytest.mark.usefixtures("needs_randomly")
def test_repeats_of_one_seed_draw_different_random_numbers_but_one_run_per_seed_does_not(tmp_path):
    repeated = draws_of_a_hunt(tmp_path, "--repeat-seed", "2")
    assert len(repeated) == 6
    assert all(repeated[i] != repeated[i + 1] for i in (0, 2, 4)), "repeats of the same seed must not replay the same draws"
    single = draws_of_a_hunt(tmp_path)
    assert len(single) == 3
    assert draws_of_a_hunt(tmp_path) == single, "one run per seed keeps pytest-randomly's per-seed reseeding"


@pytest.mark.usefixtures("needs_randomly")
def test_one_run_per_seed_cannot_tell_the_two_kinds_apart_and_says_so(tmp_path):
    code, report = hunt(tmp_path, ALTERNATING)
    assert code == 1
    assert [f["kind"] for f in report["flaky"]] == ["order-dependent or nondeterministic (use --repeat-seed 3 to tell apart)"]


@pytest.mark.usefixtures("needs_randomly")
def test_a_consistently_failing_test_is_not_flaky(tmp_path):
    code, report = hunt(tmp_path, {"tests/test_red.py": "def test_red():\n    assert 1 + 1 == 3\n"}, "--repeat-seed", "2")
    assert (code, report) == (0, {"tests": 1, "flaky": []})


@pytest.mark.usefixtures("needs_randomly")
def test_a_flaky_xfail_test_is_reported_and_a_steady_xfail_is_not(tmp_path):
    code, report = hunt(tmp_path, XFAIL_ALTERNATING, "--repeat-seed", "2")
    assert code == 1
    assert [(f["test"], f["kind"]) for f in report["flaky"]] == [("tests.test_alt::test_fails_every_other_run", "nondeterministic")]
    steady = {"tests/test_steady.py": "import pytest\n\n\n@pytest.mark.xfail(reason='known')\ndef test_known():\n    assert 1 == 2\n\n\ndef test_ok():\n    assert 1 == 1\n"}
    assert hunt(tmp_path / "steady", steady) == (0, {"tests": 2, "flaky": []})


def test_skipped_tests_next_to_executed_ones_are_not_an_error(tmp_path):
    files = {**CLEAN, "tests/test_skipped.py": "import pytest\n\n\n@pytest.mark.skip(reason='later')\ndef test_later():\n    assert 1 == 1\n"}
    assert hunt(tmp_path, files) == (0, {"tests": 2, "flaky": []})


def test_clean_suite_exits_0(tmp_path):
    assert hunt(tmp_path, CLEAN) == (0, {"tests": 1, "flaky": []})


# --- what the arguments do ------------------------------------------------------------------------------


@pytest.mark.usefixtures("needs_randomly")
def test_arguments_after_the_double_dash_reach_pytest(tmp_path):
    files = {**CLEAN, "tests/test_other.py": "def test_other():\n    assert 2 + 2 == 4\n"}
    proc = run_hunt(tmp_path, files, ["--runs", "2", "--format", "json"], pytest_args=("-p", "no:asyncio", "tests/test_other.py"))
    assert (proc.returncode, json.loads(proc.stdout)) == (0, {"tests": 1, "flaky": []})
    proc = run_hunt(tmp_path, files, ["--runs", "2", "--format", "json"], pytest_args=("-p", "no:asyncio", "tests/", "-k", "test_ok"))
    assert (proc.returncode, json.loads(proc.stdout)) == (0, {"tests": 1, "flaky": []})


@pytest.mark.usefixtures("needs_randomly")
def test_first_seed_and_runs_choose_the_seeds(tmp_path):
    proc = run_hunt(tmp_path, ALTERNATING, ["--runs", "3", "--first-seed", "100", "--repeat-seed", "2", "--format", "json"])
    assert [f["failing_seeds"] for f in json.loads(proc.stdout)["flaky"]] == [[100, 101, 102]]


def test_default_is_31_runs_and_help_says_so(tmp_path):
    proc = run_hunt(tmp_path, CLEAN, [])
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0] in ("1 tests x 31 seeds x 1 repeat(s)", "1 tests x 31 runs (pytest-randomly not installed; order fixed)")
    helped = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True).stdout
    assert "default 31" in helped
    assert "30" not in helped.replace("1 in 30", "")


# --- text output ----------------------------------------------------------------------------------------


@pytest.mark.usefixtures("needs_randomly")
def test_text_output_of_a_clean_suite(tmp_path):
    proc = run_hunt(tmp_path, CLEAN, ["--runs", "3"])
    assert (proc.returncode, proc.stdout) == (0, "1 tests x 3 seeds x 1 repeat(s)\nno flaky tests found\n")


@pytest.mark.usefixtures("needs_randomly")
def test_text_output_of_a_flaky_suite(tmp_path):
    proc = run_hunt(tmp_path, ALTERNATING, ["--runs", "3", "--repeat-seed", "2"])
    assert proc.returncode == 1
    assert proc.stdout == textwrap.dedent("""\
        1 tests x 3 seeds x 2 repeat(s)
        FLAKY tests.test_alt::test_fails_every_other_run
              nondeterministic; fails with --randomly-seed in [1, 2, 3]

        1 flaky test(s)
        """)


@pytest.mark.usefixtures("needs_randomly")
def test_text_output_lists_at_most_the_first_eight_failing_seeds(tmp_path):
    proc = run_hunt(tmp_path, ALTERNATING, ["--runs", "10", "--repeat-seed", "2"])
    assert proc.returncode == 1
    assert "nondeterministic; fails with --randomly-seed in [1, 2, 3, 4, 5, 6, 7, 8]\n" in proc.stdout


def test_without_pytest_randomly_it_warns_and_labels_nondeterministic(tmp_path, monkeypatch, capsys):
    module = load_script("scripts/flake_hunt.py")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name, *a: None if name == "pytest_randomly" else real_find_spec(name, *a))
    write(tmp_path, ALTERNATING)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    code = module.main(["--runs", "3", "--repeat-seed", "2", "--", "-p", "no:asyncio", "tests/"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err == "flake_hunt: pytest-randomly not installed; order is fixed, so only nondeterminism is detectable\n"
    assert captured.out == textwrap.dedent("""\
        1 tests x 6 runs (pytest-randomly not installed; order fixed)
        FLAKY tests.test_alt::test_fails_every_other_run
              nondeterministic; fails in runs [1, 2, 3] of 6

        1 flaky test(s)
        """)


# --- exit 2: the hunt could not be carried out ---------------------------------------------------------


def test_the_smallest_valid_numbers_are_accepted(tmp_path):
    proc = run_hunt(tmp_path, CLEAN, ["--runs", "1", "--repeat-seed", "1", "--format", "json"])
    assert (proc.returncode, json.loads(proc.stdout)) == (0, {"tests": 1, "flaky": []})


@pytest.mark.parametrize("own, fragment", [
    (["--runs", "0"], "0 is not a positive integer"),
    (["--runs", "-3"], "-3 is not a positive integer"),
    (["--runs", "many"], "'many' is not an integer"),
    (["--repeat-seed", "0"], "0 is not a positive integer"),
], ids=["runs-0", "runs-negative", "runs-text", "repeat-0"])
def test_bad_numbers_are_a_usage_error_not_a_clean_report(tmp_path, own, fragment):
    proc = run_hunt(tmp_path, CLEAN, own)
    assert proc.returncode == 2
    assert fragment in proc.stderr
    assert "no flaky tests found" not in proc.stdout


@pytest.mark.parametrize("pytest_args, fragment", [
    (("tests/missing_dir",), "pytest exit 4 (usage error, e.g. a path that does not exist)"),
    (("tests/", "-k", "no_such_test_name"), "pytest collected no tests (exit 5)"),
    (("tests/", "-n", "2"), "pytest-xdist option '-n'"),
    (("tests/", "--numprocesses=2"), "pytest-xdist option '--numprocesses=2'"),
    (("tests/", "-m", "nothing_has_this_marker"), "pytest collected no tests (exit 5)"),
], ids=["missing-path", "k-matches-nothing", "xdist-short", "xdist-long", "marker-matches-nothing"])
def test_a_hunt_that_selected_or_ran_nothing_exits_2(tmp_path, pytest_args, fragment):
    proc = run_hunt(tmp_path, CLEAN, ["--runs", "2"], pytest_args=("-p", "no:asyncio", *pytest_args))
    assert proc.returncode == 2
    assert fragment in proc.stderr
    assert proc.stderr.startswith("flake_hunt:") or "pytest-randomly not installed" in proc.stderr
    assert "no flaky tests found" not in proc.stdout


def test_a_collection_error_exits_2_with_pytests_output(tmp_path):
    proc = run_hunt(tmp_path, {"tests/test_broken.py": "def test_x(:\n"}, ["--runs", "2"])
    assert proc.returncode == 2
    assert "pytest exit 2 (interrupted, usually a collection error)" in proc.stderr
    assert "SyntaxError" in proc.stderr


def test_a_pytest_crash_is_reported_as_a_crash_not_as_bad_arguments(tmp_path):
    proc = run_hunt(tmp_path, {"tests/test_dies.py": "import os\n\n\ndef test_dies():\n    os._exit(7)\n"}, ["--runs", "2"])
    assert proc.returncode == 2
    assert "pytest crashed (exit 7)" in proc.stderr
    assert "check the arguments" not in proc.stderr


def test_a_suite_where_everything_is_skipped_exits_2(tmp_path):
    files = {"tests/test_skipped.py": "import pytest\n\n\n@pytest.mark.skip(reason='later')\ndef test_later():\n    assert 1 == 1\n"}
    proc = run_hunt(tmp_path, files, ["--runs", "2"])
    assert proc.returncode == 2
    assert "no test was executed" in proc.stderr
    assert "no flaky tests found" not in proc.stdout


# --- the pure parts -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flake_mod():
    return load_script("scripts/flake_hunt.py")


@pytest.mark.parametrize("args, expected", [
    (["-n", "2"], "-n"),
    (["-n2"], "-n2"),
    (["-nauto"], "-nauto"),
    (["--numprocesses", "auto"], "--numprocesses"),
    (["--numprocesses=4"], "--numprocesses=4"),
    (["tests/", "-n", "logical"], "-n"),
    (["-n"], "-n"),
    (["tests/", "-n"], "-n"),
    (["-n", "0"], None),
    (["tests/", "-n", "0"], None),
    (["-n", "0", "tests/"], None),
    (["-n0"], None),
    (["--numprocesses=0"], None),
    (["tests/", "-k", "name", "-x", "--no-cov"], None),
    ([], None),
])
def test_xdist_option_detection(flake_mod, args, expected):
    assert flake_mod.xdist_option(args) == expected


JUNIT = """<testsuites><testsuite>
<testcase classname="m" name="passes"/>
<testcase classname="m" name="fails"><failure message="x"/></testcase>
<testcase classname="m" name="errors"><error message="x"/></testcase>
<testcase classname="m" name="skipped"><skipped type="pytest.skip" message="s"/></testcase>
<testcase classname="m" name="xfailed"><skipped type="pytest.xfail" message="r"/></testcase>
</testsuite></testsuites>"""


def test_outcomes_reads_each_junit_result(flake_mod, tmp_path):
    report = tmp_path / "r.xml"
    report.write_text(JUNIT)
    assert flake_mod.outcomes(report) == {
        "m::passes": "pass", "m::fails": "fail", "m::errors": "fail", "m::skipped": "skip", "m::xfailed": "fail",
    }


@pytest.mark.parametrize("per_seed, repeat, randomly, expected", [
    ({1: ["pass"], 2: ["fail"]}, 1, True, "order-dependent or nondeterministic (use --repeat-seed 3 to tell apart)"),
    ({1: ["pass", "pass"], 2: ["fail", "fail"]}, 2, True, "order-dependent"),
    ({1: ["pass", "fail"], 2: ["pass", "pass"]}, 2, True, "nondeterministic"),
    ({1: ["pass", "pass"], 2: ["pass", "pass"]}, 2, True, None),
    ({1: ["fail", "fail"], 2: ["fail", "fail"]}, 2, True, None),
    ({1: ["pass", "skip"], 2: ["skip", "skip"]}, 2, True, None),
    ({1: ["pass", "pass"], 2: ["fail", "fail"]}, 2, False, "nondeterministic"),
    ({1: ["pass"], 2: ["fail"]}, 1, False, "nondeterministic"),
    ({1: ["pass", "skip"], 2: ["fail", "skip"]}, 2, True, "order-dependent"),
], ids=["one-run", "window", "mixed-within-seed", "all-pass", "all-fail", "skips", "no-randomly-window", "no-randomly-single", "skip-ignored"])
def test_classify(flake_mod, per_seed, repeat, randomly, expected):
    assert flake_mod.classify(per_seed, repeat, randomly) == expected
