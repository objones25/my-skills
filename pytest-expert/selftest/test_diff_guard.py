"""diff_guard: each test-gaming edit is reported, and legitimate edits are not."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

BASE_TEST = '''import pytest


def test_withdraw_rejects_overdraft():
    with pytest.raises(ValueError, match="insufficient"):
        withdraw(10, 20)


def test_balance_after_deposit():
    assert deposit(10, 5) == 15
    assert history() == ["deposit"]


class TestFees:
    def test_fee_is_one_percent(self):
        assert fee(1000) == 10
'''

PYPROJECT = '''[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra"]

[tool.coverage.report]
fail_under = 90
'''


@pytest.fixture(scope="session")
def guard_mod():
    from conftest import load_script

    return load_script("scripts/diff_guard.py")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key, value in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_wallet.py").write_text(BASE_TEST)
    (tmp_path / "tests" / "conftest.py").write_text("import pytest\n")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.fixture
def codes(repo, guard_mod):
    def _codes(edits: dict[str, str | None]) -> list[str]:
        for rel, text in edits.items():
            path = repo / rel
            if text is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        return [f.code for f in guard_mod.guard(repo)]

    return _codes


TEST_FILE = "tests/test_wallet.py"

GAMING = [
    pytest.param({TEST_FILE: None}, "TEST_FILE_DELETED", id="file-deleted"),
    pytest.param({TEST_FILE: BASE_TEST.replace("def test_withdraw_rejects_overdraft():\n    with pytest.raises(ValueError, match=\"insufficient\"):\n        withdraw(10, 20)\n", "")}, "TEST_DELETED", id="test-deleted"),
    pytest.param({TEST_FILE: BASE_TEST.replace("    assert history() == [\"deposit\"]\n", "")}, "ASSERTS_REDUCED", id="assert-dropped"),
    pytest.param({TEST_FILE: BASE_TEST.replace("assert fee(1000) == 10", "assert True")}, "ASSERT_TRIVIALIZED", id="assert-true"),
    pytest.param({TEST_FILE: BASE_TEST.replace("assert fee(1000) == 10", "assert fee(1000) == fee(1000)")}, "ASSERT_TRIVIALIZED", id="self-compare"),
    pytest.param({TEST_FILE: BASE_TEST.replace("def test_balance_after_deposit():\n", "@pytest.mark.skip(reason=\"flaky\")\ndef test_balance_after_deposit():\n")}, "SKIP_ADDED", id="skip-marker"),
    pytest.param({TEST_FILE: BASE_TEST.replace("class TestFees:", "@pytest.mark.xfail(reason=\"later\", strict=False)\nclass TestFees:")}, "SKIP_ADDED", id="class-xfail"),
    pytest.param({TEST_FILE: BASE_TEST.replace("import pytest\n", "import pytest\n\npytestmark = pytest.mark.skip(reason=\"wip\")\n", 1)}, "SKIP_ADDED", id="module-pytestmark"),
    pytest.param({TEST_FILE: BASE_TEST.replace("    assert deposit(10, 5) == 15\n", "    pytest.skip(\"todo\")\n    assert deposit(10, 5) == 15\n")}, "SKIP_ADDED", id="skip-call"),
    pytest.param({TEST_FILE: BASE_TEST.replace("    assert fee(1000) == 10\n", "    return\n        assert fee(1000) == 10\n")}, "TEST_EMPTIED", id="early-return"),
    pytest.param({TEST_FILE: BASE_TEST.replace("    assert fee(1000) == 10\n", "        pass\n")}, "TEST_EMPTIED", id="body-pass"),
    pytest.param({TEST_FILE: BASE_TEST.replace("assert deposit(10, 5) == 15", "assert deposit(10, 5) == 15  # audit: ignore[FLOAT_EQ] fine")}, "PRAGMA_ADDED", id="audit-pragma"),
    pytest.param({"tests/conftest.py": "import pytest\ncollect_ignore = ['test_wallet.py']\n"}, "COLLECTION_NARROWED", id="collect-ignore"),
    pytest.param({"tests/conftest.py": "import pytest\n\ndef pytest_collection_modifyitems(config, items):\n    items[:] = [i for i in items if 'fee' not in i.name]\n"}, "COLLECTION_NARROWED", id="modifyitems-hook"),
    pytest.param({"pyproject.toml": PYPROJECT.replace('addopts = ["-ra"]', 'addopts = ["-ra", "--deselect", "tests/test_wallet.py::test_balance_after_deposit"]')}, "CONFIG_NARROWED", id="deselect"),
    pytest.param({"pyproject.toml": PYPROJECT.replace('addopts = ["-ra"]', 'addopts = ["-ra", "-k", "not fee"]')}, "CONFIG_NARROWED", id="dash-k"),
    pytest.param({"pyproject.toml": PYPROJECT.replace('addopts = ["-ra"]', 'addopts = ["-ra", "--ignore=tests/test_wallet.py"]')}, "CONFIG_NARROWED", id="ignore-equals"),
    pytest.param({"pyproject.toml": PYPROJECT.replace('testpaths = ["tests"]', 'testpaths = ["tests/unit"]')}, "CONFIG_NARROWED", id="testpaths-shrunk"),
    pytest.param({"pyproject.toml": PYPROJECT.replace("fail_under = 90", "fail_under = 60")}, "COVERAGE_LOWERED", id="fail-under-lowered"),
    pytest.param({"pyproject.toml": PYPROJECT.replace('addopts = ["-ra"]', 'addopts = ["-ra", "--cov-fail-under=50"]')}, "COVERAGE_LOWERED", id="cov-fail-under-flag"),
    pytest.param({TEST_FILE: BASE_TEST.replace("assert fee(1000) == 10", "assert fee(1000) == 11")}, "EXPECTATION_CHANGED", id="expected-literal-changed"),
    pytest.param({TEST_FILE: BASE_TEST + "\ndef test_broken(:\n"}, "SYNTAX", id="unparsable-file"),
]


@pytest.mark.parametrize(("edits", "code"), GAMING)
def test_gaming_edit_is_reported(codes, edits, code):
    assert code in codes(edits)


LEGITIMATE = [
    pytest.param({}, id="no-change"),
    pytest.param({TEST_FILE: BASE_TEST + "\n\ndef test_new_behavior():\n    assert deposit(0, 1) == 1\n"}, id="test-added"),
    pytest.param({TEST_FILE: BASE_TEST.replace("    assert fee(1000) == 10\n", "    assert fee(1000) == 10\n        assert fee(0) == 0\n")}, id="assert-added"),
    pytest.param({"tests/test_other.py": "def test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == ['deposit']\n",
                  TEST_FILE: BASE_TEST.replace("def test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == [\"deposit\"]\n", "")}, id="test-moved-to-another-file"),
    pytest.param({"pyproject.toml": PYPROJECT.replace("fail_under = 90", "fail_under = 95")}, id="fail-under-raised"),
    pytest.param({"src/wallet.py": "def fee(x):\n    return x // 100\n"}, id="source-only-change"),
]


def test_inline_snapshot_rewrite_is_an_expectation_change(repo, guard_mod):
    path = repo / "tests" / "test_snap.py"
    path.write_text("from inline_snapshot import snapshot\n\ndef test_total():\n    assert total() == snapshot(250)\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "snap"], cwd=repo, check=True)
    path.write_text(path.read_text().replace("snapshot(250)", "snapshot(251)"))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo)] == [("EXPECTATION_CHANGED", "test_total: 250 -> 251")]


def test_syrupy_snapshot_rewrite_is_flagged(repo, guard_mod):
    snap = repo / "tests" / "__snapshots__" / "test_wallet.ambr"
    snap.parent.mkdir()
    snap.write_text("# name: test_total\n  250\n# ---\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "snap"], cwd=repo, check=True)
    snap.write_text("# name: test_total\n  251\n# ---\n")
    assert [f.code for f in guard_mod.guard(repo)] == ["SNAPSHOT_CHANGED"]


@pytest.mark.parametrize("edits", LEGITIMATE)
def test_legitimate_edit_is_not_an_error(codes, guard_mod, edits):
    errors = [c for c in codes(edits) if guard_mod.CODES[c][0] == "error"]
    assert errors == []


def test_renaming_a_test_reads_as_a_deletion(codes):
    """The guard cannot tell a rename from delete-and-add; a human approves with --allow."""
    renamed = BASE_TEST.replace("def test_balance_after_deposit", "def test_balance_after_a_deposit_of_five")
    assert codes({TEST_FILE: renamed}) == ["TEST_DELETED"]


def test_syntax_error_is_not_reported_as_mass_deletion(codes):
    assert codes({TEST_FILE: BASE_TEST + "\ndef test_broken(:\n"}) == ["SYNTAX"]


def test_uncommitted_new_file_counts_as_current(codes):
    assert codes({"tests/test_new.py": "def test_x():\n    assert True\n"}) == []


def test_allow_suppresses_a_code(repo, guard_mod):
    (repo / TEST_FILE).unlink()
    assert guard_mod.guard(repo, allow=frozenset({"TEST_FILE_DELETED"})) == []


def test_base_ref_option_compares_against_older_commit(repo, guard_mod):
    (repo / TEST_FILE).write_text(BASE_TEST.replace("    assert history() == [\"deposit\"]\n", ""))
    subprocess.run(["git", "commit", "-q", "-am", "weaken"], cwd=repo, check=True)
    assert [f.code for f in guard_mod.guard(repo)] == []
    assert [f.code for f in guard_mod.guard(repo, base="HEAD~1")] == ["ASSERTS_REDUCED"]


def test_cli_exit_codes_and_json(repo, guard_mod, capsys):
    assert guard_mod.main([str(repo)]) == 0
    (repo / TEST_FILE).write_text(BASE_TEST.replace("assert fee(1000) == 10", "assert True"))
    assert guard_mod.main([str(repo), "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert [f["code"] for f in payload] == ["ASSERT_TRIVIALIZED"]


def test_warning_only_passes_unless_warnings_as_errors(repo, guard_mod):
    (repo / TEST_FILE).write_text(BASE_TEST.replace("assert fee(1000) == 10", "assert fee(1000) == 11"))
    assert guard_mod.main([str(repo)]) == 0
    assert guard_mod.main([str(repo), "--warnings-as-errors"]) == 1


def test_not_a_repository_exits_2(tmp_path, guard_mod):
    assert guard_mod.main([str(tmp_path)]) == 2


def test_unknown_allow_code_exits_2(repo, guard_mod):
    assert guard_mod.main([str(repo), "--allow", "NOPE"]) == 2


SRC = "src/wallet.py"
BASE_SRC = "class Money:\n    def __init__(self, cents):\n        self.cents = cents\n\n\ndef fee(cents):\n    return cents // 100\n"


@pytest.fixture
def repo_with_src(repo):
    (repo / "src").mkdir()
    (repo / SRC).write_text(BASE_SRC)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "src"], cwd=repo, check=True)
    return repo


def test_constant_eq_in_source_is_flagged(repo_with_src, guard_mod):
    (repo_with_src / SRC).write_text(BASE_SRC.replace("        self.cents = cents\n", "        self.cents = cents\n\n    def __eq__(self, other):\n        return True\n"))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo_with_src)] == [("CONSTANT_EQ", "Money.__eq__ returns a constant")]


def test_special_casing_a_test_input_is_flagged(repo_with_src, guard_mod):
    (repo_with_src / SRC).write_text(BASE_SRC.replace("    return cents // 100\n", "    if cents == 1000:\n        return 10\n    return cents // 100\n"))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo_with_src)] == [("SPECIAL_CASE", "compares with 1000, which a test also uses")]


def test_comparison_with_a_literal_no_test_uses_is_not_flagged(repo_with_src, guard_mod):
    (repo_with_src / SRC).write_text(BASE_SRC.replace("    return cents // 100\n", "    if cents == 77777:\n        return 0\n    return cents // 100\n"))
    assert guard_mod.guard(repo_with_src) == []


def test_source_warnings_do_not_fail_the_cli(repo_with_src, guard_mod):
    (repo_with_src / SRC).write_text(BASE_SRC.replace("    return cents // 100\n", "    if cents == 1000:\n        return 10\n    return cents // 100\n"))
    assert guard_mod.main([str(repo_with_src)]) == 0
    assert guard_mod.main([str(repo_with_src), "--warnings-as-errors"]) == 1


CASES = '@pytest.mark.parametrize(("a", "b"), [(1, 2), (5000, 0), (3, 4)])\ndef test_rows(a, b):\n    assert f(a) == b\n'


def test_dropping_a_parametrize_row_is_flagged(repo, guard_mod):
    path = repo / "tests" / "test_rows.py"
    path.write_text("import pytest\n\n" + CASES)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "rows"], cwd=repo, check=True)
    path.write_text(path.read_text().replace("(5000, 0), ", ""))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo)] == [("CASES_REDUCED", "test_rows: 3 parametrize rows -> 2")]


def test_adding_a_parametrize_row_is_fine(repo, guard_mod):
    path = repo / "tests" / "test_rows.py"
    path.write_text("import pytest\n\n" + CASES)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "rows"], cwd=repo, check=True)
    path.write_text(path.read_text().replace("(3, 4)]", "(3, 4), (7, 8)]"))
    assert guard_mod.guard(repo) == []



def test_deleting_a_raises_exc_check_is_reported(repo, guard_mod):
    check = "    with pytest.RaisesExc(ValueError, match=\"negative\"):\n        deposit(10, -1)\n"
    path = repo / TEST_FILE
    path.write_text(BASE_TEST + "\n\ndef test_rejects_negative():\n" + check)
    subprocess.run(["git", "commit", "-qam", "add RaisesExc test"], cwd=repo, check=True)
    path.write_text(path.read_text().replace(check, "    deposit(10, -1)\n"))
    assert [f.code for f in guard_mod.guard(repo)] == ["ASSERTS_REDUCED"]
