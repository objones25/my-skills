"""One regression test per audit code, plus the holes found in the 2026-10 audit."""

from __future__ import annotations

import json

import pytest

HOLES = [
    pytest.param("def test_x():\n    assert f() == -0.5\n", "FLOAT_EQ", id="negative-float-literal"),
    pytest.param("def test_x():\n    assert 0.5 == f()\n", "FLOAT_EQ", id="float-literal-on-left"),
    pytest.param("import pytest\ndef test_x():\n    f() == pytest.approx(0.5)\n", "NO_ASSERT", id="approx-without-assert"),
    pytest.param("import pytest\ndef test_x():\n    f() == pytest.approx(0.5)\n", "BARE_COMPARISON", id="bare-comparison"),
    pytest.param("def test_x():\n    assert (f() == 1, 'msg')\n", "ALWAYS_TRUE", id="tuple-assert"),
    pytest.param("from unittest import mock\ndef test_x():\n    with mock.patch.object(mock, 'sentinel') as m:\n        m.assert_not_called()\n", "PATCH_NO_SPEC", id="patch-object-no-spec"),
    pytest.param("def test_x():\n    try:\n        assert f() == 1\n    except AssertionError:\n        pass\n", "SWALLOWED_FAILURE", id="swallowed-assertion"),
    pytest.param("def test_x():\n    assert f() is not None\n", "WEAK_ASSERT", id="weak-only-assertion"),
]

DETECTS = [
    pytest.param("def test_x():\n    f()\n", "NO_ASSERT", id="no-assert"),
    pytest.param("def test_x():\n    assert 'documentation'\n", "ALWAYS_TRUE", id="truthy-string"),
    pytest.param("import math\ndef test_x():\n    assert f() != math.nan\n", "NAN_COMPARE", id="nan-compare"),
    pytest.param("def test_x():\n    assert f() == float('nan')\n", "NAN_COMPARE", id="nan-float-call"),
    pytest.param("def test_x():\n    try:\n        f()\n    except Exception:\n        pass\n    assert True is not False\n", "SWALLOWED_FAILURE", id="swallowed-exception"),
    pytest.param("def test_x():\n    try:\n        f()\n    except:\n        pass\n    assert g() == 1\n", "SWALLOWED_FAILURE", id="bare-except"),
    pytest.param("def test_x(m):\n    m.called_once_with(3)\n", "FAKE_MOCK_ASSERT", id="fake-mock-call"),
    pytest.param("def test_x(m):\n    m.assert_called_once_wtih(3)\n", "FAKE_MOCK_ASSERT", id="mock-typo"),
    pytest.param("import pytest\ndef test_x():\n    with pytest.raises(Exception):\n        f()\n", "BROAD_RAISES", id="broad-raises"),
    pytest.param("import time\ndef test_x():\n    time.sleep(1)\n    assert f() == 1\n", "SLEEP", id="sleep"),
    pytest.param("import requests\ndef test_x():\n    assert requests.get('u').ok == 1\n", "NETWORK", id="network"),
    pytest.param("import random\ndef test_x():\n    assert f(random.random()) == 1\n", "UNSEEDED_RANDOM", id="unseeded"),
    pytest.param("def test_x():\n    print(f())\n    assert f() == 1\n", "PRINT", id="print"),
    pytest.param("def test_x():\n    for c in [1, 2]:\n        assert f(c) == c\n", "BRANCHING", id="for-over-cases"),
    pytest.param("import os\ndef test_x():\n    os.environ['A'] = '1'\n    assert f() == 1\n", "ENV_MUTATION", id="environ-assign"),
    pytest.param("import os\ndef test_x():\n    os.environ.update(A='1')\n    assert f() == 1\n", "ENV_MUTATION", id="environ-update"),
    pytest.param("import sys\ndef test_x():\n    sys.path.insert(0, '.')\n    assert f() == 1\n", "SYS_PATH", id="sys-path"),
    pytest.param("import pytest\n@pytest.mark.skip\ndef test_x():\n    assert f() == 1\n", "SKIP_NO_REASON", id="skip-no-reason"),
    pytest.param("import pytest\n@pytest.mark.xfail(reason='r')\ndef test_x():\n    assert f() == 1\n", "XFAIL_NOT_STRICT", id="xfail-not-strict"),
    pytest.param("from unittest import mock\ndef test_x():\n    with mock.patch('a.b') as m:\n        m.assert_not_called()\n", "PATCH_NO_SPEC", id="patch-no-spec"),
    pytest.param("import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(f(), 1)\n", "UNITTEST_CLASS", id="unittest"),
    pytest.param("SEEN = []\ndef test_x():\n    SEEN.append(1)\n    assert len(SEEN) == 1\n", "MUTABLE_MODULE_STATE", id="module-state"),
    pytest.param("def test_x():\n    assert f() == 1\ndef test_x():\n    assert f() == 2\n", "SHADOWED_TEST_NAME", id="shadowed"),
    pytest.param("def test_x(:\n", "SYNTAX", id="syntax"),
    pytest.param("import pytest\n@pytest.yield_fixture\ndef res():\n    yield 1\ndef test_x(res):\n    assert res == 1\n", "YIELD_FIXTURE", id="yield-fixture"),
    pytest.param("import pytest\n@pytest.yield_fixture(scope='module')\ndef res():\n    yield 1\ndef test_x(res):\n    assert res == 1\n", "YIELD_FIXTURE", id="yield-fixture-call"),
]

CLEAN = [
    pytest.param("import pytest\ndef test_x():\n    with pytest.RaisesGroup(ValueError):\n        f()\n", id="raises-group-asserts"),
    pytest.param("import pytest\ndef test_x():\n    with pytest.RaisesExc(ValueError, match='^x$'):\n        f()\n", id="raises-exc-asserts"),
    pytest.param("import asyncio\nasync def test_x():\n    await asyncio.sleep(0)\n    assert f() == 1\n", id="sleep-zero-yields"),
    pytest.param("class TestA:\n    def test_x(self):\n        assert f() == 1\nclass TestB:\n    def test_x(self):\n        assert f() == 1\n", id="same-name-different-class"),
    pytest.param("import pytest\n@pytest.mark.skip('tracked in #1')\ndef test_x():\n    assert f() == 1\n", id="skip-positional-reason"),
    pytest.param("import random\ndef test_x():\n    rng = random.Random(7)\n    assert f(rng.random()) == 1\n", id="explicitly-seeded-rng"),
    pytest.param("def test_x():\n    try:\n        f()\n    except ValueError:\n        return\n    assert g() == 1\n", id="specific-except-is-fine"),
    pytest.param("def test_x():\n    result = f()\n    assert result is not None\n    assert result.total == 3\n", id="weak-plus-strong"),
    pytest.param("from unittest import mock\ndef test_x():\n    with mock.patch.object(mock, 'sentinel', autospec=True) as m:\n        m.assert_not_called()\n", id="patch-object-autospec"),
    pytest.param("import pytest\ndef test_x():\n    assert f() == pytest.approx(0.5)\n", id="approx-asserted"),
    pytest.param("import pytest, requests\n@pytest.mark.integration\ndef test_x():\n    assert requests.get('u').status_code == 200\n", id="integration-may-use-network"),
]


@pytest.mark.parametrize(("source", "code"), HOLES + DETECTS)
def test_audit_detects(run_audit, source, code):
    assert code in run_audit(source)


@pytest.mark.parametrize("source", CLEAN)
def test_audit_stays_quiet_on_correct_code(run_audit, source):
    assert run_audit(source) == []


def test_integration_directory_exempts_network(run_audit):
    source = "import requests\ndef test_x():\n    assert requests.get('u').status_code == 200\n"
    assert run_audit(source, name="integration/test_api.py") == []


def test_xfail_strict_from_pyproject_silences_xfail_check(run_audit):
    source = "import pytest\n@pytest.mark.xfail(reason='bug #3')\ndef test_x():\n    assert f() == 1\n"
    extra = {"pyproject.toml": "[tool.pytest.ini_options]\nstrict = true\n"}
    assert run_audit(source, extra=extra) == []


def test_seeding_conftest_silences_unseeded_random(run_audit):
    source = "import random\ndef test_x():\n    assert f(random.random()) == 1\n"
    extra = {"conftest.py": "import random\nrandom.seed(1)\n"}
    assert run_audit(source, extra=extra) == []


def test_pythonhashseed_alone_does_not_count_as_seeding(run_audit):
    source = "import random\ndef test_x():\n    assert f(random.random()) == 1\n"
    extra = {"conftest.py": "def f(mp):\n    mp.setenv('PYTHONHASHSEED', '0')\n"}
    assert run_audit(source, extra=extra) == ["UNSEEDED_RANDOM"]


def test_line_pragma_with_reason_suppresses(run_audit):
    source = "def test_x():\n    assert f() == 0.5  # audit: ignore[FLOAT_EQ] exact binary fraction\n"
    assert run_audit(source) == []


def test_def_line_pragma_covers_whole_function(run_audit):
    source = "def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise\n    f()\n"
    assert run_audit(source) == []


def test_pragma_without_reason_is_itself_a_finding(run_audit):
    source = "def test_x():\n    assert f() == 0.5  # audit: ignore[FLOAT_EQ]\n"
    assert sorted(run_audit(source)) == ["FLOAT_EQ", "SUPPRESSION_NO_REASON"]


def test_sarif_output_is_valid_shape(tmp_path, audit_mod, capsys):
    (tmp_path / "test_a.py").write_text("def test_x():\n    f()\n")
    exit_code = audit_mod.main([str(tmp_path), "--format", "sarif"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["runs"][0]["results"][0]["ruleId"] == "NO_ASSERT"


def test_github_output_uses_workflow_command_syntax(tmp_path, audit_mod, capsys):
    (tmp_path / "test_a.py").write_text("def test_x():\n    f()\n")
    audit_mod.main([str(tmp_path), "--format", "github"])
    assert capsys.readouterr().out.startswith("::error file=")


def test_warnings_do_not_fail_the_gate_by_default(tmp_path, audit_mod):
    (tmp_path / "test_a.py").write_text("def test_x():\n    assert f() is not None\n")
    assert audit_mod.main([str(tmp_path)]) == 0
    assert audit_mod.main([str(tmp_path), "--warnings-as-errors"]) == 1


def test_every_code_is_documented(audit_mod):
    rules = (audit_mod.Path(audit_mod.__file__).parents[1] / "references" / "rules.md").read_text()
    missing = [code for code in audit_mod.CODES if code != "SYNTAX" and f"`{code}`" not in rules]
    assert missing == []
