"""Regression tests for scripts/audit_tests.py: one per rule, every CLI path, and the holes found
in the 2026-10 and 2026-11 audits.

Every DETECTS case pins the exact (code, line) pairs the audit reports, not just that a code is
among them, so an extra, missing or mislocated finding fails. Every detection has a near-miss in
CLEAN that must stay quiet.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

SEVERITY = {
    "NO_ASSERT": "error",
    "ALWAYS_TRUE": "error",
    "BARE_COMPARISON": "error",
    "NAN_COMPARE": "error",
    "FLOAT_EQ": "error",
    "SWALLOWED_FAILURE": "error",
    "FAKE_MOCK_ASSERT": "error",
    "BROAD_RAISES": "error",
    "WEAK_ASSERT": "warning",
    "SLEEP": "error",
    "NETWORK": "error",
    "UNSEEDED_RANDOM": "error",
    "PRINT": "warning",
    "BRANCHING": "error",
    "ENV_MUTATION": "error",
    "SYS_PATH": "error",
    "SKIP_NO_REASON": "error",
    "XFAIL_NOT_STRICT": "error",
    "PATCH_NO_SPEC": "error",
    "UNITTEST_CLASS": "error",
    "MUTABLE_MODULE_STATE": "error",
    "SHADOWED_TEST_NAME": "error",
    "YIELD_FIXTURE": "error",
    "SUPPRESSION_NO_REASON": "error",
    "SUPPRESSION_INERT": "warning",
    "SYNTAX": "error",
}


def src(*lines: str) -> str:
    return "\n".join(lines) + "\n"


CODES_EXPECTED_BY_CASES: set[str] = set()


def case(case_id: str, source: str, *expected: tuple[str, int]):
    CODES_EXPECTED_BY_CASES.update(code for code, _ in expected)
    return pytest.param(source, sorted(expected, key=lambda e: (e[1], e[0])), id=case_id)


HOLES = [
    case("negative-float-literal", src("def test_x():", "    assert f() == -0.5"), ("FLOAT_EQ", 2)),
    case("float-literal-on-left", src("def test_x():", "    assert 0.5 == f()"), ("FLOAT_EQ", 2)),
    case("approx-without-assert", src("import pytest", "def test_x():", "    f() == pytest.approx(0.5)"),
         ("NO_ASSERT", 2), ("BARE_COMPARISON", 3)),
    case("tuple-assert", src("def test_x():", "    assert (f() == 1, 'msg')"), ("ALWAYS_TRUE", 2)),
    case("patch-object-no-spec",
         src("from unittest import mock", "def test_x():", "    with mock.patch.object(mock, 'sentinel') as m:",
             "        m.assert_not_called()"), ("PATCH_NO_SPEC", 3)),
    case("swallowed-assertion",
         src("def test_x():", "    try:", "        assert f() == 1", "    except AssertionError:", "        pass"),
         ("SWALLOWED_FAILURE", 4)),
    case("weak-only-assertion", src("def test_x():", "    assert f() is not None"), (("WEAK_ASSERT", 1))),
]

DETECTS = [
    case("no-assert", src("def test_x():", "    f()"), ("NO_ASSERT", 1)),
    case("truthy-string", src("def test_x():", "    assert 'documentation'"), ("ALWAYS_TRUE", 2)),
    case("nan-compare", src("import math", "def test_x():", "    assert f() != math.nan"), ("NAN_COMPARE", 3)),
    case("nan-float-call", src("def test_x():", "    assert f() == float('nan')"), ("NAN_COMPARE", 2)),
    case("swallowed-exception",
         src("def test_x():", "    try:", "        f()", "    except Exception:", "        pass", "    assert True is not False"),
         ("SWALLOWED_FAILURE", 4)),
    case("bare-except",
         src("def test_x():", "    try:", "        f()", "    except:", "        pass", "    assert g() == 1"),
         ("SWALLOWED_FAILURE", 4)),
    case("fake-mock-call", src("def test_x(m):", "    m.called_once_with(3)"),
         ("NO_ASSERT", 1), ("FAKE_MOCK_ASSERT", 2)),
    case("mock-typo", src("def test_x(m):", "    m.assert_called_once_wtih(3)"),
         ("NO_ASSERT", 1), ("FAKE_MOCK_ASSERT", 2)),
    case("broad-raises", src("import pytest", "def test_x():", "    with pytest.raises(Exception):", "        f()"),
         ("BROAD_RAISES", 3)),
    case("sleep", src("import time", "def test_x():", "    time.sleep(1)", "    assert f() == 1"), ("SLEEP", 3)),
    case("network", src("import requests", "def test_x():", "    assert requests.get('u').ok == 1"), ("NETWORK", 3)),
    case("unseeded", src("import random", "def test_x():", "    assert f(random.random()) == 1"),
         ("UNSEEDED_RANDOM", 3)),
    case("print", src("def test_x():", "    print(f())", "    assert f() == 1"), ("PRINT", 2)),
    case("for-over-cases", src("def test_x():", "    for c in [1, 2]:", "        assert f(c) == c"), ("BRANCHING", 2)),
    case("environ-assign", src("import os", "def test_x():", "    os.environ['A'] = '1'", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("environ-update", src("import os", "def test_x():", "    os.environ.update(A='1')", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("sys-path", src("import sys", "def test_x():", "    sys.path.insert(0, '.')", "    assert f() == 1"),
         ("SYS_PATH", 3)),
    case("skip-no-reason", src("import pytest", "@pytest.mark.skip", "def test_x():", "    assert f() == 1"),
         ("SKIP_NO_REASON", 2)),
    case("xfail-not-strict",
         src("import pytest", "@pytest.mark.xfail(reason='r')", "def test_x():", "    assert f() == 1"),
         ("XFAIL_NOT_STRICT", 2)),
    case("patch-no-spec",
         src("from unittest import mock", "def test_x():", "    with mock.patch('a.b') as m:", "        m.assert_not_called()"),
         ("PATCH_NO_SPEC", 3)),
    case("unittest",
         src("import unittest", "class T(unittest.TestCase):", "    def test_x(self):", "        self.assertEqual(f(), 1)"),
         ("UNITTEST_CLASS", 2)),
    case("module-state", src("SEEN = []", "def test_x():", "    SEEN.append(1)", "    assert len(SEEN) == 1"),
         ("MUTABLE_MODULE_STATE", 3)),
    case("shadowed", src("def test_x():", "    assert f() == 1", "def test_x():", "    assert f() == 2"),
         ("SHADOWED_TEST_NAME", 3)),
    case("syntax", src("def test_x(:"), ("SYNTAX", 1)),
    case("yield-fixture",
         src("import pytest", "@pytest.yield_fixture", "def res():", "    yield 1", "def test_x(res):", "    assert res == 1"),
         ("YIELD_FIXTURE", 2)),
    case("yield-fixture-call",
         src("import pytest", "@pytest.yield_fixture(scope='module')", "def res():", "    yield 1",
             "def test_x(res):", "    assert res == 1"), ("YIELD_FIXTURE", 2)),
]

# Aliased and from-imports: one helper resolves them for every rule that keys on a module name.
ALIASED = [
    case("pytest-alias-broad-raises", src("import pytest as pt", "def test_x():", "    with pt.raises(Exception):", "        f()"),
         ("BROAD_RAISES", 3)),
    case("from-import-raises", src("from pytest import raises", "def test_x():", "    with raises(Exception):", "        f()"),
         ("BROAD_RAISES", 3)),
    case("from-import-nan", src("from math import nan", "def test_x():", "    assert f() == nan"), ("NAN_COMPARE", 3)),
    case("numpy-alias-nan", src("import numpy as npy", "def test_x():", "    assert f() == npy.nan"), ("NAN_COMPARE", 3)),
    case("float-negative-nan", src("def test_x():", "    assert f() == float('-nan')"), ("NAN_COMPARE", 2)),
    case("float-padded-nan", src("def test_x():", "    assert f() == float(' nan ')"), ("NAN_COMPARE", 2)),
    case("from-import-sleep", src("from time import sleep", "def test_x():", "    sleep(1)", "    assert f() == 1"), ("SLEEP", 3)),
    case("aliased-from-import-sleep",
         src("from time import sleep as nap", "def test_x():", "    nap(1)", "    assert f() == 1"), ("SLEEP", 3)),
    case("module-alias-sleep", src("import time as t", "def test_x():", "    t.sleep(1)", "    assert f() == 1"), ("SLEEP", 3)),
    case("asyncio-sleep-nonzero",
         src("import asyncio", "async def test_x():", "    await asyncio.sleep(1)", "    assert f() == 1"), ("SLEEP", 3)),
    case("sleep-in-fixture",
         src("import pytest, time", "@pytest.fixture", "def slow():", "    time.sleep(1)", "def test_x(slow):", "    assert f() == 1"),
         ("SLEEP", 4)),
    case("from-import-isolated-async-testcase",
         src("from unittest import IsolatedAsyncioTestCase", "class T(IsolatedAsyncioTestCase):",
             "    async def test_x(self):", "        self.assertEqual(f(), 1)"), ("UNITTEST_CLASS", 2)),
    case("aliased-unittest", src("import unittest as ut", "class T(ut.TestCase):", "    def test_x(self):",
                                 "        self.assertEqual(f(), 1)"), ("UNITTEST_CLASS", 2)),
    case("indirect-unittest-subclass",
         src("import unittest", "class Base(unittest.TestCase):", "    pass", "class T(Base):", "    def test_x(self):",
             "        self.assertEqual(f(), 1)"), ("UNITTEST_CLASS", 2), ("UNITTEST_CLASS", 4)),
    case("from-import-environ", src("from os import environ", "def test_x():", "    environ['A'] = '1'", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("aliased-os-environ", src("import os as o", "def test_x():", "    o.environ['A'] = '1'", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("aliased-requests", src("import requests as r", "def test_x():", "    assert r.get('u').ok == 1"), ("NETWORK", 3)),
    case("from-import-requests-get", src("from requests import get", "def test_x():", "    assert get('u').ok == 1"),
         ("NETWORK", 3)),
    case("from-import-urlopen", src("from urllib.request import urlopen", "def test_x():", "    assert urlopen('u').status == 200"),
         ("NETWORK", 3)),
    case("from-import-random-choice", src("from random import choice", "def test_x():", "    assert f(choice([1])) == 1"),
         ("UNSEEDED_RANDOM", 3)),
    case("numpy-alias-random", src("import numpy as npy", "def test_x():", "    assert f(npy.random.rand()) == 1"),
         ("UNSEEDED_RANDOM", 3)),
    case("aliased-mock-module",
         src("import unittest.mock as um", "def test_x():", "    with um.patch('a.b') as m:", "        m.assert_not_called()"),
         ("PATCH_NO_SPEC", 3)),
    case("aliased-patch-function",
         src("from unittest.mock import patch as p", "def test_x():", "    with p('a.b') as m:", "        m.assert_not_called()"),
         ("PATCH_NO_SPEC", 3)),
    case("sys-path-from-import", src("from sys import path", "def test_x():", "    path.insert(0, 'x')", "    assert f() == 1"),
         ("SYS_PATH", 3)),
    case("sys-path-augassign", src("import sys", "def test_x():", "    sys.path += ['x']", "    assert f() == 1"), ("SYS_PATH", 3)),
    case("sys-path-remove", src("import sys", "def test_x():", "    sys.path.remove('x')", "    assert f() == 1"), ("SYS_PATH", 3)),
]

# Rules that plainly ought to catch these (found as false negatives in the 2026-11 audit).
FALSE_NEGATIVES = [
    case("bare-raises-statement", src("import pytest", "def test_x():", "    f()", "    pytest.raises(ValueError)"),
         ("NO_ASSERT", 2)),
    case("bare-warns-statement", src("import pytest", "def test_x():", "    f()", "    pytest.warns(UserWarning)"),
         ("NO_ASSERT", 2)),
    case("assert-or-true", src("def test_x():", "    assert f() == 1 or True"), ("ALWAYS_TRUE", 2)),
    case("assert-name-or-true", src("def test_x():", "    assert f() or True"), ("ALWAYS_TRUE", 2)),
    case("assert-constant-compare", src("def test_x():", "    assert 1 == 1"), ("ALWAYS_TRUE", 2)),
    case("assert-negative-number", src("def test_x():", "    assert -1"), ("ALWAYS_TRUE", 2)),
    case("assert-not-false", src("def test_x():", "    assert not False"), ("ALWAYS_TRUE", 2)),
    case("assert-len-ge-zero", src("def test_x():", "    assert len(f()) >= 0"), ("ALWAYS_TRUE", 2)),
    case("assert-fstring-literal", src("def test_x():", "    assert f'abc'"), ("ALWAYS_TRUE", 2)),
    case("assert-lambda", src("def test_x():", "    assert lambda: 0"), ("ALWAYS_TRUE", 2)),
    case("suppress-assertion-error",
         src("import contextlib", "def test_x():", "    with contextlib.suppress(AssertionError):", "        assert f() == 1"),
         ("SWALLOWED_FAILURE", 3)),
    case("suppress-exception-from-import",
         src("from contextlib import suppress", "def test_x():", "    with suppress(Exception):", "        f()", "    assert g() == 1"),
         ("SWALLOWED_FAILURE", 3)),
    case("mock-assert-without-parentheses", src("def test_x(m):", "    f()", "    m.assert_called_once"),
         ("NO_ASSERT", 1), ("FAKE_MOCK_ASSERT", 3)),
    case("assert-on-mock-method-object", src("def test_x(m):", "    f()", "    assert m.assert_called_once"),
         ("FAKE_MOCK_ASSERT", 3)),
    case("assert-not-on-fake-attribute", src("def test_x(m):", "    f()", "    assert not m.not_called"),
         ("FAKE_MOCK_ASSERT", 3)),
    case("mock-typo-has-calls", src("def test_x(m):", "    m.assert_has_cals([])"), ("NO_ASSERT", 1), ("FAKE_MOCK_ASSERT", 2)),
    case("assert-after-return", src("def test_x():", "    return", "    assert f() == 1"), ("NO_ASSERT", 1)),
    case("assert-after-raise", src("def test_x():", "    raise RuntimeError", "    assert f() == 1"), ("NO_ASSERT", 1)),
    case("assert-in-uncalled-nested-function",
         src("def test_x():", "    def check():", "        assert f() == 1", "    f()"), ("NO_ASSERT", 1)),
    case("raises-match-empty",
         src("import pytest", "def test_x():", "    with pytest.raises(Exception, match=''):", "        f()"), ("BROAD_RAISES", 3)),
    case("raises-match-dot-star",
         src("import pytest", "def test_x():", "    with pytest.raises(Exception, match='.*'):", "        f()"), ("BROAD_RAISES", 3)),
    case("raises-match-none",
         src("import pytest", "def test_x():", "    with pytest.raises(Exception, match=None):", "        f()"), ("BROAD_RAISES", 3)),
    case("raises-keyword-exception",
         src("import pytest", "def test_x():", "    with pytest.raises(expected_exception=Exception):", "        f()"), ("BROAD_RAISES", 3)),
    case("raises-exception-in-tuple",
         src("import pytest", "def test_x():", "    with pytest.raises((ValueError, Exception)):", "        f()"), ("BROAD_RAISES", 3)),
    case("environ-augassign", src("import os", "def test_x():", "    os.environ['PATH'] += ':/x'", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("environ-tuple-target",
         src("import os", "def test_x():", "    os.environ['B'], other = '2', 3", "    assert f() == other"), ("ENV_MUTATION", 3)),
    case("environ-annotated-target", src("import os", "def test_x():", "    os.environ['A']: str = '1'", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("environ-or-assign", src("import os", "def test_x():", "    os.environ |= {'A': '1'}", "    assert f() == 1"),
         ("ENV_MUTATION", 3)),
    case("environ-popitem", src("import os", "def test_x():", "    os.environ.popitem()", "    assert f() == 1"), ("ENV_MUTATION", 3)),
    case("module-state-annotated",
         src("SEEN: list[str] = []", "def test_x():", "    SEEN.append('a')", "    assert SEEN == ['a']"),
         ("MUTABLE_MODULE_STATE", 3)),
    case("module-state-sort", src("ITEMS = [3, 1]", "def test_x():", "    ITEMS.sort()", "    assert ITEMS == [1, 3]"),
         ("MUTABLE_MODULE_STATE", 3)),
    case("module-state-nested-subscript",
         src("COUNTS = {}", "def test_x():", "    COUNTS['a']['b'] = 1", "    assert COUNTS['a']['b'] == 1"),
         ("MUTABLE_MODULE_STATE", 3)),
    case("module-state-global-augassign",
         src("LOG = []", "def test_x():", "    global LOG", "    LOG += [1]", "    assert LOG == [1]"),
         ("MUTABLE_MODULE_STATE", 4)),
    case("xfail-strict-false",
         src("import pytest", "@pytest.mark.xfail(reason='r', strict=False)", "def test_x():", "    assert f() == 1"),
         ("XFAIL_NOT_STRICT", 2)),
    case("skip-empty-reason",
         src("import pytest", "@pytest.mark.skip(reason='')", "def test_x():", "    assert f() == 1"), ("SKIP_NO_REASON", 2)),
    case("skip-empty-positional-reason",
         src("import pytest", "@pytest.mark.skip('')", "def test_x():", "    assert f() == 1"), ("SKIP_NO_REASON", 2)),
    case("skipif-no-reason",
         src("import pytest", "@pytest.mark.skipif(True)", "def test_x():", "    assert f() == 1"), ("SKIP_NO_REASON", 2)),
    case("class-level-skip-no-reason",
         src("import pytest", "@pytest.mark.skip", "class TestA:", "    def test_x(self):", "        assert f() == 1"),
         ("SKIP_NO_REASON", 2)),
    case("module-pytestmark-skip-no-reason",
         src("import pytest", "pytestmark = pytest.mark.skip", "def test_x():", "    assert f() == 1"), ("SKIP_NO_REASON", 2)),
    case("module-pytestmark-list-skipif-no-reason",
         src("import pytest", "pytestmark = [pytest.mark.skipif(True)]", "def test_x():", "    assert f() == 1"),
         ("SKIP_NO_REASON", 2)),
    case("class-pytestmark-skip-no-reason",
         src("import pytest", "class TestA:", "    pytestmark = pytest.mark.skip", "    def test_x(self):",
             "        assert f() == 1"), ("SKIP_NO_REASON", 3)),
    case("class-level-xfail-not-strict",
         src("import pytest", "@pytest.mark.xfail(reason='r')", "class TestA:", "    def test_x(self):",
             "        assert f() == 1"), ("XFAIL_NOT_STRICT", 2)),
    case("autospec-false",
         src("from unittest import mock", "def test_x():", "    with mock.patch('a.b', autospec=False) as m:",
             "        m.assert_not_called()"), ("PATCH_NO_SPEC", 3)),
    case("float-sum", src("def test_x():", "    assert f() == 0.1 + 0.2"), ("FLOAT_EQ", 2)),
    case("float-in-list", src("def test_x():", "    assert f() == [0.1, 0.2]"), ("FLOAT_EQ", 2)),
    case("float-in-tuple-membership", src("def test_x():", "    assert f() in (0.1, 0.2)"), ("FLOAT_EQ", 2)),
    case("float-from-string", src("def test_x():", "    assert f() == float('0.1')"), ("FLOAT_EQ", 2)),
    case("float-true-division", src("def test_x():", "    assert f() == 1 / 3"), ("FLOAT_EQ", 2)),
    case("weak-not-equal-none", src("def test_x():", "    assert f() != None"), (("WEAK_ASSERT", 1))),
    case("weak-none-on-left", src("def test_x():", "    assert None is not f()"), (("WEAK_ASSERT", 1))),
    case("weak-not-is-none", src("def test_x():", "    assert not f() is None"), (("WEAK_ASSERT", 1))),
    case("weak-zero-less-than-len", src("def test_x():", "    assert 0 < len(f())"), (("WEAK_ASSERT", 1))),
    case("weak-bool-call", src("def test_x():", "    assert bool(f())"), (("WEAK_ASSERT", 1))),
    case("bare-boolop-comparison", src("def test_x():", "    f() == 1 and g() == 2"),
         ("NO_ASSERT", 1), ("BARE_COMPARISON", 2)),
    case("bare-negated-comparison", src("def test_x():", "    not (f() == 1)"), ("NO_ASSERT", 1), ("BARE_COMPARISON", 2)),
]

CLEAN = [
    pytest.param(src("import pytest", "def test_x():", "    with pytest.RaisesGroup(ValueError):", "        f()"), id="raises-group-asserts"),
    pytest.param(src("import pytest", "def test_x():", "    with pytest.RaisesExc(ValueError, match='^x$'):", "        f()"),
                 id="raises-exc-asserts"),
    pytest.param(src("import asyncio", "async def test_x():", "    await asyncio.sleep(0)", "    assert f() == 1"), id="sleep-zero-yields"),
    pytest.param(src("def test_x(sleep):", "    sleep(1)", "    assert f() == 1"), id="injected-sleep-fixture-is-not-time-sleep"),
    pytest.param(src("class TestA:", "    def test_x(self):", "        assert f() == 1", "class TestB:", "    def test_x(self):",
                     "        assert f() == 1"), id="same-name-different-class"),
    pytest.param(src("import pytest", "@pytest.mark.skip('tracked in #1')", "def test_x():", "    assert f() == 1"),
                 id="skip-positional-reason"),
    pytest.param(src("import pytest", "@pytest.mark.skip(**KW)", "def test_x():", "    assert f() == 1"), id="skip-kwargs-may-hold-reason"),
    pytest.param(src("import pytest", "@pytest.mark.skipif(True, reason='r')", "def test_x():", "    assert f() == 1"),
                 id="skipif-with-reason"),
    pytest.param(src("import pytest", "@pytest.mark.skip(reason='r')", "class TestA:", "    def test_x(self):", "        assert f() == 1"),
                 id="class-skip-with-reason"),
    pytest.param(src("import pytest", "@pytest.mark.xfail(reason='r', strict=True)", "def test_x():", "    assert f() == 1"),
                 id="xfail-strict-true"),
    pytest.param(src("import pytest", "@pytest.mark.xfail(reason='r', strict=STRICT)", "def test_x():", "    assert f() == 1"),
                 id="xfail-strict-variable"),
    pytest.param(src("import random", "def test_x():", "    rng = random.Random(7)", "    assert f(rng.random()) == 1"),
                 id="explicitly-seeded-rng"),
    pytest.param(src("import numpy as np", "def test_x():", "    rng = np.random.default_rng(seed=42)", "    assert f(rng.random()) == 1"),
                 id="default-rng-keyword-seed"),
    pytest.param(src("import numpy as np", "def test_x():", "    rs = np.random.RandomState(seed=0)", "    assert f(rs.rand()) == 1"),
                 id="random-state-keyword-seed"),
    pytest.param(src("import random", "def test_x():", "    random.seed(0)", "    assert f(random.random()) == 1"),
                 id="test-seeds-itself"),
    pytest.param(src("import numpy as np", "def test_x():", "    np.random.seed(0)", "    assert f(np.random.rand()) == 1"),
                 id="test-seeds-numpy-itself"),
    pytest.param(src("import random", "random.seed(1)", "def test_x():", "    assert f(random.random()) == 1"),
                 id="module-level-seed-in-test-file"),
    pytest.param(src("import random, pytest", "@pytest.fixture(autouse=True)", "def _seed():", "    random.seed(1)",
                     "def test_x():", "    assert f(random.random()) == 1"), id="autouse-seed-fixture-in-test-file"),
    pytest.param(src("def test_x():", "    try:", "        f()", "    except ValueError:", "        return", "    assert g() == 1"),
                 id="specific-except-is-fine"),
    pytest.param(src("import contextlib", "def test_x():", "    with contextlib.suppress(KeyError):", "        f()", "    assert g() == 1"),
                 id="suppress-specific-exception"),
    pytest.param(src("def test_x():", "    result = f()", "    assert result is not None", "    assert result.total == 3"),
                 id="weak-plus-strong"),
    pytest.param(src("def test_x():", "    assert len(f()) > 1"), id="len-greater-than-one-is-a-real-check"),
    pytest.param(src("from unittest import mock", "def test_x():",
                     "    with mock.patch.object(mock, 'sentinel', autospec=True) as m:", "        m.assert_not_called()"),
                 id="patch-object-autospec"),
    pytest.param(src("from unittest import mock", "def test_x():", "    with mock.patch('a.b', new_obj) as m:",
                     "        m.assert_not_called()"), id="patch-positional-new"),
    pytest.param(src("from unittest import mock", "def test_x():", "    with mock.patch('a.b', autospec=True) as m:",
                     "        m.assert_not_called()"), id="patch-autospec-true"),
    pytest.param(src("import pytest", "def test_x():", "    assert f() == pytest.approx(0.5)"), id="approx-asserted"),
    pytest.param(src("def test_x():", "    assert f() == float('inf')"), id="infinity-comparison-is-exact"),
    pytest.param(src("def test_x():", "    assert f() in (1, 2)"), id="int-membership"),
    pytest.param(src("def test_x():", "    assert f() == 2 * 3"), id="int-arithmetic"),
    pytest.param(src("import pytest, requests", "@pytest.mark.integration", "def test_x():",
                     "    assert requests.get('u').status_code == 200"), id="integration-may-use-network"),
    pytest.param(src("import pytest, requests", "pytestmark = pytest.mark.integration", "def test_x():",
                     "    assert requests.get('u').status_code == 200"), id="module-pytestmark-integration"),
    pytest.param(src("import pytest, requests", "pytestmark: list = [pytest.mark.e2e]", "def test_x():",
                     "    assert requests.get('u').status_code == 200"), id="annotated-pytestmark-e2e"),
    pytest.param(src("import pytest, requests", "@pytest.mark.integration", "class TestA:", "    def test_x(self):",
                     "        assert requests.get('u').ok == 1"), id="class-level-integration-mark"),
    pytest.param(src("import pytest, requests", "@pytest.mark.e2e", "class TestA:", "    def test_x(self):",
                     "        assert requests.get('u').ok == 1"), id="class-level-e2e-mark"),
    pytest.param(src("import pytest, requests", "class TestA:", "    pytestmark = pytest.mark.integration",
                     "    def test_x(self):", "        assert requests.get('u').ok == 1"), id="class-body-pytestmark-integration"),
    pytest.param(src("import urllib.parse", "def test_x():", "    assert urllib.parse.urlparse('http://a/b').path == '/b'"),
                 id="urllib-parse-never-opens-a-socket"),
    pytest.param(src("import http", "def test_x():", "    assert http.HTTPStatus(200).phrase == 'OK'"), id="http-status-enum"),
    pytest.param(src("import httpx", "def test_x():", "    t = httpx.MockTransport(f)", "    assert t is not None and g(t) == 1"),
                 id="httpx-mock-transport"),
    pytest.param(src("import requests", "def test_x():", "    err = requests.exceptions.ConnectionError('boom')",
                     "    assert str(err) == 'boom'"), id="requests-exception-class"),
    pytest.param(src("import socket", "def test_x():", "    assert f(socket.gethostname()) == 1"), id="socket-gethostname-is-local"),
    pytest.param(src("import pytest", "def test_x():", "    with pytest.raises(Exception, match='boom'):", "        f()"),
                 id="broad-raises-with-match"),
    pytest.param(src("import pytest", "def test_x():", "    with pytest.raises(Exception, check=lambda e: 'x' in str(e)):", "        f()"),
                 id="broad-raises-with-check"),
    pytest.param(src("import pytest", "def test_x():", "    pytest.raises(ValueError, f, 1)"), id="raises-callable-form-runs"),
    pytest.param(src("import pytest", "def test_x():", "    with pytest.raises(ValueError):", "        f()"), id="raises-specific"),
    pytest.param(src("def test_x(m):", "    f()", "    m.assert_called_once()"), id="mock-assertion-with-parentheses"),
    pytest.param(src("def test_x():", "    assert_callable(f)"), id="helper-assert_callable-is-not-a-mock-typo"),
    pytest.param(src("def test_x():", "    assert_callback_registered(f)"), id="helper-assert_callback_registered"),
    pytest.param(src("def test_x():", "    assert [*f()]"), id="starred-list-can-be-empty"),
    pytest.param(src("def test_x():", "    assert {**f()}"), id="double-star-dict-can-be-empty"),
    pytest.param(src("def test_x():", "    assert f() == 1 or g() == 2"), id="or-of-real-checks"),
    pytest.param(src("def test_x():", "    assert f() == 1", "    f() == 1 and g()"), id="comparison-as-guard-is-not-bare"),
    pytest.param(src("def test_x():", "    def check():", "        assert f() == 1", "    check()"), id="nested-function-that-is-called"),
    pytest.param(src("def test_x():", "    assert f() == 1", "    return"), id="return-after-assert"),
    pytest.param(src("import os", "from unittest import mock", "def test_x():", "    with mock.patch.dict(os.environ, {'A': '1'}):",
                     "        os.environ['B'] = '2'", "        assert f() == 1"), id="patch-dict-restores-environ"),
    pytest.param(src("import os", "def test_x():", "    assert os.environ['HOME'] == f()"), id="reading-environ"),
    pytest.param(src("TABLE = {}", "for i in range(3):", "    TABLE[i] = i * i", "def test_x():", "    assert TABLE[2] == 4"),
                 id="table-filled-at-import-time"),
    pytest.param(src("ITEMS = []", "def test_x():", "    ITEMS = []", "    ITEMS.append(1)", "    assert ITEMS == [1]"),
                 id="local-name-shadows-module-list"),
    pytest.param(src("ITEMS = []", "def test_x(ITEMS):", "    ITEMS.append(1)", "    assert ITEMS == [1]"),
                 id="parameter-shadows-module-list"),
    pytest.param("﻿" + src("def test_x():", "    assert f() == 1"), id="utf8-bom-is-valid-python"),
    pytest.param(src("# -*- coding: latin-1 -*-", "def test_x():", "    assert f() == 'caf\xe9'"), id="coding-cookie"),
]


@pytest.mark.parametrize(("source", "expected"), HOLES + DETECTS + ALIASED + FALSE_NEGATIVES)
def test_audit_reports_exactly_these_findings(run_findings, source, expected):
    findings = run_findings(source)
    assert [(f.code, f.line) for f in findings] == expected
    assert all(f.severity == SEVERITY[f.code] for f in findings)


@pytest.mark.parametrize("source", CLEAN)
def test_audit_stays_quiet_on_correct_code(run_audit, source):
    assert run_audit(source) == []


def test_every_code_has_a_detecting_case_with_the_right_severity(audit_mod, run_findings):
    seen = set(CODES_EXPECTED_BY_CASES)
    seen.add("SUPPRESSION_NO_REASON")  # covered by test_pragma_without_reason_is_itself_a_finding
    assert seen == set(audit_mod.CODES)
    assert {code: severity for code, (severity, _) in audit_mod.CODES.items()} == SEVERITY


def test_rule_code_count_is_pinned_to_the_code_and_the_docstring(audit_mod):
    assert len(audit_mod.CODES) == 26
    claimed = re.search(r"(\d+) rule codes", audit_mod.__doc__)
    assert claimed is not None, "the module docstring must state the number of rule codes"
    assert int(claimed.group(1)) == len(audit_mod.CODES)


def test_fake_mock_assert_message_does_not_claim_every_mock_attribute_is_auto_created(audit_mod):
    text = audit_mod.CODES["FAKE_MOCK_ASSERT"][1]
    assert "auto-creates any attribute" not in text
    assert "3.12" in text and "unsafe=True" in text and "parentheses" in text


def test_network_description_does_not_claim_a_real_call(audit_mod):
    assert "real network call" not in audit_mod.CODES["NETWORK"][1]


def test_xfail_description_mentions_explicit_opt_out(audit_mod):
    assert "strict=False" in audit_mod.CODES["XFAIL_NOT_STRICT"][1]


def test_yield_fixture_message_keeps_the_pytest_10_or_11_removal_wording(audit_mod):
    # Confirmed against the pytest 9.2.0 changelog in the 2026-11 audit; do not reword.
    assert "pytest 10 or 11" in audit_mod.CODES["YIELD_FIXTURE"][1]


# Added after the first mutation run: behaviour no earlier case pinned.
MORE_DETECTS = [
    case("assert-nonempty-dict", src("def test_x():", "    assert {'a': 1}"), ("ALWAYS_TRUE", 2)),
    case("assert-nonempty-list", src("def test_x():", "    assert [1]"), ("ALWAYS_TRUE", 2)),
    case("assert-nonempty-set", src("def test_x():", "    assert {1}"), ("ALWAYS_TRUE", 2)),
    case("assert-true-and-true", src("def test_x():", "    assert 1 and 2"), ("ALWAYS_TRUE", 2)),
    case("assert-not-and-false", src("def test_x():", "    assert not (f() and False)"), ("ALWAYS_TRUE", 2)),
    case("assert-chained-constant-compare", src("def test_x():", "    assert 1 < 2 < 3"), ("ALWAYS_TRUE", 2)),
    case("weak-len-greater-than-zero", src("def test_x():", "    assert len(f()) > 0"), ("WEAK_ASSERT", 1)),
    case("weak-len-not-equal-zero", src("def test_x():", "    assert len(f()) != 0"), ("WEAK_ASSERT", 1)),
    case("weak-len-at-least-one", src("def test_x():", "    assert len(f()) >= 1"), ("WEAK_ASSERT", 1)),
    case("weak-and-of-weak-checks", src("def test_x():", "    assert f() is not None and isinstance(f(), int)"), ("WEAK_ASSERT", 1)),
    case("float-in-list-after-int", src("def test_x():", "    assert f() == [1, 0.5]"), ("FLOAT_EQ", 2)),
    case("float-in-dict-value", src("def test_x():", "    assert f() == {'a': 0.1}"), ("FLOAT_EQ", 2)),
    case("sleep-in-nested-helper-of-a-test",
         src("import time", "def test_x():", "    def helper():", "        time.sleep(1)", "    helper()", "    assert f() == 1"), ("SLEEP", 4)),
    case("sleep-in-fixture-with-two-decorators",
         src("import pytest, time", "@pytest.fixture", "@pytest.mark.usefixtures('x')", "def slow():", "    time.sleep(1)",
             "def test_x(slow):", "    assert f() == 1"), ("SLEEP", 5)),
    case("environ-patch-dict-of-something-else",
         src("import os", "from unittest import mock", "def test_x():", "    with mock.patch.dict(CONFIG, {}):",
             "        os.environ['B'] = '2'", "        assert f() == 1"), ("ENV_MUTATION", 5)),
    case("environ-mutated-after-the-patch-dict-block",
         src("import os", "from unittest import mock", "def test_x():", "    with mock.patch.dict(os.environ, {}):",
             "        f()", "    os.environ['B'] = '2'", "    assert f() == 1"), ("ENV_MUTATION", 6)),
    case("environ-patch-dict-decorator-covers_only_its_test",
         src("import os", "from unittest import mock", "@mock.patch.dict(os.environ, {})", "def test_x():",
             "    os.environ['B'] = '2'", "    assert f() == 1", "def test_y():", "    os.environ['C'] = '3'", "    assert f() == 1"),
         ("ENV_MUTATION", 8)),
    case("module-state-mutated-by-a-nested-def-in-a-with-block",
         src("ITEMS = []", "def test_x():", "    with ctx():", "        def inner():", "            ITEMS.append(1)",
             "    inner()", "    assert ITEMS == [1]"), ("MUTABLE_MODULE_STATE", 5)),
    case("shadowed-test-in-the-same-branch",
         src("if C:", "    def test_x():", "        assert f() == 1", "    def test_x():", "        assert f() == 2"), ("SHADOWED_TEST_NAME", 4)),
    case("shadowed-test-class",
         src("class TestA:", "    def test_x(self):", "        assert f() == 1", "class TestA:", "    def test_y(self):", "        assert f() == 1"),
         ("SHADOWED_TEST_NAME", 4)),
    case("branching-match-statement",
         src("def test_x():", "    match f():", "        case 1:", "            assert g() == 1"), ("BRANCHING", 2)),
    case("branching-async-for-over-a-name", src("async def test_x():", "    async for i in ITEMS:", "        assert i == 1"), ("BRANCHING", 2)),
    case("assert-not-or-of-falsy-constants", src("def test_x():", "    assert not (0 or '')"), ("ALWAYS_TRUE", 2)),
    case("float-in-dict-with-other-values", src("def test_x():", "    assert f() == {'a': 0.1, 'b': 1}"), ("FLOAT_EQ", 2)),
    case("with-block-that-is-not-an-assertion", src("def test_x():", "    with custom_ctx():", "        f()"), ("NO_ASSERT", 1)),
    case("except-exception-that-only-logs",
         src("def test_x():", "    try:", "        f()", "    except Exception as e:", "        log(e)", "    assert g() == 1"),
         ("SWALLOWED_FAILURE", 4)),
    case("unseeded-random-with-arguments-is-not-a-seed",
         src("import random", "def test_x():", "    assert f(random.randint(1, 5)) == 1"), ("UNSEEDED_RANDOM", 3)),
]

MORE_CLEAN = [
    pytest.param(src("def test_x():", "    assert 3 < 2 < 4"), id="chained-compare-that-is-false-is-not-always-true"),
    pytest.param(src("def test_x():", "    assert f() and True"), id="and-with-a-live-operand"),
    pytest.param(src("def test_x():", "    assert f() == x / 2"), id="division-by-a-variable-is-not-a-float-literal"),
    pytest.param(src("def test_x():", "    assert f() == float('abc')"), id="float-of-a-non-number-string"),
    pytest.param(src("import pytest", "@pytest.mark.xfail(reason='r', **KW)", "def test_x():", "    assert f() == 1"), id="xfail-reason-plus-kwargs"),
    pytest.param(src("def test_x():", "    assert not (f() or False)"), id="not-or-with-a-live-operand"),
    pytest.param(src("def test_x():", "    assert 0 < f()"), id="zero-less-than-a-value-is-a-real-check"),
    pytest.param(src("def test_x():", "    assert f() == float(g())"), id="float-of-a-variable"),
    pytest.param(src("import time", "def helper():", "    time.sleep(1)", "def test_x():", "    assert f() == 1"), id="sleep-in-a-plain-helper-is-not-checked"),
    pytest.param(src("import os", "from unittest import mock", "def test_x():", "    with mock.patch.dict('os.environ', {'A': '1'}):",
                     "        os.environ['B'] = '2'", "        assert f() == 1"), id="patch-dict-with-string-target"),
    pytest.param(src("ITEMS = []", "def test_x():", "    def inner():", "        ITEMS = []", "        ITEMS.append(1)",
                     "        return ITEMS", "    assert inner() == [1]"), id="nested-function-binds-the-name-itself"),
    pytest.param(src("ITEMS = []", "def test_x():", "    from m import ITEMS", "    ITEMS.append(1)", "    assert ITEMS"), id="local-import-shadows-module-list"),
    pytest.param(src("import pytest", "@pytest.mark.skip(reason=WHY)", "def test_x():", "    assert f() == 1"), id="skip-reason-from-a-variable"),
    pytest.param(src("import pytest", "@pytest.mark.xfail(**KW)", "def test_x():", "    assert f() == 1"), id="xfail-kwargs-may-carry-strict"),
    pytest.param(src("import pytest", "def test_x():", "    pytest.raises(ValueError, f)"), id="raises-callable-form-without-arguments"),
    pytest.param(src("def test_x():", "    with expect_raises(ValueError):", "        f()"), id="project-helper-expect_raises"),
    pytest.param(src("def test_x():", "    with expect_warns(UserWarning):", "        f()"), id="project-helper-expect_warns"),
    pytest.param(src("from unittest import mock", "def test_x():", "    with mock.patch('a.b', **OPTS) as m:", "        m.assert_not_called()"), id="patch-kwargs-may-carry-autospec"),
    pytest.param(src("from unittest import mock", "def test_x():", "    with mock.patch('a.b', new=OBJ) as m:", "        m.assert_not_called()"), id="patch-new-keyword"),
    pytest.param(src("from unittest import mock", "def test_x():", "    with mock.patch('a.b', new_callable=mock.MagicMock) as m:", "        m.assert_not_called()"),
                 id="patch-new_callable"),
    pytest.param(src("from unittest import mock", "def test_x():", "    with mock.patch.object(OBJ, 'attr', NEW) as m:", "        m.assert_not_called()"),
                 id="patch-object-positional-new"),
    pytest.param(src("def test_x():", "    try:", "        f()", "    except Exception:", "        raise", "    assert g() == 1"), id="except-exception-that-reraises"),
    pytest.param(src("def test_x():", "    try:", "        f()", "    except Exception as e:", "        log(e)", "        raise", "    assert g() == 1"),
                 id="except-exception-logs-then-reraises"),
    pytest.param(src("import pytest", "def test_x():", "    try:", "        f()", "    except Exception as e:", "        log(e)",
                     "        pytest.fail('boom')", "    assert g() == 1"), id="except-exception-then-pytest-fail"),
    pytest.param(src("import sys", "if sys.version_info >= (3, 11):", "    def test_x():", "        assert f() == 1", "else:",
                     "    def test_x():", "        assert f() == 2"), id="conditional-definition-of-a-test"),
    pytest.param(src("def test_x():", "    for i in range(3):", "        assert f(i) == i"), id="for-over-range-is-not-a-case-list"),
    pytest.param(src("async def test_x():", "    async for i in agen():", "        assert i == 1"), id="async-for-over-a-call"),
]


@pytest.mark.parametrize(("source", "expected"), MORE_DETECTS)
def test_audit_reports_exactly_these_findings_more(run_findings, source, expected):
    findings = run_findings(source)
    assert [(f.code, f.line) for f in findings] == expected
    assert all(f.severity == SEVERITY[f.code] for f in findings)


@pytest.mark.parametrize("source", MORE_CLEAN)
def test_audit_stays_quiet_on_more_correct_code(run_audit, source):
    assert run_audit(source) == []


def test_config_that_does_not_parse_in_tox_ini_or_setup_cfg_is_skipped_with_a_warning(audit_mod, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    (tmp_path / "tox.ini").write_text("[testenv\nfoo\n")
    (tmp_path / "setup.cfg").write_text("[tool:pytest\nxfail_strict = true\n")
    (tmp_path / "test_a.py").write_text(XFAIL)
    assert audit_mod.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "tox.ini" in err and "setup.cfg" in err


def test_conftest_that_does_not_parse_cannot_seed_and_is_reported(run_audit):
    assert run_audit(SEEDING, extra={"conftest.py": "import random\nrandom.seed(1\n"}) == ["SYNTAX", "UNSEEDED_RANDOM"]


def test_def_line_pragma_covers_a_finding_on_the_decorator_line(run_findings):
    source = src("import pytest", "@pytest.mark.skip", "def test_x():  # audit: ignore[SKIP_NO_REASON] why", "    assert f() == 1")
    assert run_findings(source) == []


def test_finding_without_a_line_is_reported_as_line_1_in_sarif_and_github(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {})
    (tmp_path / "test_a.py").write_bytes(b"# -*- coding: nonesuch -*-\ndef test_x(): pass\n")
    assert audit_mod.main([str(tmp_path), "--format", "json"]) == 1
    (finding,) = json.loads(capsys.readouterr().out)
    assert (finding["code"], finding["line"]) == ("SYNTAX", 0)
    audit_mod.main([str(tmp_path), "--format", "sarif"])
    region = json.loads(capsys.readouterr().out)["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 1
    audit_mod.main([str(tmp_path), "--format", "github"])
    assert ",line=1," in capsys.readouterr().out


def test_github_message_keeps_its_commas_unescaped(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    print(f())", "    assert f() == 1")})
    audit_mod.main([str(tmp_path), "--format", "github"])
    assert "print() in a test; assert instead, or use capsys/caplog" in capsys.readouterr().out



# ----------------------------------------------------------------------------- integration paths
def test_integration_directory_exempts_network(run_audit):
    source = src("import requests", "def test_x():", "    assert requests.get('u').status_code == 200")
    assert run_audit(source, name="integration/test_api.py") == []


def test_e2e_directory_exempts_network(run_audit):
    source = src("import requests", "def test_x():", "    assert requests.get('u').status_code == 200")
    assert run_audit(source, name="tests/E2E/test_api.py") == []


def test_network_in_an_ordinary_directory_is_flagged(run_audit):
    source = src("import requests", "def test_x():", "    assert requests.get('u').status_code == 200")
    assert run_audit(source, name="tests/test_api.py") == ["NETWORK"]


def test_integration_ancestor_above_the_project_does_not_exempt(run_audit):
    source = src("import requests", "def test_x():", "    assert requests.get('u').status_code == 200")
    assert run_audit(source, under="integration/proj") == ["NETWORK"]


def test_integration_directory_below_the_project_exempts_even_when_auditing_inside_it(run_audit):
    source = src("import requests", "def test_x():", "    assert requests.get('u').status_code == 200")
    assert run_audit(source, name="tests/integration/test_api.py", root="tests/integration") == []


# ----------------------------------------------------------------------------- seeding
SEEDING = src("import random", "def test_x():", "    assert f(random.random()) == 1")


@pytest.mark.parametrize(
    "conftest",
    [
        pytest.param("import random\nrandom.seed(1)\n", id="module-level-random-seed"),
        pytest.param("import numpy as np\nnp.random.seed(1)\n", id="numpy-seed"),
        pytest.param("import torch\ntorch.manual_seed(1)\n", id="torch-manual-seed"),
        pytest.param("import random, pytest\n@pytest.fixture(autouse=True)\ndef _s():\n    random.seed(0)\n", id="autouse-fixture"),
        pytest.param("import random\ndef pytest_configure(config):\n    random.seed(0)\n", id="pytest-hook"),
        pytest.param("from random import seed\nseed(3)\n", id="from-import-seed"),
    ],
)
def test_seeding_conftest_silences_unseeded_random(run_audit, conftest):
    assert run_audit(SEEDING, extra={"conftest.py": conftest}) == []


@pytest.mark.parametrize(
    "conftest",
    [
        pytest.param("def f(mp):\n    mp.setenv('PYTHONHASHSEED', '0')\n", id="pythonhashseed-only"),
        pytest.param("import random\n# random.seed(0)\n", id="commented-out-seed-call"),
        pytest.param("import random\ndef seed_it():\n    random.seed(0)\n", id="seed-function-never-called"),
        pytest.param("import random, pytest\n@pytest.fixture\ndef seeded():\n    random.seed(0)\n", id="non-autouse-fixture"),
        pytest.param("import random\nx = 'random.seed(0)'\n", id="seed-call-in-a-string"),
        pytest.param("import random\nrandom.seed()\n", id="seed-without-argument"),
    ],
)
def test_conftest_that_does_not_really_seed_does_not_count(run_audit, conftest):
    assert run_audit(SEEDING, extra={"conftest.py": conftest}) == ["UNSEEDED_RANDOM"]


def test_sibling_directory_conftest_does_not_seed_this_test(run_audit):
    extra = {"b/conftest.py": "import random\nrandom.seed(1)\n"}
    assert run_audit(SEEDING, name="a/test_a.py", extra=extra) == ["UNSEEDED_RANDOM"]


def test_conftest_next_to_the_test_seeds_it(run_audit):
    extra = {"a/conftest.py": "import random\nrandom.seed(1)\n"}
    assert run_audit(SEEDING, name="a/test_a.py", extra=extra) == []


def test_virtualenv_conftest_does_not_seed_the_suite(run_audit):
    extra = {".venv/lib/site-packages/pkg/conftest.py": "import random\nrandom.seed(1)\n"}
    assert run_audit(SEEDING, extra=extra) == ["UNSEEDED_RANDOM"]


def test_ancestor_conftest_inside_the_project_seeds_a_narrower_audit(run_audit):
    extra = {"conftest.py": "import random\nrandom.seed(1)\n"}
    assert run_audit(SEEDING, name="tests/test_a.py", extra=extra, root="tests") == []


def test_conftest_above_the_project_root_cannot_silence_the_suite(run_audit, tmp_path):
    (tmp_path / "conftest.py").write_text("import random\nrandom.seed(1)\n")
    assert run_audit(SEEDING, under="proj") == ["UNSEEDED_RANDOM"]


# ----------------------------------------------------------------------------- config (pytest 9.2.0)
XFAIL = src("import pytest", "@pytest.mark.xfail(reason='bug #3')", "def test_x():", "    assert f() == 1")
STRICT_INI = "[pytest]\nxfail_strict = true\n"
LAX_INI = "[pytest]\nxfail_strict = false\n"
STRICT_PYPROJECT = "[tool.pytest.ini_options]\nstrict = true\n"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param({"pytest.ini": STRICT_INI}, [], id="pytest.ini"),
        pytest.param({".pytest.ini": STRICT_INI}, [], id=".pytest.ini"),
        pytest.param({"pytest.ini": "[pytest]\nxfail_strict = yes\n"}, [], id="ini-yes"),
        pytest.param({"pytest.ini": "[pytest]\nstrict_xfail = on\n"}, [], id="ini-strict_xfail-on"),
        pytest.param({"pytest.toml": "[pytest]\nstrict_xfail = true\n"}, [], id="pytest.toml"),
        pytest.param({".pytest.toml": "[pytest]\nxfail_strict = true\n"}, [], id=".pytest.toml"),
        pytest.param({"pyproject.toml": STRICT_PYPROJECT}, [], id="pyproject-ini_options"),
        pytest.param({"pyproject.toml": "[tool.pytest]\nstrict = true\n"}, [], id="pyproject-toml-mode"),
        pytest.param({"tox.ini": STRICT_INI}, [], id="tox.ini"),
        pytest.param({"setup.cfg": "[tool:pytest]\nxfail_strict = 1\n"}, [], id="setup.cfg"),
        pytest.param({"pytest.ini": LAX_INI}, ["XFAIL_NOT_STRICT"], id="ini-false"),
        pytest.param({}, ["XFAIL_NOT_STRICT"], id="no-config"),
        pytest.param({"pyproject.toml": "[tool.pytest.ini_options]\nstrict = true\nxfail_strict = false\n"},
                     ["XFAIL_NOT_STRICT"], id="explicit-xfail_strict-false-beats-strict-true"),
        pytest.param({"pyproject.toml": "[tool.pytest]\nstrict = true\nstrict_xfail = false\n"},
                     ["XFAIL_NOT_STRICT"], id="explicit-strict_xfail-false-beats-strict-true-toml"),
        pytest.param({"pytest.ini": "[pytest]\nstrict = true\nstrict_xfail = false\n"},
                     ["XFAIL_NOT_STRICT"], id="ini-strict_xfail-false-beats-strict"),
        pytest.param({"pytest.ini": "[pytest]\naddopts = -q\n", "pyproject.toml": STRICT_PYPROJECT},
                     ["XFAIL_NOT_STRICT"], id="pytest.ini-wins-over-pyproject"),
        pytest.param({"pytest.ini": "# no section at all\n", "pyproject.toml": STRICT_PYPROJECT},
                     ["XFAIL_NOT_STRICT"], id="pytest.ini-without-section-still-wins"),
        pytest.param({".pytest.ini": STRICT_INI, "pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-q'\n"},
                     [], id=".pytest.ini-wins-over-pyproject"),
        pytest.param({"pytest.toml": "[pytest]\nxfail_strict = false\n", "pytest.ini": STRICT_INI},
                     ["XFAIL_NOT_STRICT"], id="pytest.toml-wins-over-pytest.ini"),
        pytest.param({"pytest.toml": "[pytest]\nxfail_strict = false\n", ".pytest.toml": "[pytest]\nstrict = true\n"},
                     ["XFAIL_NOT_STRICT"], id="pytest.toml-wins-over-.pytest.toml"),
        pytest.param({"pytest.ini": STRICT_INI, ".pytest.ini": LAX_INI}, [], id="pytest.ini-wins-over-.pytest.ini"),
        pytest.param({"pyproject.toml": STRICT_PYPROJECT, "tox.ini": LAX_INI}, [], id="pyproject-wins-over-tox.ini"),
        pytest.param({"pyproject.toml": "[project]\nname = 'x'\n", "tox.ini": STRICT_INI}, [],
                     id="pyproject-without-pytest-tables-is-skipped"),
        pytest.param({"pyproject.toml": "[tool.pytest.ini_options]\n", "tox.ini": STRICT_INI},
                     ["XFAIL_NOT_STRICT"], id="empty-ini_options-table-still-wins"),
        pytest.param({"tox.ini": "[testenv]\ndeps = pytest\n", "setup.cfg": "[tool:pytest]\nxfail_strict = true\n"}, [],
                     id="tox.ini-without-pytest-section-is-skipped"),
        pytest.param({"tox.ini": LAX_INI, "setup.cfg": "[tool:pytest]\nxfail_strict = true\n"},
                     ["XFAIL_NOT_STRICT"], id="tox.ini-wins-over-setup.cfg"),
    ],
)
def test_xfail_strictness_follows_the_config_pytest_9_2_would_read(run_audit, files, expected):
    assert run_audit(XFAIL, extra=files) == expected


def test_xfail_strict_from_pyproject_silences_xfail_check(run_audit):
    assert run_audit(XFAIL, extra={"pyproject.toml": STRICT_PYPROJECT}) == []


def test_explicit_strict_false_is_flagged_even_when_the_config_is_strict(run_audit):
    source = src("import pytest", "@pytest.mark.xfail(reason='r', strict=False)", "def test_x():", "    assert f() == 1")
    assert run_audit(source, extra={"pytest.ini": STRICT_INI}) == ["XFAIL_NOT_STRICT"]


def test_config_in_a_parent_of_the_audited_directory_is_found(run_audit):
    assert run_audit(XFAIL, name="tests/test_x.py", extra={"pytest.ini": STRICT_INI}, root="tests") == []


def test_config_above_the_project_root_is_not_read(run_audit, tmp_path):
    (tmp_path / "pytest.ini").write_text(STRICT_INI)
    assert run_audit(XFAIL, under="proj") == ["XFAIL_NOT_STRICT"]


def test_without_tomllib_a_toml_config_gives_a_loud_warning_not_silence(audit_mod, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit_mod, "tomllib", None)
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(STRICT_PYPROJECT)
    (tmp_path / "test_a.py").write_text(XFAIL)
    assert audit_mod.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "pyproject.toml" in err and "tomllib" in err


def test_malformed_pytest_ini_is_an_audit_error_not_a_silent_default(audit_mod, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    (tmp_path / "pytest.ini").write_text("key = value before any section\n")
    (tmp_path / "test_a.py").write_text(src("def test_x():", "    assert f() == 1"))
    assert audit_mod.main([str(tmp_path)]) == 2
    assert "pytest.ini" in capsys.readouterr().err


# ----------------------------------------------------------------------------- hermetic fixture
def test_run_audit_ignores_hostile_ancestors_of_the_temp_dir(run_audit, tmp_path):
    """Regression: with --basetemp under a seeding conftest, a strict pyproject or a directory
    named integration, three of the self-tests used to fail."""
    (tmp_path / "conftest.py").write_text("import random\nrandom.seed(1)\n")
    (tmp_path / "pyproject.toml").write_text(STRICT_PYPROJECT)
    source = src(
        "import random, requests, pytest",
        "@pytest.mark.xfail(reason='r')",
        "def test_x():",
        "    assert f(random.random(), requests.get('u')) == 1",
    )
    codes = run_audit(source, under="integration/proj")
    assert codes == ["NETWORK", "UNSEEDED_RANDOM", "XFAIL_NOT_STRICT"]


# ----------------------------------------------------------------------------- path as typed
def make_project(root, files: dict[str, str]):
    (root / ".git").mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def codes_from_main(audit_mod, capsys, *argv: str) -> tuple[int, list[tuple[str, int]]]:
    exit_code = audit_mod.main([*argv, "--format", "json"])
    out = capsys.readouterr().out
    return exit_code, [(f["code"], f["line"]) for f in json.loads(out)] if out.strip().startswith("[") else []


def test_dot_relative_and_absolute_paths_give_the_same_answer(audit_mod, tmp_path, monkeypatch, capsys):
    project = make_project(tmp_path / "proj", {
        "conftest.py": "import random\nrandom.seed(1)\n",
        "tests/test_a.py": src("import random, requests", "def test_x():",
                               "    assert f(random.random(), requests.get('u')) == 1"),
    })
    absolute = codes_from_main(audit_mod, capsys, str(project / "tests"))
    monkeypatch.chdir(project / "tests")
    dot = codes_from_main(audit_mod, capsys, ".")
    monkeypatch.chdir(project)
    relative = codes_from_main(audit_mod, capsys, "tests")
    assert absolute == dot == relative == (1, [("NETWORK", 3)])


def test_dot_inside_an_e2e_directory_is_exempt_exactly_like_the_absolute_path(audit_mod, tmp_path, monkeypatch, capsys):
    project = make_project(tmp_path / "proj", {
        "e2e/tests/test_net.py": src("import requests", "def test_x():", "    assert requests.get('u').ok == 1"),
    })
    absolute = codes_from_main(audit_mod, capsys, str(project / "e2e" / "tests"))
    monkeypatch.chdir(project / "e2e" / "tests")
    dot = codes_from_main(audit_mod, capsys, ".")
    assert absolute == dot == (0, [])


def test_dot_finds_config_and_conftest_in_real_parents(audit_mod, tmp_path, monkeypatch, capsys):
    project = make_project(tmp_path / "proj", {"pytest.ini": STRICT_INI, "tests/test_x.py": XFAIL})
    monkeypatch.chdir(project / "tests")
    assert codes_from_main(audit_mod, capsys, ".") == (0, [])


# ----------------------------------------------------------------------------- collection
BAD = src("def test_x():", "    f()")


def test_project_below_a_directory_named_build_is_still_audited(run_audit):
    assert run_audit(BAD, under="build/proj") == ["NO_ASSERT"]


@pytest.mark.parametrize("skipped", [".venv", "venv", ".tox", ".nox", "node_modules", "build", "dist", "mutants", "site-packages", ".git"])
def test_skipped_directories_below_the_root_are_not_audited(run_audit, skipped):
    extra = {f"{skipped}/test_hidden.py": BAD}
    assert run_audit(src("def test_x():", "    assert f() == 1"), extra=extra) == []


def test_only_test_named_files_and_conftests_are_collected(run_audit):
    extra = {"helper.py": BAD, "b_test.py": BAD, "pkg/conftest.py": src("def test_y():", "    f()")}
    assert run_audit(src("def test_x():", "    assert f() == 1"), extra=extra) == ["NO_ASSERT", "NO_ASSERT"]


def test_a_single_file_argument_is_audited_even_inside_a_skipped_looking_directory(audit_mod, tmp_path):
    target = tmp_path / "build" / "test_a.py"
    target.parent.mkdir()
    target.write_text(BAD)
    assert [f.code for f in audit_mod.audit(target)] == ["NO_ASSERT"]


def test_a_root_with_no_test_files_is_not_clean(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"README.md": "x", "helper.py": BAD})
    assert audit_mod.main([str(tmp_path)]) == 2
    err = capsys.readouterr()
    assert "no test files" in err.err
    assert "clean" not in err.out


def test_a_root_whose_tests_are_all_in_skipped_directories_is_not_clean(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {".venv/test_a.py": BAD})
    assert audit_mod.main([str(tmp_path)]) == 2
    assert "no test files" in capsys.readouterr().err


def test_audit_function_raises_instead_of_returning_empty_for_no_files(audit_mod, tmp_path):
    make_project(tmp_path, {"README.md": "x"})
    with pytest.raises(audit_mod.AuditError, match="no test files"):
        audit_mod.audit(tmp_path)


def test_directory_named_like_a_test_file_does_not_crash(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_real.py": BAD})
    (tmp_path / "test_dir.py").mkdir()
    (tmp_path / "test_dir.py" / "test_inner.py").write_text(BAD)
    assert audit_mod.main([str(tmp_path)]) == 1
    out = capsys.readouterr()
    assert "Traceback" not in out.err
    assert out.out.count("NO_ASSERT") >= 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_unreadable_test_file_is_exit_2_not_a_traceback(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_ok.py": src("def test_x():", "    assert f() == 1")})
    locked = tmp_path / "test_locked.py"
    locked.write_text(BAD)
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):  # audit: ignore[BRANCHING] whether chmod 0 is honoured is only known at run time (root ignores it)
            pytest.skip("running with privileges that ignore file modes")
        assert audit_mod.main([str(tmp_path)]) == 2
    finally:
        locked.chmod(0o644)
    err = capsys.readouterr().err
    assert "test_locked.py" in err and "Traceback" not in err


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_broken_symlink_named_like_a_test_is_exit_2(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_ok.py": src("def test_x():", "    assert f() == 1")})
    (tmp_path / "test_gone.py").symlink_to(tmp_path / "missing.py")
    assert audit_mod.main([str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "test_gone.py" in err and "Traceback" not in err


@pytest.mark.skipif(not os.path.exists("/dev/null"), reason="no /dev/null")
def test_path_that_is_neither_file_nor_directory_is_exit_2(audit_mod, capsys):
    assert audit_mod.main(["/dev/null"]) == 2
    assert "not a file or directory" in capsys.readouterr().err


def test_missing_path_is_exit_2(audit_mod, tmp_path, capsys):
    assert audit_mod.main([str(tmp_path / "nope")]) == 2
    assert "no such path" in capsys.readouterr().err


def test_no_path_is_exit_2(audit_mod, capsys):
    assert audit_mod.main([]) == 2
    assert capsys.readouterr().err == "audit_tests: a test directory or file is required (usage: audit_tests.py PATH)\n"


def test_missing_path_message_names_the_path_exactly(audit_mod, tmp_path, capsys):
    missing = tmp_path / "nope"
    assert audit_mod.main([str(missing)]) == 2
    assert capsys.readouterr().err == f"audit_tests: no such path: {missing}\n"


# ----------------------------------------------------------------------------- --ignore
def test_ignore_removes_the_listed_codes(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    print(f())", "    for c in [1]:", "        assert f(c) == c")})
    assert audit_mod.main([str(tmp_path)]) == 1
    capsys.readouterr()
    assert audit_mod.main([str(tmp_path), "--ignore", "print,BRANCHING"]) == 0
    assert capsys.readouterr().out.startswith("clean: no findings")


def test_repeated_ignore_accumulates(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    print(f())", "    for c in [1]:", "        assert f(c) == c")})
    assert audit_mod.main([str(tmp_path), "--ignore", "PRINT", "--ignore", "BRANCHING"]) == 0


def test_ignore_unknown_code_is_exit_2_and_names_the_valid_codes(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": BAD})
    assert audit_mod.main([str(tmp_path), "--ignore", "NO_ASERT"]) == 2
    err = capsys.readouterr().err
    assert "NO_ASERT" in err and "NO_ASSERT" in err


def test_ignore_syntax_cannot_hide_an_unparsable_file(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": "def test_x(:\n"})
    assert audit_mod.main([str(tmp_path), "--ignore", "SYNTAX"]) == 2
    assert "SYNTAX" in capsys.readouterr().err


# ----------------------------------------------------------------------------- pragmas
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(src("def test_x():", "    assert f() == 0.5  # audit: ignore[FLOAT_EQ] exact binary fraction"), [], id="same-line"),
        pytest.param(src("def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise", "    f()"), [], id="def-line-same-line-finding"),
        pytest.param(src("def test_x():  # audit: ignore[FLOAT_EQ] exact binary fraction", "    assert f() == 0.5"), [],
                     id="def-line-covers-body-line-finding"),
        pytest.param(src("import pytest", "@pytest.mark.skip('why')  # audit: ignore[FLOAT_EQ] exact", "def test_x():",
                         "    assert f() == 0.5"), [], id="decorator-line-covers-body"),
        pytest.param(src("class TestA:  # audit: ignore[FLOAT_EQ] exact", "    def test_x(self):", "        assert f() == 0.5"), [],
                     id="class-line-covers-methods"),
        pytest.param(src("def test_x():", "    assert f() == 0.5  # audit: ignore[PRINT] wrong code"),
                     [("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)], id="other-code-does-not-suppress"),
        pytest.param(src("def test_x():", "    assert f() == 0.5  # audit: ignore[FLOAT_EQ] ok", "    assert g() == 0.25"),
                     [("FLOAT_EQ", 3)], id="line-pragma-covers-only-its-line"),
        pytest.param(src("def test_x():", "    print(f())  # audit: ignore[PRINT,FLOAT_EQ] demo", "    assert f() == 0.5"),
                     [("SUPPRESSION_INERT", 2), ("FLOAT_EQ", 3)], id="several-codes-only-the-first-matches"),
        pytest.param(src("def test_x():", "    print(f() == 0.5)  # audit: ignore[PRINT,FLOAT_EQ] demo", "    assert f() == 1"), [],
                     id="several-codes"),
        pytest.param(src("# audit: ignore-file[PRINT] demo module", "def test_x():", "    print(f())", "    assert f() == 1"), [],
                     id="ignore-file"),
        pytest.param(src("# audit: ignore-file[PRINT] demo module", "def test_x():", "    assert f() == 0.5"),
                     [("SUPPRESSION_INERT", 1), ("FLOAT_EQ", 3)], id="ignore-file-only-the-listed-code"),
        pytest.param(src("def test_x():", "    s = '# audit: ignore[FLOAT_EQ] in a string'", "    assert f() == 0.5"),
                     [("FLOAT_EQ", 3)], id="pragma-text-in-a-string-is-not-a-pragma"),
    ],
)
def test_pragma_scopes(run_findings, source, expected):
    assert [(f.code, f.line) for f in run_findings(source)] == expected


def test_pragma_without_reason_is_itself_a_finding(run_findings):
    source = src("def test_x():", "    assert f() == 0.5  # audit: ignore[FLOAT_EQ]")
    findings = run_findings(source)
    assert [(f.code, f.line) for f in findings] == [("FLOAT_EQ", 2), ("SUPPRESSION_NO_REASON", 2)]
    assert findings[1].severity == "error"


def test_two_findings_on_one_line_come_out_in_a_stable_order(run_findings):
    findings = run_findings(src("def test_x():", "    assert f() == 0.25 and g() == 0.5"))
    assert len(findings) == 2
    assert "0.25" in findings[0].detail and "0.5" in findings[1].detail


# ----------------------------------------------------------------------------- output formats / CLI
def test_sarif_output_is_valid_shape(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": BAD})
    exit_code = audit_mod.main([str(tmp_path), "--format", "sarif"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    run = report["runs"][0]
    assert run["results"][0]["ruleId"] == "NO_ASSERT"
    assert run["results"][0]["locations"][0]["physicalLocation"]["region"]["startLine"] == 1
    assert {rule["id"] for rule in run["tool"]["driver"]["rules"]} == set(SEVERITY)


def test_sarif_uri_is_percent_encoded(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"my tests/test_a.py": BAD})
    audit_mod.main([str(tmp_path), "--format", "sarif"])
    uri = json.loads(capsys.readouterr().out)["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert "my%20tests" in uri and " " not in uri


def test_github_output_uses_workflow_command_syntax(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": BAD})
    audit_mod.main([str(tmp_path), "--format", "github"])
    line = capsys.readouterr().out.splitlines()[0]
    assert re.fullmatch(r"::error file=[^,]+test_a\.py,line=1,title=NO_ASSERT::.+", line)


def test_github_output_escapes_commas_and_percent_in_paths_and_messages(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"a,b/test_a.py": BAD, "c%d/test_c.py": BAD})
    audit_mod.main([str(tmp_path), "--format", "github"])
    out = capsys.readouterr().out
    assert "a%2Cb/test_a.py" in out and "c%25d/test_c.py" in out
    assert all(re.fullmatch(r"::(error|warning) file=[^,]+,line=\d+,title=[A-Z_]+::[^\n]*", ln) for ln in out.splitlines())


def test_github_warning_level_for_warning_codes(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() is not None")})
    audit_mod.main([str(tmp_path), "--format", "github"])
    assert capsys.readouterr().out.startswith("::warning file=")


def test_json_output_lists_every_field(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": BAD})
    exit_code = audit_mod.main([str(tmp_path), "--format", "json"])
    (finding,) = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert set(finding) == {"file", "line", "code", "detail", "severity"}
    assert (finding["code"], finding["line"], finding["severity"]) == ("NO_ASSERT", 1, "error")
    assert finding["file"].endswith("test_a.py")


def test_text_output_marks_warnings_and_counts(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() is not None")})
    assert audit_mod.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "test_a.py:1: WEAK_ASSERT:" in out and "[warning]" in out
    assert "1 finding(s): WEAK_ASSERT=1" in out


def test_clean_text_output_says_how_many_files_were_checked(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() == 1"), "test_b.py": src("def test_y():", "    assert f() == 2")})
    assert audit_mod.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "clean: no findings (2 file(s) checked)"


def test_list_codes_prints_every_code_with_its_severity(audit_mod, capsys):
    assert audit_mod.main(["--list-codes"]) == 0
    rows = [line.split(None, 2) for line in capsys.readouterr().out.splitlines()]
    assert {code: severity for code, severity, _ in rows} == SEVERITY
    assert len(rows) == 26


def test_warnings_do_not_fail_the_gate_by_default(tmp_path, audit_mod):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() is not None")})
    assert audit_mod.main([str(tmp_path)]) == 0
    assert audit_mod.main([str(tmp_path), "--warnings-as-errors"]) == 1


def test_single_file_path_is_reported_as_typed(tmp_path, audit_mod, capsys):
    make_project(tmp_path, {"test_a.py": BAD})
    audit_mod.main([str(tmp_path / "test_a.py")])
    assert capsys.readouterr().out.startswith(f"{tmp_path / 'test_a.py'}:1: NO_ASSERT")


def test_script_runs_as_a_program_and_exits_with_the_gate_status(tmp_path):
    import subprocess

    make_project(tmp_path, {"test_a.py": BAD})
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "audit_tests.py")
    proc = subprocess.run([sys.executable, script, str(tmp_path)], capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 1 and "NO_ASSERT" in proc.stdout


def test_every_code_is_documented(audit_mod):
    rules = (audit_mod.Path(audit_mod.__file__).parents[1] / "references" / "rules.md").read_text()
    missing = [code for code in audit_mod.CODES if code != "SYNTAX" and f"`{code}`" not in rules]
    assert missing == []


# ----------------------------------------------------------------------------- --assert-helper
def helper_run(audit_mod, tmp_path, source, *patterns, files=None):
    make_project(tmp_path, {"test_a.py": source, **(files or {})})
    report = audit_mod.audit_report(tmp_path, assert_helpers=tuple(patterns))
    return [(f.code, f.line) for f in report.findings]


HELPER_CALLS = [
    pytest.param(src("def test_x():", "    check_response(f())"), "check_*", id="bare-name-glob"),
    pytest.param(src("def test_x():", "    verify_response(f())"), "verify_response", id="exact-name"),
    pytest.param(src("def test_x():", "    self.check_x(f())"), "check_*", id="final-attribute-of-a-method-call"),
    pytest.param(src("def test_x():", "    helpers.check_x(f())"), "check_*", id="final-attribute-of-a-module-call"),
    pytest.param(src("def test_x():", "    helpers.expect_ok(f())"), "helpers.expect_*", id="dotted-pattern-matches-the-written-name"),
    pytest.param(src("import helpers as h", "def test_x():", "    h.expect_ok(f())"), "helpers.expect_*",
                 id="dotted-pattern-matches-the-import-alias-resolved-name"),
    pytest.param(src("from helpers import expect_ok as ok", "def test_x():", "    ok(f())"), "helpers.expect_*",
                 id="dotted-pattern-matches-a-from-import-alias"),
    pytest.param(src("async def test_x():", "    await check_response(f())"), "check_*", id="awaited-helper"),
    pytest.param(src("def test_x():", "    def inner():", "        check_response(f())", "    inner()"), "check_*",
                 id="helper-in-a-nested-function-that-is-called"),
    pytest.param(src("def test_x():", "    with ctx():", "        check_response(f())"), "check_*", id="helper-inside-a-with-block"),
]


@pytest.mark.parametrize(("source", "pattern"), HELPER_CALLS)
def test_assert_helper_call_counts_as_an_assertion_for_no_assert(audit_mod, tmp_path, source, pattern):
    assert [c for c, _ in helper_run(audit_mod, tmp_path / "plain", source)] == ["NO_ASSERT"], "without the option the test is reported"
    assert helper_run(audit_mod, tmp_path / "helped", source, pattern) == []


HELPER_NEAR_MISSES = [
    pytest.param(src("def test_x():", "    check_response(f())"), "verify_*", id="pattern-does-not-match-the-name"),
    pytest.param(src("def test_x():", "    precheck_response(f())"), "check_*", id="glob-is-anchored-at-the-start"),
    pytest.param(src("def test_x():", "    check_response(f())"), "Check_*", id="matching-is-case-sensitive"),
    pytest.param(src("def test_x():", "    other.expect_ok(f())"), "helpers.expect_*", id="dotted-pattern-needs-the-module-part"),
    pytest.param(src("def test_x():", "    x = check_response", "    f()"), "check_*", id="reference-without-a-call"),
    pytest.param(src("def test_x():", "    return", "    check_response(f())"), "check_*", id="call-after-an-unconditional-return"),
    pytest.param(src("def test_x():", "    raise RuntimeError", "    check_response(f())"), "check_*", id="call-after-an-unconditional-raise"),
    pytest.param(src("def test_x():", "    def inner():", "        check_response(f())", "    f()"), "check_*",
                 id="call-in-a-nested-function-nobody-calls"),
    pytest.param(src("def test_x():", "    g = lambda: check_response(f())", "    f()"), "check_*", id="call-inside-a-lambda"),
    pytest.param(src("def test_x():", "    f('check_response')"), "check_*", id="name-in-a-string"),
]


@pytest.mark.parametrize(("source", "pattern"), HELPER_NEAR_MISSES)
def test_assert_helper_does_not_count_when_the_call_cannot_assert(audit_mod, tmp_path, source, pattern):
    assert [c for c, _ in helper_run(audit_mod, tmp_path, source, pattern)] == ["NO_ASSERT"]


def test_assert_helper_leaves_every_other_rule_alone(audit_mod, tmp_path):
    source = src(
        "import time",
        "def test_x():",
        "    time.sleep(1)",
        "    print(f())",
        "    assert f() == 0.5",
        "    check_response(f())",
        "def test_y():",
        "    assert f() is not None",
        "    check_response(f())",
    )
    expected = [("SLEEP", 3), ("PRINT", 4), ("FLOAT_EQ", 5), ("WEAK_ASSERT", 7)]
    assert helper_run(audit_mod, tmp_path / "a", source) == expected
    assert helper_run(audit_mod, tmp_path / "b", source, "check_*") == expected


def test_assert_helper_is_repeatable_and_any_pattern_may_match(audit_mod, tmp_path):
    source = src("def test_x():", "    verify_invoice(f())", "def test_y():", "    check_response(f())", "def test_z():", "    f()")
    assert helper_run(audit_mod, tmp_path, source, "check_*", "verify_*") == [("NO_ASSERT", 5)]


@pytest.mark.parametrize(
    "pattern",
    ["", "   ", "*", "**", "?", "*?*", "*.*", "[a-z]", "[!x]*", "check_* ", "a b", "a,b", "check_*,verify_*", "x()"],
)
def test_useless_or_malformed_assert_helper_pattern_is_exit_2_with_a_reason(audit_mod, tmp_path, capsys, pattern):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    f()")})
    assert audit_mod.main([str(tmp_path), "--assert-helper", pattern]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--assert-helper" in captured.err and "clean" not in captured.err
    assert "NO_ASSERT" in captured.err or "literal" in captured.err or "name" in captured.err


def test_star_pattern_message_says_it_would_switch_off_no_assert(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    f()")})
    assert audit_mod.main([str(tmp_path), "--assert-helper", "*"]) == 2
    err = capsys.readouterr().err
    assert "would match every call" in err and "NO_ASSERT" in err


@pytest.mark.parametrize("pattern", ["check_*", "verify_response", "helpers.expect_*", "*_ok", "check?", "assert[A-Z]*", "*.check_*"])
def test_reasonable_assert_helper_patterns_are_accepted(audit_mod, tmp_path, pattern):
    assert helper_run(audit_mod, tmp_path, src("def test_x():", "    assert f() == 1"), pattern) == []


def test_text_summary_echoes_the_helpers_in_effect(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    check_response(f())")})
    assert audit_mod.main([str(tmp_path), "--assert-helper", "check_*", "--assert-helper", "verify_*"]) == 0
    assert capsys.readouterr().out.strip() == (
        "clean: no findings (1 file(s) checked); NO_ASSERT relaxed by assert helpers: check_*, verify_*"
    )


def test_text_summary_with_findings_also_echoes_the_helpers(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    f()")})
    assert audit_mod.main([str(tmp_path), "--assert-helper", "check_*"]) == 1
    out = capsys.readouterr().out
    assert "1 finding(s): NO_ASSERT=1" in out
    assert out.strip().splitlines()[-1].endswith("; NO_ASSERT relaxed by assert helpers: check_*")


def test_text_summary_is_unchanged_without_helpers(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() == 1")})
    assert audit_mod.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "clean: no findings (1 file(s) checked)"


def test_json_is_a_plain_list_without_helpers_and_an_object_with_metadata_with_them(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    f()")})
    audit_mod.main([str(tmp_path), "--format", "json"])
    assert [f["code"] for f in json.loads(capsys.readouterr().out)] == ["NO_ASSERT"]
    audit_mod.main([str(tmp_path), "--format", "json", "--assert-helper", "check_*"])
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"findings", "assert_helpers"}
    assert data["assert_helpers"] == ["check_*"]
    assert [(f["code"], f["line"]) for f in data["findings"]] == [("NO_ASSERT", 1)]


def test_sarif_run_properties_carry_the_helpers(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    check_response(f())")})
    audit_mod.main([str(tmp_path), "--format", "sarif", "--assert-helper", "check_*"])
    run = json.loads(capsys.readouterr().out)["runs"][0]
    assert run["properties"] == {"assertHelpers": ["check_*"]}
    assert run["results"] == []
    audit_mod.main([str(tmp_path), "--format", "sarif"])
    assert json.loads(capsys.readouterr().out)["runs"][0]["properties"] == {"assertHelpers": []}


def test_github_output_announces_the_helpers_with_a_notice(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    check_response(f())")})
    audit_mod.main([str(tmp_path), "--format", "github", "--assert-helper", "check_*"])
    assert capsys.readouterr().out.strip() == "::notice title=assert-helpers::NO_ASSERT relaxed by assert helpers: check_*"
    audit_mod.main([str(tmp_path), "--format", "github"])
    assert "::notice" not in capsys.readouterr().out


# --- project config: [tool.pytest-expert] assert_helpers
def test_pyproject_assert_helpers_apply_without_the_flag(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {
        "pyproject.toml": '[tool.pytest-expert]\nassert_helpers = ["check_*", "verify_invoice"]\n',
        "tests/test_a.py": src("def test_x():", "    check_response(f())", "def test_y():", "    verify_invoice(f())", "def test_z():", "    f()"),
    })
    assert audit_mod.main([str(tmp_path / "tests"), "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["assert_helpers"] == ["check_*", "verify_invoice"]
    assert [(f["code"], f["line"]) for f in data["findings"]] == [("NO_ASSERT", 5)]


def test_flag_adds_to_the_configured_helpers(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {
        "pyproject.toml": '[tool.pytest-expert]\nassert_helpers = ["check_*"]\n',
        "test_a.py": src("def test_x():", "    check_response(f())", "def test_y():", "    verify_invoice(f())"),
    })
    assert audit_mod.main([str(tmp_path), "--assert-helper", "verify_*", "--assert-helper", "check_*", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["assert_helpers"] == ["check_*", "verify_*"]


def test_pyproject_helpers_work_next_to_a_pytest_ini_and_without_a_tool_pytest_table(audit_mod, tmp_path):
    source = src("def test_x():", "    check_response(f())")
    with_ini = make_project(tmp_path / "a", {
        "pytest.ini": "[pytest]\n", "pyproject.toml": '[tool.pytest-expert]\nassert_helpers = ["check_*"]\n', "test_a.py": source})
    assert audit_mod.audit_report(with_ini).findings == []
    alone = make_project(tmp_path / "b", {"pyproject.toml": '[tool.pytest-expert]\nassert_helpers = ["check_*"]\n', "t/test_a.py": source})
    assert audit_mod.audit_report(alone / "t").findings == []


def test_pyproject_without_the_table_leaves_no_assert_strict(audit_mod, tmp_path):
    source = src("def test_x():", "    check_response(f())")
    make_project(tmp_path, {"pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-q'\n", "test_a.py": source})
    assert [(f.code, f.line) for f in audit_mod.audit_report(tmp_path).findings] == [("NO_ASSERT", 1)]


@pytest.mark.parametrize(
    ("toml", "words"),
    [
        pytest.param('[tool.pytest-expert]\nassert_helpers = "check_*"\n', "list of strings", id="a-string-instead-of-a-list"),
        pytest.param('[tool.pytest-expert]\nassert_helpers = [1]\n', "list of strings", id="a-non-string-entry"),
        pytest.param('[tool.pytest-expert]\nassert_helpers = ["*"]\n', "would match every call", id="match-everything"),
        pytest.param('[tool.pytest-expert]\nassert_helpers = [""]\n', "empty", id="empty-entry"),
    ],
)
def test_bad_configured_helpers_are_exit_2_naming_the_file(audit_mod, tmp_path, capsys, toml, words):
    make_project(tmp_path, {"pyproject.toml": toml, "test_a.py": src("def test_x():", "    f()")})
    assert audit_mod.main([str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "pyproject.toml" in err and "assert_helpers" in err and words in err


def test_without_tomllib_a_pyproject_with_assert_helpers_warns_loudly(audit_mod, tmp_path, monkeypatch, capsys):
    make_project(tmp_path, {
        "pyproject.toml": '[tool.pytest-expert]\nassert_helpers = ["check_*"]\n',
        "test_a.py": src("def test_x():", "    check_response(f())"),
    })
    monkeypatch.setattr(audit_mod, "tomllib", None)
    assert audit_mod.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "audit_tests: warning:" in err and "assert_helpers" in err and "no tomllib" in err


def test_without_tomllib_a_pyproject_without_the_table_gives_no_helper_warning(audit_mod, tmp_path, monkeypatch, capsys):
    make_project(tmp_path, {"pyproject.toml": "[project]\nname = 'x'\n", "test_a.py": src("def test_x():", "    assert f() == 1")})
    monkeypatch.setattr(audit_mod, "tomllib", None)
    assert audit_mod.main([str(tmp_path)]) == 0
    assert "assert_helpers" not in capsys.readouterr().err


# ----------------------------------------------------------------------------- pragmas that suppress nothing
INERT_CASES = [
    case("unknown-code-with-a-near-match",
         src("def test_x():  # audit: ignore[NO_ASERT] smoke test", "    f()"), ("NO_ASSERT", 1), ("SUPPRESSION_INERT", 1)),
    case("unknown-code-with-no-near-match",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[QQQ_ZZZ] why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("lowercase-code",
         src("def test_x():  # audit: ignore[no_assert] smoke test", "    f()"), ("NO_ASSERT", 1), ("SUPPRESSION_INERT", 1)),
    case("mixed-case-code",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[Float_Eq] why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("empty-brackets",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[] why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("blank-brackets",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[ , ] why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("ALL-is-not-a-code",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[ALL] why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("lowercase-all-is-not-a-code",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[all] why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("no-brackets",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore FLOAT_EQ why"), ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2)),
    case("ignore-file-with-an-unknown-code",
         src("# audit: ignore-file[PRNT] demo module", "def test_x():", "    print(f())", "    assert f() == 1"),
         ("SUPPRESSION_INERT", 1), ("PRINT", 3)),
    case("code-that-pragmas-cannot-suppress",
         src("def test_x():", "    assert f() == 1  # audit: ignore[SUPPRESSION_NO_REASON] why"), ("SUPPRESSION_INERT", 2)),
    case("stale-line-pragma",
         src("def test_x():", "    assert f() == 1  # audit: ignore[FLOAT_EQ] was a float once"), ("SUPPRESSION_INERT", 2)),
    case("stale-pragma-for-a-code-on-another-line",
         src("def test_x():", "    assert f() == 0.5", "    assert g() == 1  # audit: ignore[FLOAT_EQ] why"),
         ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 3)),
    case("stale-def-line-pragma",
         src("def test_x():  # audit: ignore[FLOAT_EQ] was a float once", "    assert f() == 1"), ("SUPPRESSION_INERT", 1)),
    case("stale-class-line-pragma",
         src("class TestA:  # audit: ignore[PRINT] why", "    def test_x(self):", "        assert f() == 1"), ("SUPPRESSION_INERT", 1)),
    case("def-pragma-does-not-reach-the-next-function",
         src("def test_x():  # audit: ignore[FLOAT_EQ] why", "    assert f() == 0.5", "def test_y():", "    assert f() == 0.5"),
         ("FLOAT_EQ", 4)),
    case("stale-ignore-file",
         src("# audit: ignore-file[PRINT] demo module", "def test_x():", "    assert f() == 1"), ("SUPPRESSION_INERT", 1)),
    case("only-the-unused-code-of-a-pragma-is-reported",
         src("def test_x():", "    print(f() == 0.5)  # audit: ignore[PRINT,FLOAT_EQ,SLEEP] why", "    assert f() == 1"),
         ("SUPPRESSION_INERT", 2)),
    case("reasonless-pragma-with-a-bad-code-gets-both",
         src("def test_x():", "    assert f() == 0.5  # audit: ignore[FLOAT_EK]"),
         ("FLOAT_EQ", 2), ("SUPPRESSION_INERT", 2), ("SUPPRESSION_NO_REASON", 2)),
]


@pytest.mark.parametrize(("source", "expected"), INERT_CASES)
def test_pragma_that_suppresses_nothing_is_reported_at_its_line(run_findings, source, expected):
    findings = run_findings(source)
    assert sorted((f.code, f.line) for f in findings) == sorted(expected)
    assert all(f.severity == SEVERITY[f.code] for f in findings)


INERT_CLEAN = [
    pytest.param(src("def test_x():", "    assert f() == 0.5  # audit: ignore[FLOAT_EQ] exact binary fraction"), id="used-line-pragma"),
    pytest.param(src("def test_x():  # audit: ignore[NO_ASSERT] smoke test", "    f()"), id="used-def-line-pragma-same-line"),
    pytest.param(src("def test_x():  # audit: ignore[FLOAT_EQ] exact", "    f()", "    g()", "    assert h() == 0.5"),
                 id="def-line-pragma-used-by-a-finding-deep-in-its-scope"),
    pytest.param(src("import pytest", "@pytest.mark.skip", "def test_x():  # audit: ignore[SKIP_NO_REASON] why", "    assert f() == 1"),
                 id="def-line-pragma-used-by-a-finding-on-the-decorator-line"),
    pytest.param(src("class TestA:  # audit: ignore[FLOAT_EQ] exact", "    def test_x(self):", "        assert f() == 0.5"),
                 id="class-line-pragma-used-by-a-method"),
    pytest.param(src("# audit: ignore-file[PRINT] demo module", "def test_x():", "    print(f())", "    assert f() == 1"), id="used-ignore-file"),
    pytest.param(src("def test_x():", "    print(f() == 0.5)  # audit: ignore[PRINT,FLOAT_EQ] demo", "    assert f() == 1"), id="every-code-of-a-pragma-used"),
    pytest.param(src("def test_x():", "    s = '# audit: ignore[NO_ASERT] in a string'", "    assert f() == 1"), id="bad-pragma-text-in-a-string"),
    pytest.param(src("def test_x():", "    \"\"\"# audit: ignore[ALL] in a docstring\"\"\"", "    assert f() == 1"), id="bad-pragma-text-in-a-docstring"),
    pytest.param(src("def test_x():", "    assert f() == 1  # audit: ignored files are listed in docs"), id="prose-that-only-starts-with-ignore"),
    pytest.param(src("def test_x():", "    assert f() == 1  # audit results are in docs"), id="comment-that-only-mentions-audit"),
    pytest.param(src("def test_x():", "    assert f() == 1  # noqa: S101"), id="other-tools-pragmas"),
]


@pytest.mark.parametrize("source", INERT_CLEAN)
def test_pragmas_that_do_their_job_or_are_not_pragmas_stay_quiet(run_findings, source):
    assert [(f.code, f.line) for f in run_findings(source)] == []


def test_inert_pragma_message_says_what_is_wrong_and_suggests_the_nearest_code(run_findings):
    typo = run_findings(src("def test_x():  # audit: ignore[NO_ASERT] smoke test", "    f()"))
    (inert,) = [f for f in typo if f.code == "SUPPRESSION_INERT"]
    assert "NO_ASERT" in inert.detail and "unknown code" in inert.detail and "did you mean NO_ASSERT" in inert.detail
    far = run_findings(src("def test_x():", "    assert f() == 0.5  # audit: ignore[QQQ_ZZZ] why"))
    (inert,) = [f for f in far if f.code == "SUPPRESSION_INERT"]
    assert "unknown code QQQ_ZZZ" in inert.detail and "did you mean" not in inert.detail
    lower = run_findings(src("def test_x():  # audit: ignore[no_assert] smoke test", "    f()"))
    (inert,) = [f for f in lower if f.code == "SUPPRESSION_INERT"]
    assert "upper case" in inert.detail and "NO_ASSERT" in inert.detail
    empty = run_findings(src("def test_x():", "    assert f() == 0.5  # audit: ignore[] why"))
    (inert,) = [f for f in empty if f.code == "SUPPRESSION_INERT"]
    assert "empty" in inert.detail
    every = run_findings(src("def test_x():", "    assert f() == 0.5  # audit: ignore[ALL] why"))
    (inert,) = [f for f in every if f.code == "SUPPRESSION_INERT"]
    assert "ALL" in inert.detail and "name each code" in inert.detail
    stale = run_findings(src("def test_x():", "    print(f() == 0.5)  # audit: ignore[PRINT,SLEEP] why", "    assert f() == 1"))
    (inert,) = [f for f in stale if f.code == "SUPPRESSION_INERT"]
    assert "SLEEP" in inert.detail and "PRINT" not in inert.detail and "no SLEEP finding" in inert.detail


def test_a_pragma_cannot_suppress_the_inert_pragma_finding(run_findings):
    source = src("def test_x():", "    assert f() == 1  # audit: ignore[SUPPRESSION_INERT,FLOAT_EQ] why")
    assert [(f.code, f.line) for f in run_findings(source)] == [("SUPPRESSION_INERT", 2)]


def test_inert_pragma_is_a_warning_so_it_does_not_fail_the_gate_unless_asked(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() == 1  # audit: ignore[FLOAT_EQ] stale")})
    assert audit_mod.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "test_a.py:2: SUPPRESSION_INERT:" in out and "[warning]" in out and "SUPPRESSION_INERT=1" in out
    assert audit_mod.main([str(tmp_path), "--warnings-as-errors"]) == 1


def test_ignore_on_the_command_line_removes_the_inert_pragma_finding(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() == 1  # audit: ignore[FLOAT_EQ] stale")})
    assert audit_mod.main([str(tmp_path), "--ignore", "SUPPRESSION_INERT", "--warnings-as-errors"]) == 0
    assert capsys.readouterr().out.strip() == "clean: no findings (1 file(s) checked)"


def test_a_command_line_ignore_does_not_make_a_pragma_stale(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() == 0.5  # audit: ignore[FLOAT_EQ] exact")})
    assert audit_mod.main([str(tmp_path), "--ignore", "FLOAT_EQ", "--warnings-as-errors"]) == 0
    assert capsys.readouterr().out.startswith("clean: no findings")


def test_assert_helper_makes_a_no_assert_pragma_stale(audit_mod, tmp_path):
    source = src("def test_x():  # audit: ignore[NO_ASSERT] checks through a helper", "    check_response(f())")
    make_project(tmp_path, {"test_a.py": source})
    assert [f.code for f in audit_mod.audit_report(tmp_path).findings] == []
    assert [(f.code, f.line) for f in audit_mod.audit_report(tmp_path, assert_helpers=("check_*",)).findings] == [("SUPPRESSION_INERT", 1)]


def test_inert_pragma_in_json_and_sarif_has_the_warning_severity(audit_mod, tmp_path, capsys):
    make_project(tmp_path, {"test_a.py": src("def test_x():", "    assert f() == 1  # audit: ignore[FLOAT_EQ] stale")})
    audit_mod.main([str(tmp_path), "--format", "json"])
    (finding,) = json.loads(capsys.readouterr().out)
    assert (finding["code"], finding["line"], finding["severity"]) == ("SUPPRESSION_INERT", 2, "warning")
    audit_mod.main([str(tmp_path), "--format", "sarif"])
    result = json.loads(capsys.readouterr().out)["runs"][0]["results"][0]
    assert (result["ruleId"], result["level"]) == ("SUPPRESSION_INERT", "warning")


def test_list_codes_text_for_the_new_code(audit_mod, capsys):
    audit_mod.main(["--list-codes"])
    (row,) = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("SUPPRESSION_INERT")]
    assert row.split(None, 2)[1] == "warning"
    assert "suppresses nothing" in row
