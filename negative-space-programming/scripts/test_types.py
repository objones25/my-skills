#!/usr/bin/env python3
"""Type-checker regression test. Runs under pytest or bare python:

    python scripts/test_types.py

Needs `pyright` on PATH (Pylance reports the same rule IDs). Without it the
tests skip loudly rather than pass silently.

1. The skill's own code is clean under pyright strict (pyproject.toml).
2. `typing_before.py` still reproduces every rule ID it documents, so the
   fixture keeps testing what it claims to.
3. `typing_after.py` (the fix for each of those) is clean.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "assets" / "examples"
EXPECTED_IN_BEFORE = frozenset({
    "reportTypedDictNotRequiredAccess",
    "reportOptionalSubscript",
    "reportOptionalMemberAccess",
    "reportGeneralTypeIssues",
    "reportArgumentType",
    "reportPossiblyUnboundVariable",  # config key: reportPossiblyUnbound
    "reportUnnecessaryIsInstance",
    "reportImplicitOverride",
})
PYRIGHT_TIMEOUT_S = 300
# The skill's pyproject excludes typing_before.py so a plain `pyright` run is
# clean, and pyright skips excluded files even when named on the command line.
# So the fixture is checked in a scratch directory with this config instead.
FIXTURE_CONFIG = {
    "typeCheckingMode": "strict",
    "pythonVersion": "3.10",
    "reportImplicitOverride": "error",
}


class SkipTestError(Exception):
    """Raised under bare python when a prerequisite is missing."""


def _skip(reason: str) -> None:
    try:
        pytest = importlib.import_module("pytest")
    except ImportError:
        raise SkipTestError(reason) from None
    pytest.skip(reason)


def _pyright(*paths: Path, cwd: Path = ROOT) -> list[dict[str, object]]:
    exe = shutil.which("pyright")
    if exe is None:
        _skip("pyright not on PATH (npm install -g pyright, or pip install pyright)")
        return []
    proc = subprocess.run(  # noqa: S603  fixed argv, no shell
        [exe, "--outputjson", *map(str, paths)],
        cwd=cwd, capture_output=True, text=True, check=False,
        timeout=PYRIGHT_TIMEOUT_S,
    )
    # pyright exits 1 when it reports errors; anything else is a crash.
    assert proc.returncode in {0, 1}, f"pyright crashed ({proc.returncode}): {proc.stderr}"
    report: dict[str, object] = json.loads(proc.stdout)
    diagnostics = report["generalDiagnostics"]
    assert isinstance(diagnostics, list), "pyright JSON changed shape"
    items = cast("list[object]", diagnostics)  # verified by the isinstance above
    return [cast("dict[str, object]", d) for d in items if isinstance(d, dict)]


def _render(diags: list[dict[str, object]]) -> str:
    return "\n".join(f"  {d.get('file')}:{d.get('rule')}: {d.get('message')}" for d in diags)


def test_skill_code_is_strict_clean() -> None:
    diags = _pyright()
    assert diags == [], f"pyright strict found problems:\n{_render(diags)}"


def test_before_fixture_reproduces_each_rule() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for src in (EXAMPLES / "typing_before.py", ROOT / "assets" / "negative_space.py"):
            (work / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        (work / "pyrightconfig.json").write_text(json.dumps(FIXTURE_CONFIG),
                                                 encoding="utf-8")
        diags = _pyright(work / "typing_before.py", cwd=work)
    assert diags, "pyright reported nothing; was the fixture analyzed at all?"
    fired = {d.get("rule") for d in diags}
    missing = EXPECTED_IN_BEFORE - fired
    assert not missing, f"typing_before.py no longer triggers: {sorted(missing)}"


def test_after_fixture_is_clean() -> None:
    diags = _pyright(EXAMPLES / "typing_after.py")
    assert diags == [], f"typing_after.py has problems:\n{_render(diags)}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except SkipTestError as exc:
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
