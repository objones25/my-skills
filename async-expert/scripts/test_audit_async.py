"""Regression tests for audit_async.py. Stdlib only.

    python -m unittest scripts/test_audit_async.py        (from the skill root)
    AUDIT_SCRIPT=path/to/audit_async.py python -m unittest ...   (test another copy)

Each case is a snippet plus the exact set of codes the audit must report for it.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = Path(os.environ.get("AUDIT_SCRIPT", HERE / "audit_async.py")).resolve()
BAD_EXAMPLE = HERE.parent / "assets" / "examples" / "bad_example.py"

_spec = importlib.util.spec_from_file_location("audit_async_under_test", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit  # @dataclass looks its module up in sys.modules
_spec.loader.exec_module(audit)


def codes(src: str) -> list[str]:
    return sorted(f.code for f in audit.audit_source("t.py", textwrap.dedent(src)))


class FalseNegatives(unittest.TestCase):
    def test_from_imported_gather_is_checked(self):
        src = """
            import asyncio
            from asyncio import gather
            async def fetch(u): ...
            async def main(urls):
                await gather(*[fetch(u) for u in urls])
        """
        self.assertEqual(codes(src), ["BARE_GATHER", "UNBOUNDED_FANOUT"])

    def test_except_star_cancelled_swallow(self):
        src = """
            import asyncio
            async def main():
                try:
                    await asyncio.sleep(1)
                except* asyncio.CancelledError:
                    pass
        """
        self.assertEqual(codes(src), ["SWALLOWED_CANCEL"])

    def test_ensure_future_discarded(self):
        src = """
            import asyncio
            async def work(): ...
            async def main():
                asyncio.ensure_future(work())
        """
        self.assertEqual(codes(src), ["FLOATING_TASK"])

    def test_yield_in_cm_in_async_generator(self):
        src = """
            import asyncio
            async def rows():
                async with asyncio.timeout(1):
                    yield 1
        """
        self.assertEqual(codes(src), ["ASYNC_GEN_YIELD_IN_CM"])


class FalsePositives(unittest.TestCase):
    def test_method_with_same_name_as_coroutine(self):
        src = """
            async def update(x):
                return x
            async def main(d: dict):
                d.update({"a": 1})
        """
        self.assertEqual(codes(src), [])

    def test_awaited_async_path_read(self):
        src = """
            import anyio
            async def main(p: anyio.Path):
                return await p.read_text()
        """
        self.assertEqual(codes(src), [])

    def test_open_inside_lambda_handed_to_thread(self):
        src = """
            import asyncio
            async def main(path):
                return await asyncio.to_thread(lambda: open(path).read())
        """
        self.assertEqual(codes(src), [])

    def test_timeout_via_double_star_kwargs(self):
        src = """
            import httpx
            async def main():
                kw = {"timeout": 5.0}
                async with httpx.AsyncClient(**kw) as c:
                    await c.get("http://x")
        """
        self.assertEqual(codes(src), [])

    def test_asynccontextmanager_yield_is_fine(self):
        src = """
            import asyncio, contextlib
            @contextlib.asynccontextmanager
            async def session():
                async with asyncio.timeout(1):
                    yield 1
        """
        self.assertEqual(codes(src), [])

    def test_still_flags_real_blocking_and_unawaited(self):
        src = """
            async def fetch(u): ...
            async def main(p):
                p.read_text()
                fetch("x")
        """
        self.assertEqual(codes(src), ["BLOCKING_OPEN", "UNAWAITED_COROUTINE"])


class CommandLine(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True)

    def test_missing_path_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(str(Path(tmp) / "no_such_dir"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_bad_example_counts(self):
        result = self.run_cli(str(BAD_EXAMPLE))
        self.assertEqual(result.returncode, 1)
        self.assertIn("19 finding(s)", result.stdout)
        self.assertIn("ASYNC_GEN_YIELD_IN_CM=1", result.stdout)

    def test_clean_assets_pass(self):
        assets = HERE.parent / "assets"
        result = self.run_cli(str(assets / "aioutil"), str(assets / "examples" / "good_pipeline.py"),
                              str(assets / "examples" / "fastapi_service.py"))
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
