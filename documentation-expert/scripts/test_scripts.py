#!/usr/bin/env python3
"""Regression tests for the three bundled scripts. Stdlib unittest only.

Run from anywhere:
    python3 -m unittest discover -s scripts -p "test_*.py"
    python3 scripts/test_scripts.py

Each test builds a throwaway repository in a temporary directory, runs a script
as a subprocess, and checks its output or exit code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFY = HERE / "verify_claims.py"
LINT = HERE / "lint_prose.py"
SCAN = HERE / "scan_repo.py"


def run(script: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-I", str(script), *args],
                          capture_output=True, text=True, cwd=cwd, env=env, check=False)


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


def make_repo(root: Path) -> None:
    write(root, "pyproject.toml", """
        [project]
        name = "widgetkit"
        version = "0.1.0"

        [project.scripts]
        widgetkit = "widgetkit.cli:main"
    """)
    write(root, "src/widgetkit/__init__.py", """
        from widgetkit.core import build_widget
        __all__ = ["build_widget"]
    """)
    write(root, "src/widgetkit/core.py", """
        import os
        def build_widget(name):
            return os.environ.get("WIDGETKIT_TIMEOUT", name)
    """)
    write(root, "src/widgetkit/cli.py", """
        import argparse
        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--dry-run", action="store_true")
            ap.add_argument("--size", type=int)
            return ap.parse_args()
    """)
    write(root, "docs/guide.md", """
        # Guide

        ## my_var option

        Text.

        ## Install & configure

        Text.
    """)


def ledger(root: Path, doc: Path, *extra: str) -> list[dict]:
    res = run(VERIFY, str(doc), "--repo", str(root), "--json", *extra)
    return json.loads(res.stdout)[0]["claims"]


def verdict(claims: list[dict], value: str) -> str:
    hits = [c["verdict"] for c in claims if c["value"] == value]
    return hits[0] if hits else "(no claim)"


class VerifyClaims(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_repo(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_github_slugs_keep_double_hyphens_and_underscores(self):
        doc = write(self.root, "README.md", """
            # W

            ## Install & configure

            See [install](#install--configure),
            [var](docs/guide.md#my_var-option),
            [both](docs/guide.md#install--configure).
        """)
        claims = ledger(self.root, doc)
        anchors = {c["value"]: c["verdict"] for c in claims if c["kind"] == "anchor"}
        self.assertEqual(anchors.get("#install--configure"), "verified", anchors)
        self.assertEqual(anchors.get("docs/guide.md#my_var-option"), "verified", anchors)
        self.assertEqual(anchors.get("docs/guide.md#install--configure"), "verified", anchors)

    def test_collapsed_slug_is_reported_missing(self):
        doc = write(self.root, "README.md", """
            # W

            ## Install & configure

            See [install](#install-configure).
        """)
        self.assertEqual(verdict(ledger(self.root, doc), "#install-configure"), "missing")

    def test_flag_prefix_does_not_verify(self):
        doc = write(self.root, "README.md", """
            # W

            ```bash
            widgetkit --dry
            widgetkit --dry-run
            widgetkit --siz 3
            ```
        """)
        claims = ledger(self.root, doc)
        self.assertEqual(verdict(claims, "--dry"), "missing")
        self.assertEqual(verdict(claims, "--dry-run"), "verified")
        self.assertEqual(verdict(claims, "--siz"), "missing")

    def test_env_var_prefix_does_not_verify(self):
        doc = write(self.root, "README.md", """
            # W

            Set `WIDGETKIT_TIME` or `WIDGETKIT_TIMEOUT`.
        """)
        claims = ledger(self.root, doc)
        self.assertEqual(verdict(claims, "WIDGETKIT_TIME"), "missing")
        self.assertEqual(verdict(claims, "WIDGETKIT_TIMEOUT"), "verified")

    def test_runner_prefix_still_checks_project_flags(self):
        doc = write(self.root, "README.md", """
            # W

            ```bash
            uv run widgetkit --bogus
            ```
        """)
        self.assertEqual(verdict(ledger(self.root, doc), "--bogus"), "missing")

    def test_python_m_package_needs_dunder_main(self):
        doc = write(self.root, "README.md", """
            # W

            ```bash
            python -m widgetkit
            python -m widgetkit.cli
            python -m venv .venv
            python -m pip install -e .
            ```
        """)
        claims = ledger(self.root, doc)
        by_value = {c["value"]: c["verdict"] for c in claims if c["kind"] == "command"}
        self.assertEqual(by_value.get("python -m widgetkit"), "missing", by_value)
        self.assertEqual(by_value.get("python -m widgetkit.cli"), "verified", by_value)
        self.assertEqual(by_value.get("python -m venv"), "skipped", by_value)
        self.assertEqual(by_value.get("python -m pip"), "skipped", by_value)
        write(self.root, "src/widgetkit/__main__.py", "from widgetkit.cli import main\nmain()\n")
        by_value = {c["value"]: c["verdict"] for c in ledger(self.root, doc) if c["kind"] == "command"}
        self.assertEqual(by_value.get("python -m widgetkit"), "verified", by_value)

    def test_shell_control_flow_and_mime_types_are_not_claims(self):
        doc = write(self.root, "README.md", """
            # W

            Send `application/json` and use `POST`. Set `HTTPS_PROXY` if needed.

            ```bash
            if widgetkit --dry-run; then
              echo ok
            fi
            ```
        """)
        claims = ledger(self.root, doc)
        noisy = [c for c in claims if c["severity"] in ("missing", "warn")]
        self.assertEqual(noisy, [])

    def test_fail_on_is_a_threshold(self):
        doc = write(self.root, "README.md", """
            # W

            See [gone](docs/gone.md).
        """)
        self.assertEqual(run(VERIFY, str(doc), "--repo", str(self.root), "--fail-on", "warn").returncode, 1)
        self.assertEqual(run(VERIFY, str(doc), "--repo", str(self.root)).returncode, 1)

    def test_fail_on_rejects_unknown_severity(self):
        doc = write(self.root, "README.md", "# W\n\nSee [gone](docs/gone.md).\n")
        self.assertEqual(run(VERIFY, str(doc), "--repo", str(self.root), "--fail-on", "mising").returncode, 2)

    def test_rst_is_skipped_not_misread(self):
        doc = write(self.root, "README.rst", """
            Title
            =====

            Usage
            ~~~~~

            Wrap ``tqdm()`` around any iterable.
        """)
        res = run(VERIFY, str(doc), "--repo", str(self.root))
        self.assertEqual(res.returncode, 2)
        self.assertIn("Markdown only", res.stderr)

    def test_indented_snippet_and_bare_decorator_parse(self):
        doc = write(self.root, "README.md", """
            # W

            ```python
                def f():
                    return 1
            ```

            ```python
            @click.option("--upper", flag_value="upper")
            ```
        """)
        claims = ledger(self.root, doc)
        self.assertEqual([c for c in claims if c["verdict"] == "syntax-error"], [])

    def test_real_syntax_error_still_fails(self):
        doc = write(self.root, "README.md", "# W\n\n```python\ndef broken(:\n    pass\n```\n")
        res = run(VERIFY, str(doc), "--repo", str(self.root))
        self.assertEqual(res.returncode, 1)
        self.assertIn("syntax", res.stdout)

    def test_setup_cfg_extras_are_not_entrypoints(self):
        (self.root / "pyproject.toml").unlink()
        write(self.root, "setup.cfg", """
            [options.entry_points]
            console_scripts =
                widgetkit = widgetkit.cli:main

            [options.extras_require]
            dev =
                pytest
        """)
        doc = write(self.root, "README.md", "# W\n\n```bash\ndev --flag\nwidgetkit --size 2\n```\n")
        claims = ledger(self.root, doc)
        commands = {c["value"]: c["verdict"] for c in claims if c["kind"] == "command"}
        self.assertNotEqual(commands.get("dev --flag"), "verified", commands)
        self.assertEqual(commands.get("widgetkit --size 2"), "verified", commands)

    def test_poetry_scripts_are_entrypoints(self):
        write(self.root, "pyproject.toml", """
            [tool.poetry]
            name = "widgetkit"

            [tool.poetry.scripts]
            wk = "widgetkit.cli:main"
        """)
        doc = write(self.root, "README.md", "# W\n\n```bash\nwk --size 2\nwk --nope\n```\n")
        claims = ledger(self.root, doc)
        self.assertEqual(verdict(claims, "--size"), "verified")
        self.assertEqual(verdict(claims, "--nope"), "missing")


class LintProse(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_list_rules_needs_no_document(self):
        res = run(LINT, "--list-rules")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("A11Y003", res.stdout)

    def test_long_sentence_across_wrapped_lines(self):
        doc = write(self.root, "notes.md", """
            # Notes

            This sentence is deliberately made extremely long so that it definitely
            exceeds the twenty six word limit that the accessibility rule enforces for
            readers here today and then it keeps going.
        """)
        res = run(LINT, str(doc), "--json")
        rules = [(f["rule"], f["line"]) for f in json.loads(res.stdout)[0]["findings"]]
        self.assertIn(("A11Y003", 3), rules)

    def test_twenty_six_words_is_already_too_long(self):
        words = " ".join(["word"] * 26) + "."
        doc = write(self.root, "notes.md", f"# Notes\n\n{words}\n")
        res = run(LINT, str(doc), "--json")
        self.assertIn("A11Y003", [f["rule"] for f in json.loads(res.stdout)[0]["findings"]])

    def test_unknown_disable_rule_is_an_error(self):
        doc = write(self.root, "notes.md", "# Notes\n\nText.\n")
        self.assertEqual(run(LINT, str(doc), "--disable", "STYLE002").returncode, 2)

    def test_clean_readme_passes(self):
        doc = write(self.root, "README.md", """
            # Widget toolkit

            Build widgets from Python with typed sizes.

            ## Install

            Install the package with pip:

            ```bash
            pip install widgetkit
            ```

            ## Usage

            The following example builds one widget:

            ```python
            from widgetkit import build_widget
            ```

            ## Contributing

            Open an issue before sending a pull request.

            ## License

            MIT
        """)
        res = run(LINT, str(doc))
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertIn("Totals: clean", res.stdout)


class NoWrites(unittest.TestCase):
    def test_scripts_leave_no_bytecode_cache(self):
        cache = HERE / "__pycache__"
        before = set(cache.iterdir()) if cache.exists() else set()
        with tempfile.TemporaryDirectory() as tmp:
            doc = write(Path(tmp), "README.md", "# W\n\nText.\n")
            env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
            for script in (LINT, SCAN):
                args = [str(doc)] if script is LINT else [tmp]
                subprocess.run([sys.executable, str(script), *args], env=env,
                               capture_output=True, check=False)
        after = set(cache.iterdir()) if cache.exists() else set()
        new_files = {p.name for p in after - before if p.name.startswith("verify_claims")}
        self.assertEqual(new_files, set())


class ScanRepo(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.root), "-c", "user.email=t@t", "-c", "user.name=t",
                        *args], check=True, capture_output=True)

    def test_drift_counts_flat_layout_and_excludes_doc_commit(self):
        write(self.root, "setup.py", "from setuptools import setup\nsetup(name='flat')\n")
        write(self.root, "flatpkg/__init__.py", "X = 1\n")
        write(self.root, "README.md", "# Flat\n\nUses `X`.\n")
        try:
            self.git("init", "-q")
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git not available")
        self.git("add", "-A")
        self.git("commit", "-qm", "both at once")
        data = json.loads(run(SCAN, str(self.root), "--json").stdout)
        readme = next(d for d in data["docs"] if d["path"] == "README.md")
        self.assertEqual(readme.get("source_commits_since_doc_touched"), 0, readme)
        write(self.root, "flatpkg/__init__.py", "X = 2\n")
        self.git("commit", "-qam", "code only")
        data = json.loads(run(SCAN, str(self.root), "--json").stdout)
        readme = next(d for d in data["docs"] if d["path"] == "README.md")
        self.assertEqual(readme.get("source_commits_since_doc_touched"), 1, readme)


if __name__ == "__main__":
    unittest.main()
