#!/usr/bin/env python3
"""Regression tests for the three bundled scripts. Stdlib unittest only.

Run from anywhere. Paths inside the tests resolve from this file, so the
commands work from any directory when given the skill's absolute path:
    python3 -m unittest discover -s /path/to/documentation-expert/scripts -p "test_*.py"
    python3 /path/to/documentation-expert/scripts/test_scripts.py

Each test builds a throwaway repository in a temporary directory, runs a script
as a subprocess, and checks its output or exit code.
"""

from __future__ import annotations

import json
import os
import re
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

    def test_slug_keeps_leading_and_trailing_underscores(self):
        doc = write(self.root, "README.md", """
            # W

            ## _private

            ## trailing_

            See [a](#_private), [b](#trailing_), [c](#private).
        """)
        claims = ledger(self.root, doc)
        anchors = {c["value"]: c["verdict"] for c in claims if c["kind"] == "anchor"}
        self.assertEqual(anchors.get("#_private"), "verified", anchors)
        self.assertEqual(anchors.get("#trailing_"), "verified", anchors)
        self.assertEqual(anchors.get("#private"), "missing", anchors)

    def test_link_schemes_are_case_insensitive(self):
        doc = write(self.root, "README.md", """
            # W

            See [a](HTTPS://example.com), [b](ftp://host/file), [c](//cdn.example/x.js),
            [d](MAILTO:a@b.c).
        """)
        claims = {c["value"]: c["verdict"] for c in ledger(self.root, doc) if c["kind"] == "link"}
        self.assertEqual(claims.get("HTTPS://example.com"), "external", claims)
        self.assertEqual(claims.get("ftp://host/file"), "skipped", claims)
        self.assertEqual(claims.get("//cdn.example/x.js"), "external", claims)
        self.assertEqual(claims.get("MAILTO:a@b.c"), "skipped", claims)

    def test_comment_line_does_not_make_a_shell_block_a_transcript(self):
        doc = write(self.root, "README.md", """
            # W

            ```bash
            # build it
            widgetkit --dry-run
            nosuchtool --run
            ```
        """)
        claims = ledger(self.root, doc)
        self.assertEqual(verdict(claims, "nosuchtool --run"), "unresolved", claims)

    def test_make_options_and_variables_are_not_targets(self):
        write(self.root, "Makefile", "test:\n\techo hi\n")
        doc = write(self.root, "README.md", """
            # W

            ```bash
            make VAR=1 test
            make -C . test
            make -f Makefile test
            make -j 4 test
            make nope
            ```
        """)
        cmds = {c["value"]: c["verdict"] for c in ledger(self.root, doc) if c["kind"] == "command"}
        for key in ("make VAR=1 test", "make -C .", "make -f Makefile", "make -j 4"):
            self.assertEqual(cmds.get(key), "verified", cmds)
        self.assertEqual(cmds.get("make nope"), "missing", cmds)

    def test_setup_cfg_extras_requirement_is_not_an_entrypoint(self):
        # The extras entry carries "=" so only the end-of-section guard keeps it
        # out of the entrypoints; an extras line without "=" would pass either way.
        (self.root / "pyproject.toml").unlink()
        write(self.root, "setup.cfg", """
            [options.entry_points]
            console_scripts =
                widgetkit = widgetkit.cli:main

            [options.extras_require]
            dev =
                pytest==8.0
        """)
        doc = write(self.root, "README.md", "# W\n\n```bash\npytest --size 2\n```\n")
        cmds = {c["value"]: c["verdict"] for c in ledger(self.root, doc) if c["kind"] == "command"}
        self.assertIn("pytest --size 2", cmds)
        self.assertNotEqual(cmds["pytest --size 2"], "verified", cmds)


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

    def lint_rules(self, text: str, *extra: str) -> list[str]:
        doc = write(self.root, "README.md", text)
        res = run(LINT, str(doc), "--json", *extra)
        return [f["rule"] for f in json.loads(res.stdout)[0]["findings"]]

    def test_h1_in_code_fence_or_span_is_not_a_second_title(self):
        rules = self.lint_rules("# T\n\n```html\n<h1>Demo</h1>\n```\n\nUse `<h1>` for titles.\n")
        self.assertNotIn("DOC001", rules)

    def test_raw_html_h1_still_counts(self):
        rules = self.lint_rules("# T\n\n<h1>Second</h1>\n")
        self.assertIn("DOC001", rules)

    def test_linked_badges_are_counted_once(self):
        def badges(n: int) -> str:
            one = "[![b](https://img.shields.io/x.svg)](https://ci.example/x) "
            return "# T\n\n" + one * n + "\n\nText.\n"
        self.assertNotIn("RM003", self.lint_rules(badges(6), "--readme"))
        self.assertIn("RM003", self.lint_rules(badges(7), "--readme"))

    def test_plain_badges_still_count(self):
        one = "![b](https://img.shields.io/x.svg) "
        self.assertNotIn("RM003", self.lint_rules("# T\n\n" + one * 6 + "\n", "--readme"))
        self.assertIn("RM003", self.lint_rules("# T\n\n" + one * 7 + "\n", "--readme"))

    def test_description_of_exactly_120_characters_is_flagged(self):
        def para(n: int) -> str:
            return "# T\n\n" + "a" * n + "\n"
        self.assertIn("RM004", self.lint_rules(para(120), "--readme"))
        self.assertNotIn("RM004", self.lint_rules(para(119), "--readme"))

    def test_code001_source_is_not_attributed_to_google_code_samples(self):
        res = run(LINT, "--list-rules")
        block = res.stdout.split("CODE001", 1)[1].split("CODE002", 1)[0]
        self.assertNotIn("Google style: code samples", block)

    def test_style_rules_word_table_matches_linter(self):
        table = (HERE.parent / "references" / "style-rules.md").read_text(encoding="utf-8")
        section = table.split("## Words to cut or replace", 1)[1]
        words = []
        for row in section.splitlines():
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if not row.startswith("|") or len(cells) < 4 or cells[0] in {"Word or phrase", "---"}:
                continue
            if cells[0].startswith("---") or "not flagged" in cells[0]:
                continue
            for w in cells[0].split(", "):
                words.append(re.sub(r"\s*\(.*?\)", "", w).strip())
        self.assertGreater(len(words), 30, words)
        doc = write(self.root, "w.md", "# T\n\n" + "".join(f"This is {w} here.\n\n" for w in words))
        res = run(LINT, str(doc), "--json")
        flagged = {f["excerpt"] for f in json.loads(res.stdout)[0]["findings"]}
        missing = [w for w in words if not any(f"This is {w} here." == e for e in flagged)]
        self.assertEqual(missing, [])


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

    def test_unreadable_directory_exits_2(self):
        if os.geteuid() == 0:
            self.skipTest("root can read any directory")
        locked = self.root / "locked"
        locked.mkdir()
        locked.chmod(0o000)
        try:
            res = run(SCAN, str(locked))
        finally:
            locked.chmod(0o755)
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_nested_conditional_exports_do_not_crash(self):
        write(self.root, "package.json", json.dumps(
            {"name": "j7", "exports": {".": {"import": {"types": "./dist/index.d.ts",
                                                          "default": "./dist/index.js"}}}}))
        write(self.root, "dist/index.js", "export function alpha() {}\n")
        res = run(SCAN, str(self.root), "--json")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(json.loads(res.stdout)["public_surface_count"], 1)

    def test_env_var_read_only_under_nested_test_dir_is_ignored(self):
        write(self.root, "src/pkg/tests/t.py", "import os\nos.environ.get('ONLY_NESTED_TEST')\n")
        write(self.root, "src/pkg/a.py", "import os\nos.environ.get('REAL_CONFIG_VAR')\n")
        data = json.loads(run(SCAN, str(self.root), "--json").stdout)
        envs = " ".join(data["gaps"]["undocumented_env_vars"])
        self.assertIn("REAL_CONFIG_VAR", envs)
        self.assertNotIn("ONLY_NESTED_TEST", envs)

    def test_docstring_does_not_promise_stale_commands_or_config_keys(self):
        doc = " ".join(SCAN.read_text(encoding="utf-8").split('"""', 2)[1].split())
        self.assertNotIn("commands that no longer exist", doc)
        self.assertNotIn("config key", doc)


class Meta(unittest.TestCase):
    def test_verify_claims_docstring_does_not_promise_a_line_for_every_claim(self):
        doc = " ".join(VERIFY.read_text(encoding="utf-8").split('"""', 2)[1].split())
        self.assertNotIn("when verified, a `file:line` citation", doc)

    def test_run_from_anywhere_commands_do_not_use_relative_paths(self):
        doc = __doc__.split("Run from anywhere", 1)[1].split("Each test", 1)[0]
        commands = [ln.strip() for ln in doc.splitlines() if ln.strip().startswith("python3")]
        self.assertTrue(commands)
        for cmd in commands:
            self.assertNotRegex(cmd, r"(?<![\w/.-])scripts/", cmd)
            self.assertNotRegex(cmd, r"-s scripts\b", cmd)

    def test_suite_runs_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run(
                [sys.executable, "-I", str(HERE / "test_scripts.py"),
                 "LintProse.test_list_rules_needs_no_document"],
                capture_output=True, text=True, cwd=tmp, check=False,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
