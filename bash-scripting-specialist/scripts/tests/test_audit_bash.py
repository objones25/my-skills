#!/usr/bin/env python3
"""Tests for audit_bash.py. Run: python3 scripts/tests/test_audit_bash.py

Each fixture marks the findings it should produce with `# expect: B123 B456` on the
offending line, or `# expect-file: B203` for whole-file findings reported on line 1.
A test passes when the set of (line, rule) findings equals the set of expectations,
so both missed findings and false positives fail.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
SCRIPT = os.path.join(HERE, "..", "audit_bash.py")
ASSETS = os.path.join(HERE, "..", "..", "assets")
sys.path.insert(0, os.path.join(HERE, ".."))

import audit_bash  # noqa: E402


def expected(path: str) -> set[tuple[int, str]]:
    exp = set()
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            m = re.search(r"#\s*expect:\s*([B0-9 ]+)", line)
            if m:
                exp |= {(no, r) for r in m.group(1).split()}
            m = re.search(r"#\s*expect-file:\s*([B0-9 ]+)", line)
            if m:
                exp |= {(1, r) for r in m.group(1).split()}
    return exp


def actual(path: str, target=(4, 4), platform="any") -> set[tuple[int, str]]:
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    return {(f.line, f.rule) for f in audit_bash.audit(path, src, target, platform)}


class FixtureTests(unittest.TestCase):
    def check(self, name: str, **kw) -> None:
        path = os.path.join(FIXTURES, name)
        exp, got = expected(path), actual(path, **kw)
        self.assertEqual(sorted(got - exp), [], f"{name}: unexpected findings")
        self.assertEqual(sorted(exp - got), [], f"{name}: missed findings")

    def test_errexit_traps(self):
        self.check("errexit_traps.sh")

    def test_versions(self):
        self.check("versions.sh")

    def test_gnu_only(self):
        self.check("gnu_only.sh")

    def test_security(self):
        self.check("security.sh")

    def test_indented_code_in_functions(self):
        self.check("indented.sh", target=(3, 2))

    def test_function_endings(self):
        self.check("functions_b106.sh")

    def test_empty_args_on_32(self):
        self.check("args_32.sh", target=(3, 2))

    def test_clean_has_no_findings(self):
        self.check("clean.sh")

    def test_not_bash_is_skipped(self):
        self.assertEqual(actual(os.path.join(FIXTURES, "not_bash.sh")), set())

    def test_missing_shebang(self):
        self.assertIn((1, "B401"), actual(os.path.join(FIXTURES, "no_shebang.sh")))

    def test_linux_platform_skips_gnu_checks(self):
        got = actual(os.path.join(FIXTURES, "gnu_only.sh"), platform="linux")
        self.assertFalse(any(rule == "B204" for _, rule in got))

    def test_template_is_clean_for_macos_bash(self):
        got = actual(os.path.join(ASSETS, "template.sh"), target=(3, 2), platform="macos")
        self.assertEqual(sorted(got), [])


class SuppressionTests(unittest.TestCase):
    def run_src(self, src: str):
        return {(f.line, f.rule) for f in audit_bash.audit("x.sh", src, (4, 4), "any")}

    HEAD = "#!/usr/bin/env bash\nset -Eeuo pipefail\nshopt -s inherit_errexit\n"

    def test_same_line(self):
        self.assertEqual(self.run_src(self.HEAD + 'eval "$x" # audit-bash disable=B301\n'), set())

    def test_line_above(self):
        self.assertEqual(self.run_src(self.HEAD + "# audit-bash disable=B301\neval \"$x\"\n"), set())

    def test_file_level(self):
        src = self.HEAD + "# audit-bash disable-file=B301\n\neval a\n\neval b\n"
        self.assertEqual(self.run_src(src), set())

    def test_other_rule_not_suppressed(self):
        got = self.run_src(self.HEAD + 'eval "$x" # audit-bash disable=B999\n')
        self.assertEqual(got, {(4, "B301")})


class LexerTests(unittest.TestCase):
    def test_hash_in_parameter_expansion_is_not_a_comment(self):
        (line,) = audit_bash.lex('echo "${#arr[@]}" $# # real comment')
        self.assertIn("${#arr[@]}", line.code)
        self.assertNotIn("real comment", line.code)

    def test_single_quotes_masked_double_quotes_kept(self):
        (line,) = audit_bash.lex("echo 'eval x' \"$y\"")
        self.assertNotIn("eval", line.code)
        self.assertIn('"$y"', line.code)

    def test_heredoc_body_skipped(self):
        lines = audit_bash.lex("cat <<-EOF\n\teval x\n\tEOF\necho after")
        self.assertEqual([l.code for l in lines], ["cat <<-EOF", "echo after"])

    def test_ansi_c_string_with_escaped_quote(self):
        lines = audit_bash.lex("q=$'it\\'s'\neval \"$1\"")
        self.assertEqual(lines[1].code, 'eval "$1"')

    def test_continuation_joined(self):
        lines = audit_bash.lex("curl -fsSL https://x \\\n  | bash")
        self.assertEqual(len(lines), 1)
        self.assertIn("| bash", lines[0].code)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)

    def test_rules_lists_every_rule(self):
        r = self.run_cli("--rules")
        self.assertEqual(r.returncode, 0)
        for rule_id in audit_bash.RULES:
            self.assertIn(rule_id, r.stdout)

    def test_exit_codes(self):
        self.assertEqual(self.run_cli(os.path.join(FIXTURES, "clean.sh")).returncode, 0)
        self.assertEqual(self.run_cli(os.path.join(FIXTURES, "security.sh")).returncode, 1)  # has an error (B302)
        self.assertEqual(self.run_cli(os.path.join(FIXTURES, "gnu_only.sh")).returncode, 0)  # warnings only
        self.assertEqual(self.run_cli("--strict", os.path.join(FIXTURES, "gnu_only.sh")).returncode, 1)
        self.assertEqual(self.run_cli("--target", "x", "a.sh").returncode, 2)

    def test_json(self):
        import json
        r = self.run_cli("--json", os.path.join(FIXTURES, "security.sh"))
        data = json.loads(r.stdout)
        self.assertTrue(all({"path", "line", "rule", "message", "severity"} <= d.keys() for d in data))


if __name__ == "__main__":
    unittest.main(verbosity=1)
