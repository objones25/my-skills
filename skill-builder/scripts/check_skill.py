#!/usr/bin/env python3
"""Check a skill folder against the skill standard.

Usage:
  check_skill.py SKILL_DIR [--json] [--strict] [--ignore CODE ...]
  check_skill.py --list-codes

Exit codes: 0 no errors, 1 errors found (or warnings with --strict), 2 could not check.
Standard library only. If PyYAML is installed, frontmatter also gets a strict YAML parse.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

RULES = {
    "SB002": ("error", "frontmatter is missing or is not valid YAML"),
    "SB003": ("error", "name must match the folder and use lowercase letters, digits and single hyphens (64 max)"),
    "SB004": ("error", "description must exist, be at most 1024 characters, and have no angle brackets"),
    "SB005": ("warning", "description does not say when to use the skill"),
    "SB006": ("warning", "description has no negative scope or handoff"),
    "SB007": ("warning", "SKILL.md names a bundled path that does not exist (ignore this if it means a path in the user's project)"),
    "SB008": ("warning", "no CHANGELOG.md at the skill root"),
    "SB009": ("warning", "no evals/evals.json"),
    "SB010": ("warning", "evals/trigger_evals.json is missing or lacks both should-trigger and should-not-trigger cases"),
    "SB011": ("error", "evals/evals.json does not match the schema"),
    "SB012": ("error", "cache or OS junk in the skill folder"),
    "SB013": ("warning", "em dash in a markdown file"),
    "SB020": ("error", "scripts with no tests"),
    "SB022": ("warning", "tests exist but metadata.test-command does not say how to run them"),
    "SB030": ("warning", "cites URLs but has no references/sources.md or references/evidence.md"),
    "SB031": ("warning", "has a source record but no AUDIT.md"),
    "SB040": ("warning", "reference file is not routed from SKILL.md"),
    "SB041": ("warning", "SKILL.md is over 150 lines and no test checks the skill's structure"),
    "SB042": ("warning", "SKILL.md is over 500 lines"),
    "SB050": ("warning", "test result stated without a date and machine"),
    "SB051": ("warning", "reference file names a file that does not exist"),
}

JUNK_NAMES = {".DS_Store", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".hypothesis", ".coverage"}
BUNDLE_DIRS = "references|scripts|assets|examples|tests|evals|selftest|modes|templates"
PATH_RE = re.compile(rf"`((?:{BUNDLE_DIRS})/[^`\s]+)`")
PROJECT_FILES = {"CONTRIBUTING.md", "README.md", "NOTES.md", "CLAUDE.md", "AGENTS.md", "CHANGELOG.md", "AUDIT.md", "INSTALL.md"}
BARE_MD_RE = re.compile(r"`([A-Za-z0-9_.-]+\.md)`")
WHEN_RE = re.compile(r"\b[Uu]se (?:this |it )?(?:when|whenever|to|for)\b|\bTriggers? on\b|\bwhen the user\b")
NEGATIVE_RE = re.compile(r"\bNot for\b|\bnot for\b|\b[Dd]o not use\b|\bFor [^.]{1,60}\buse [a-z0-9:-]+")
RESULT_RE = re.compile(r"\b\d+ (?:passed|failed)\b|\b[Tt]ested on\b")
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b(?:January|February|March|April|May|June|July|August|September|"
                     r"October|November|December) 20\d\d\b")
MACHINE_RE = re.compile(r"macos|linux|windows|arm64|aarch64|x86|apple silicon|not recorded", re.I)
TEST_FILE_RE = re.compile(r"^(test_.*\.py|.*_test\.py|.*\.bats|run_all\.sh)$")
STRUCTURE_TEST_RE = re.compile(r"^test_skill.*\.py$")
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class Finding:
    code: str
    severity: str
    path: str
    message: str


def parse_frontmatter(text: str) -> tuple[dict | None, str | None]:
    """Return (fields, problem). A small parser for the fields this script needs."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if match is None:
        return None, "no YAML frontmatter block at the top of the file"
    block = match.group(1)
    if yaml is not None:
        try:
            loaded = yaml.safe_load(block)
        except yaml.YAMLError as exc:
            return None, f"invalid YAML: {str(exc).splitlines()[0]}"
        return (loaded if isinstance(loaded, dict) else {}), None
    return _parse_without_yaml(block)


def _parse_without_yaml(block: str) -> tuple[dict | None, str | None]:
    fields: dict = {}
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        match = re.match(r"^([A-Za-z_-]+):\s*(.*)$", lines[i])
        i += 1
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        nested = []
        while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
            nested.append(lines[i].strip())
            i += 1
        if value in (">", "|", ">-", "|-", ">+", "|+"):
            fields[key] = " ".join(part for part in nested if part)
        elif not value:
            fields[key] = dict(re.match(r"^([A-Za-z_-]+):\s*(.*)$", n).groups() for n in nested
                               if re.match(r"^[A-Za-z_-]+:", n))
            fields[key] = {k: v.strip("\"'") for k, v in fields[key].items()}
        elif value[0] in "\"'":
            if value[-1] != value[0] or len(value) < 2:
                return None, f"invalid YAML: unterminated quote in {key}"
            fields[key] = value[1:-1]
        elif ": " in value or value.endswith(":"):
            return None, f"invalid YAML: unquoted colon in {key}"
        else:
            fields[key] = value
    return fields, None


def _markdown_lines_outside_code(text: str) -> list[tuple[int, str]]:
    out, fenced = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            out.append((number, line))
    return out


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.findings: list[Finding] = []
        self.files = sorted(p for p in root.rglob("*") if p.is_file())
        self.skill_text = (root / "SKILL.md").read_text(encoding="utf-8")

    def add(self, code: str, path: str, detail: str = "") -> None:
        severity, summary = RULES[code]
        self.findings.append(Finding(code, severity, path, f"{summary}{': ' + detail if detail else ''}"))

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def clean_files(self) -> list[Path]:
        return [p for p in self.files if not JUNK_NAMES.intersection(p.relative_to(self.root).parts)]

    def run(self) -> list[Finding]:
        self.check_frontmatter()
        self.check_paths()
        self.check_core_files()
        self.check_evals()
        self.check_junk()
        self.check_tooling()
        self.check_knowledge()
        self.check_size_and_routing()
        self.check_results()
        return self.findings

    def check_frontmatter(self) -> None:
        self.fields, problem = parse_frontmatter(self.skill_text)
        if self.fields is None:
            self.add("SB002", "SKILL.md", problem or "")
            self.fields = {}
            return
        name = self.fields.get("name")
        if name != self.root.name or not NAME_RE.match(str(name)) or len(str(name)) > 64:
            self.add("SB003", "SKILL.md", f"name is {name!r}, folder is {self.root.name!r}")
        description = self.fields.get("description")
        description = description if isinstance(description, str) else ""
        if not description or len(description) > 1024 or re.search(r"[<>]", description):
            self.add("SB004", "SKILL.md", f"{len(description)} characters")
            return
        if not WHEN_RE.search(description):
            self.add("SB005", "SKILL.md")
        if not NEGATIVE_RE.search(description):
            self.add("SB006", "SKILL.md")

    def check_paths(self) -> None:
        for mention in sorted(set(PATH_RE.findall(self.skill_text))):
            mention = mention.rstrip(".,")
            if re.search(r"[<>{}*$\[]|\.\.\.", mention):
                continue
            if not (self.root / mention).exists():
                self.add("SB007", "SKILL.md", mention)
        for path in self.clean_files():
            if (path.suffix != ".md" or path.relative_to(self.root).parts[0] != "references"
                    or path.name in ("CHANGELOG.md", "AUDIT.md")):
                continue
            for name in sorted(set(BARE_MD_RE.findall(path.read_text(encoding="utf-8"))) - PROJECT_FILES):
                if not any(c.name == name for c in self.files):
                    self.add("SB051", self.rel(path), name)

    def check_core_files(self) -> None:
        if not (self.root / "CHANGELOG.md").exists():
            self.add("SB008", "CHANGELOG.md")
        for path in self.clean_files():
            if path.suffix == ".md" and "—" in path.read_text(encoding="utf-8"):
                self.add("SB013", self.rel(path))

    def check_evals(self) -> None:
        evals_path = self.root / "evals" / "evals.json"
        if not evals_path.exists():
            self.add("SB009", "evals/evals.json")
        else:
            problem = self._evals_problem(evals_path)
            if problem:
                self.add("SB011", "evals/evals.json", problem)
        trigger_path = self.root / "evals" / "trigger_evals.json"
        kinds = set()
        if trigger_path.exists():
            try:
                cases = json.loads(trigger_path.read_text(encoding="utf-8"))
                kinds = {c.get("should_trigger") for c in cases if isinstance(c, dict) and c.get("query")}
            except (json.JSONDecodeError, TypeError, AttributeError):
                kinds = set()
        if not {True, False} <= kinds:
            self.add("SB010", "evals/trigger_evals.json")

    def _evals_problem(self, path: Path) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return f"not JSON: {exc}"
        if not isinstance(data, dict) or data.get("skill_name") != self.root.name:
            return f"skill_name must be {self.root.name!r}"
        cases = data.get("evals")
        if not isinstance(cases, list) or not cases:
            return "evals must be a non-empty list"
        ids = [c.get("id") for c in cases if isinstance(c, dict)]
        if len(ids) != len(cases) or len(set(map(str, ids))) != len(ids):
            return "every case needs a unique id"
        for case in cases:
            if not case.get("prompt"):
                return f"case {case.get('id')} has no prompt"
            expectations = case.get("expectations")
            if not isinstance(expectations, list) or not expectations:
                return f"case {case.get('id')} needs a non-empty expectations list"
        return ""

    def check_junk(self) -> None:
        junk = sorted({part for p in self.files for part in p.relative_to(self.root).parts if part in JUNK_NAMES})
        for name in junk:
            self.add("SB012", name)

    def check_tooling(self) -> None:
        clean = self.clean_files()
        outside_samples = [p for p in clean if p.relative_to(self.root).parts[0] not in ("evals", "examples")]
        tests = [p for p in outside_samples if TEST_FILE_RE.match(p.name)
                 or (p.suffix == ".py" and p.relative_to(self.root).parts[0] == "tests" and p.name != "conftest.py")]
        scripts = [p for p in clean if p.relative_to(self.root).parts[0] == "scripts"
                   and p.suffix in (".py", ".sh") and not TEST_FILE_RE.match(p.name) and "tests" not in p.parts]
        if scripts and not tests:
            self.add("SB020", "scripts/", f"{len(scripts)} script(s)")
        metadata = self.fields.get("metadata")
        if tests and not (isinstance(metadata, dict) and metadata.get("test-command")):
            self.add("SB022", "SKILL.md")
        self.has_structure_test = any(STRUCTURE_TEST_RE.match(p.name) for p in tests)

    def check_knowledge(self) -> None:
        references = self.root / "references"
        has_record = (references / "sources.md").exists() or (references / "evidence.md").exists()
        documents = [self.root / "SKILL.md", *(p for p in self.clean_files()
                     if p.suffix == ".md" and p.relative_to(self.root).parts[0] == "references")]
        cites = any(re.search(r"https?://", p.read_text(encoding="utf-8")) for p in documents)
        if cites and not has_record:
            self.add("SB030", "references/sources.md")
        if has_record and not (self.root / "AUDIT.md").exists():
            self.add("SB031", "AUDIT.md")

    def check_size_and_routing(self) -> None:
        lines = self.skill_text.count("\n")
        if lines > 150 and not self.has_structure_test:
            self.add("SB041", "SKILL.md", f"{lines} lines")
        if lines > 500:
            self.add("SB042", "SKILL.md", f"{lines} lines")
        references = self.root / "references"
        if not references.is_dir():
            return
        for path in sorted(references.rglob("*.md")):
            if JUNK_NAMES.intersection(path.parts) or path.name in ("sources.md", "evidence.md"):
                continue
            rel = self.rel(path)
            index = self.rel(path.parent / "README.md")
            if path.name not in self.skill_text and not (path.parent != references and index in self.skill_text):
                self.add("SB040", rel)

    def check_results(self) -> None:
        for path in self.clean_files():
            parts = path.relative_to(self.root).parts
            if path.suffix != ".md" or parts[0] in ("evals", "examples", "tests", "assets"):
                continue
            lines = _markdown_lines_outside_code(path.read_text(encoding="utf-8"))
            heading = ""
            for position, (number, line) in enumerate(lines):
                if line.startswith("#"):
                    heading = line
                if not RESULT_RE.search(re.sub(r'"[^"]*"|`[^`]*`', "", line)):
                    continue
                context = " ".join([heading, *(text for _, text in lines[max(0, position - 2):position + 1])])
                if not (DATE_RE.search(context) and MACHINE_RE.search(context)):
                    self.add("SB050", f"{self.rel(path)}:{number}", line.strip()[:80])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skill_dir", nargs="?", type=Path)
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--ignore", action="append", default=[], metavar="CODE", help="skip this rule; repeatable")
    parser.add_argument("--list-codes", action="store_true", help="print every rule and exit")
    args = parser.parse_args()
    if args.list_codes:
        for code, (severity, summary) in RULES.items():
            print(f"{code}  {severity:7}  {summary}")
        return 0
    if args.skill_dir is None:
        parser.print_usage(sys.stderr)
        return 2
    root = args.skill_dir.resolve()
    if not (root / "SKILL.md").is_file():
        print(f"could not check {root}: no SKILL.md", file=sys.stderr)
        return 2
    findings = [f for f in Checker(root).run() if f.code not in args.ignore]
    errors = sum(f.severity == "error" for f in findings)
    warnings = len(findings) - errors
    if args.json:
        print(json.dumps({"skill": root.name, "errors": errors, "warnings": warnings,
                          "findings": [asdict(f) for f in findings]}, indent=2))
    else:
        for f in findings:
            print(f"{f.severity:7} {f.code} {f.path}: {f.message}")
        print(f"{root.name}: {errors} error(s), {warnings} warning(s)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
