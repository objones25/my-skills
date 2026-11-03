#!/usr/bin/env python3
"""Check a skill folder against the skill standard.

Usage:
  check_skill.py SKILL_DIR [--json] [--strict] [--ignore CODE ...]
  check_skill.py --list-codes

Exit codes: 0 no errors, 1 errors found (or warnings with --strict), 2 could not check.
Standard library only. If PyYAML is installed, frontmatter gets a strict YAML parse. Without
it, a small parser handles common frontmatter and exits 2 on anything it cannot read reliably.
"""
from __future__ import annotations

import argparse
import json
import os
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
    "SB014": ("warning", "TODO left in the skill's own files"),
    "SB015": ("error", "file cannot be read as UTF-8 text"),
    "SB020": ("error", "scripts with no tests"),
    "SB022": ("warning", "tests exist but metadata.test-command does not say how to run them"),
    "SB030": ("warning", "cites URLs but has no references/sources.md or references/evidence.md"),
    "SB031": ("warning", "has a source record but no AUDIT.md"),
    "SB040": ("warning", "reference file is not routed from SKILL.md"),
    "SB041": ("warning", "large skill (over 150 lines or several references) with no test of its structure"),
    "SB042": ("warning", "SKILL.md is over 500 lines"),
    "SB043": ("warning", "large skill (over 150 lines or several references) with no 'Extend this skill' section"),
    "SB050": ("warning", "test result stated without a date and machine"),
    "SB051": ("warning", "reference file names a file that does not exist"),
}

JUNK_NAMES = {".DS_Store", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".hypothesis", ".coverage"}
ENVIRONMENT_NAMES = {".git", ".venv", "venv", ".tox", "node_modules", ".aws", ".ssh"}
CREDENTIAL_NAMES = {".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}
CREDENTIAL_SUFFIXES = {".pem", ".key", ".pyc"}
ENV_TEMPLATES = {".env.example", ".env.sample"}
BUNDLE_DIRS = "references|scripts|assets|examples|tests|evals|selftest|modes|templates"
PATH_RE = re.compile(rf"`((?:{BUNDLE_DIRS})/[^`\s]+)`")
PROJECT_FILES = {"CONTRIBUTING.md", "README.md", "NOTES.md", "CLAUDE.md", "AGENTS.md", "CHANGELOG.md", "AUDIT.md", "INSTALL.md"}
BARE_MD_RE = re.compile(r"`([A-Za-z0-9_.-]+\.md)`")
WHEN_RE = re.compile(r"\b[Uu]se (?:this |it )?(?:when|whenever|to|for)\b|\bTriggers? on\b|\bwhen the user\b")
NEGATIVE_RE = re.compile(r"\bNot for\b|\bnot for\b|\b[Dd]o not use\b|\bFor [^.]{1,60}\buse [a-z0-9:-]+")
RESULT_RE = re.compile(r"\b\d+ (?:passed|failed)\b|\b[Tt]ested on (?:[A-Z0-9]|v\d)")
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b(?:January|February|March|April|May|June|July|August|September|"
                     r"October|November|December) 20\d\d\b")
MACHINE_RE = re.compile(r"macos|linux|windows|arm64|aarch64|x86|apple silicon|not recorded", re.I)
TEST_FILE_RE = re.compile(r"^(test_.*\.py|.*_test\.py|.*\.bats|run_all\.sh)$")
STRUCTURE_TEST_RE = re.compile(r"^test_skill.*\.py$")
NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RESERVED_NAMES = {"synced", "anthropic-skills"}  # Claude Code does not load these folders outside plugins
TODO_RE = re.compile(r"\bTODO\b")
QUOTED_RE = re.compile(r'"[^"]*"|`[^`]*`')


@dataclass
class Finding:
    code: str
    severity: str
    path: str
    message: str


def is_excluded(relative: Path) -> bool:
    """Files that never belong in a skill's checks or its zip: junk, environments, credentials."""
    for part in relative.parts:
        lowered = part.lower()
        if part in JUNK_NAMES or part in ENVIRONMENT_NAMES or part in CREDENTIAL_NAMES:
            return True
        if lowered not in ENV_TEMPLATES and (lowered.startswith(".env") or lowered.endswith(".env")):
            return True
    return relative.suffix.lower() in CREDENTIAL_SUFFIXES


def walk_files(root: Path) -> list[Path]:
    """Every file under root, raising OSError on a folder that cannot be read."""
    def fail(error: OSError) -> None:
        raise error

    found = []
    for folder, subfolders, names in os.walk(root, onerror=fail):
        subfolders[:] = [d for d in subfolders if d not in ENVIRONMENT_NAMES]
        found.extend(Path(folder) / name for name in names)
    return sorted(p for p in found if p.is_file())


class NeedsPyYAML(Exception):
    """The frontmatter uses YAML the small parser will not guess at."""


def parse_frontmatter(text: str) -> tuple[dict | None, str | None]:
    """Return (fields, problem), or raise NeedsPyYAML."""
    match = re.match(r"^---\n(.*?)\n---\n", text.replace("\r\n", "\n"), re.S)
    if match is None:
        return None, "no YAML frontmatter block at the top of the file"
    block = match.group(1)
    if yaml is not None:
        try:
            loaded = yaml.safe_load(block)
        except yaml.YAMLError as exc:
            return None, f"invalid YAML: {str(exc).splitlines()[0]}"
        return (loaded if isinstance(loaded, dict) else {}), None
    return _parse_mapping(block.split("\n"))


def _parse_mapping(lines: list[str], depth: int = 0) -> tuple[dict | None, str | None]:
    fields: dict = {}
    i = 0
    while i < len(lines):
        match = re.match(r"^([A-Za-z_-]+):(?:\s+(.*))?$", lines[i])
        i += 1
        if match is None:
            if lines[i - 1].strip() and not lines[i - 1].lstrip().startswith("#"):
                raise NeedsPyYAML(f"cannot read line: {lines[i - 1]!r}")
            continue
        key, raw = match.group(1), (match.group(2) or "").strip()
        nested = []
        while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
            nested.append(lines[i])
            i += 1
        value, problem = _parse_value(key, raw, nested, depth)
        if problem:
            return None, problem
        fields[key] = value
    return fields, None


def _parse_value(key: str, raw: str, nested: list[str], depth: int) -> tuple[object, str | None]:
    text = [line.strip() for line in nested if line.strip()]
    quoted = re.match(r"""^("(?:[^"\\]|\\.)*"|'(?:[^']|'')*')(?:\s+#.*)?$""", raw)
    if quoted:
        if text:
            raise NeedsPyYAML(f"{key} has text after its closing quote")
        literal = quoted.group(1)
        if literal[0] == '"':
            try:
                return json.loads(literal), None
            except json.JSONDecodeError as exc:
                raise NeedsPyYAML(f"{key} uses escapes this parser cannot read") from exc
        return literal[1:-1].replace("''", "'"), None
    if raw[:1] in ('"', "'"):
        if text:
            raise NeedsPyYAML(f"{key} has a quoted value over several lines")
        return None, f"invalid YAML: unterminated quote in {key}"
    if raw[:1] in ("{", "[", "&", "*", "!"):
        raise NeedsPyYAML(f"{key} uses YAML this parser cannot read")
    value = re.sub(r"\s+#.*$", "", raw)
    if value[:1] == ">" and re.fullmatch(r">[+-]?", value):
        return " ".join(text), None
    if value[:1] in (">", "|"):
        raise NeedsPyYAML(f"{key} uses a block style this parser will not guess at")
    if not value:
        if not text:
            return "", None
        if depth > 0:
            raise NeedsPyYAML(f"{key} is nested more deeply than this parser reads")
        indent = min(len(line) - len(line.lstrip()) for line in nested if line.strip())
        return _parse_mapping([line[indent:] for line in nested], depth + 1)
    if ": " in value or value.endswith(":"):
        return None, f"invalid YAML: unquoted colon in {key}"
    return " ".join([value, *text]), None


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
        self.files = walk_files(root)
        self.skill_text = (root / "SKILL.md").read_text(encoding="utf-8")
        self._texts: dict[Path, str] = {}

    def read(self, path: Path) -> str:
        """Text of a file, or empty after reporting SB015 once if it cannot be read."""
        if path not in self._texts:
            try:
                self._texts[path] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                self._texts[path] = ""
                self.add("SB015", self.rel(path), type(exc).__name__)
        return self._texts[path]

    def add(self, code: str, path: str, detail: str = "") -> None:
        severity, summary = RULES[code]
        self.findings.append(Finding(code, severity, path, f"{summary}{': ' + detail if detail else ''}"))

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def clean_files(self) -> list[Path]:
        return [p for p in self.files if not is_excluded(p.relative_to(self.root))]

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
        self.check_todos()
        return self.findings

    def check_frontmatter(self) -> None:
        self.fields, problem = parse_frontmatter(self.skill_text)
        if self.fields is None:
            self.add("SB002", "SKILL.md", problem or "")
            self.fields = {}
            return
        name = self.fields.get("name")
        if (name != self.root.name or not NAME_RE.fullmatch(str(name)) or len(str(name)) > 64
                or name in RESERVED_NAMES):
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
            for name in sorted(set(BARE_MD_RE.findall(self.read(path))) - PROJECT_FILES):
                if not any(c.name == name for c in self.files):
                    self.add("SB051", self.rel(path), name)

    def check_core_files(self) -> None:
        if not (self.root / "CHANGELOG.md").exists():
            self.add("SB008", "CHANGELOG.md")
        for path in self.clean_files():
            if path.suffix == ".md" and "\u2014" in self.read(path):
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
                cases = json.loads(self.read(trigger_path))
                kinds = {c.get("should_trigger") for c in cases if isinstance(c, dict) and c.get("query")}
            except (json.JSONDecodeError, TypeError, AttributeError):
                kinds = set()
        if not {True, False} <= kinds:
            self.add("SB010", "evals/trigger_evals.json")

    def _evals_problem(self, path: Path) -> str:
        try:
            data = json.loads(self.read(path))
        except json.JSONDecodeError as exc:
            return f"not JSON: {exc}"
        if not isinstance(data, dict) or data.get("skill_name") != self.root.name:
            return f"skill_name must be {self.root.name!r}"
        cases = data.get("evals")
        if not isinstance(cases, list) or not cases:
            return "evals must be a non-empty list"
        ids = [c.get("id") for c in cases if isinstance(c, dict)]
        if (len(ids) != len(cases) or len(set(map(str, ids))) != len(ids)
                or not all((isinstance(i, int) and not isinstance(i, bool)) or (isinstance(i, str) and i) for i in ids)):
            return "every case needs a unique id, an integer or a name"
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
        cites = any(re.search(r"https?://", self.skill_text if p.name == "SKILL.md" else self.read(p)) for p in documents)
        if cites and not has_record:
            self.add("SB030", "references/sources.md")
        if has_record and not (self.root / "AUDIT.md").exists():
            self.add("SB031", "AUDIT.md")

    def check_todos(self) -> None:
        own = [self.root / n for n in ("SKILL.md", "CHANGELOG.md", "AUDIT.md", "references/sources.md",
                                       "evals/evals.json", "evals/trigger_evals.json")]
        for path in own:
            if not path.exists():
                continue
            text = self.read(path)
            if path.suffix == ".md":
                frontmatter = re.match(r"^---\n.*?\n---\n", text, re.S)
                head = frontmatter.group(0) if frontmatter else ""
                body = text[len(head):]
                text = head + "\n".join(QUOTED_RE.sub("", line) for _, line in _markdown_lines_outside_code(body))
            if TODO_RE.search(text):
                self.add("SB014", self.rel(path))

    def check_size_and_routing(self) -> None:
        lines = self.skill_text.count("\n")
        references = self.root / "references"
        routed = [p for p in references.rglob("*.md") if p.name not in ("sources.md", "evidence.md", "CHANGELOG.md", "README.md")]
        large = lines > 150 or len(routed) > 1
        if large and not self.has_structure_test:
            self.add("SB041", "SKILL.md", f"{lines} lines, {len(routed)} references")
        if large and "Extend this skill" not in self.skill_text:
            self.add("SB043", "SKILL.md")
        if lines > 500:
            self.add("SB042", "SKILL.md", f"{lines} lines")
        if not references.is_dir():
            return
        for path in sorted(references.rglob("*.md")):
            if is_excluded(path.relative_to(self.root)) or path.name in ("sources.md", "evidence.md"):
                continue
            rel = self.rel(path)
            index = self.rel(path.parent / "README.md")
            if path.name not in self.skill_text and not (path.parent != references and index in self.skill_text):
                self.add("SB040", rel)

    def check_results(self) -> None:
        for path in self.clean_files():
            parts = path.relative_to(self.root).parts
            if path.suffix != ".md" or parts[0] in ("evals", "examples", "tests", "assets") or "papers" in parts:
                continue
            lines = _markdown_lines_outside_code(self.read(path))
            headings: dict[int, str] = {}
            for position, (number, line) in enumerate(lines):
                heading = re.match(r"^(#+) ", line)
                if heading:
                    depth = len(heading.group(1))
                    headings = {d: text for d, text in headings.items() if d < depth}
                    headings[depth] = line
                if not RESULT_RE.search(QUOTED_RE.sub("", line)):
                    continue
                recent = [text for _, text in lines[max(0, position - 2):position + 1]]
                context = " ".join([*headings.values(), *recent])
                if "not recorded" in context:
                    continue
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
    unknown = sorted(set(args.ignore) - set(RULES))
    if unknown:
        print(f"unknown code for --ignore: {', '.join(unknown)}", file=sys.stderr)
        return 2
    root = args.skill_dir.resolve()
    if not (root / "SKILL.md").is_file():
        print(f"could not check {root}: no SKILL.md", file=sys.stderr)
        return 2
    try:
        findings = [f for f in Checker(root).run() if f.code not in args.ignore]
    except NeedsPyYAML as exc:
        print(f"could not check {root}: {exc}. Install PyYAML to read this frontmatter.", file=sys.stderr)
        return 2
    except (OSError, UnicodeDecodeError) as exc:
        print(f"could not check {root}: cannot read SKILL.md: {exc}", file=sys.stderr)
        return 2
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
