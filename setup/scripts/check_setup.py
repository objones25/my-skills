#!/usr/bin/env python3
"""Lint the Claude Code project configuration that the setup skill writes.

Usage: check_setup.py [PROJECT_DIR] [--strict] [--json]

Checks CLAUDE.md, .claude/settings*.json, .claude/rules, .claude/agents,
.claude/skills and .mcp.json for mistakes that load silently or leak secrets.
Hooks are linted by the hook-writer skill's check_hooks.py, not here.

Exit codes: 0 no errors (warnings allowed unless --strict), 1 findings, 2 could not check.
Secret values are never printed: findings name the file and key only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CLAUDE_MD_LINES_MAX = 200  # memory docs: "target under 200 lines per CLAUDE.md file"
SENSITIVE_KEY = re.compile(r"TOKEN|SECRET|PASSW|API_?KEY|AUTH|CREDENTIAL|PRIVATE", re.IGNORECASE)
SECRET_PREFIX = re.compile(r"\b(ghp_|gho_|github_pat_|sk-|xox[abp]-|AKIA|AIza|glpat-)[A-Za-z0-9_-]{6,}")
ENV_REFERENCE = re.compile(r"\$\{[^}]*\}")
LONG_RUN = re.compile(r"[A-Za-z0-9_\-]{12,}")
IMPORT = re.compile(r"(?<![\w.@])@(~?[\w./-]*[./][\w./-]*)")
QUERY_SECRET = re.compile(r"(token|key|secret|passw\w*|auth\w*)[\w-]*=([^&\s]+)", re.IGNORECASE)
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
PERMISSION_LISTS = ("allow", "ask", "deny")


class CannotCheck(Exception):
    """A file that must be read is unreadable or malformed."""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CannotCheck(f"cannot read {path}: {exc}") from exc


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" or "warn"
    code: str
    path: str
    message: str


def parse_frontmatter(text: str) -> dict[str, object] | None:
    """Parse the YAML subset used in Claude Code frontmatter: scalars, string lists, and | or > blocks.

    Nested mappings are skipped (their indented lines are ignored), which is enough for the
    top-level keys this skill reads.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    data: dict[str, object] = {}
    key: str | None = None
    block: list[str] | None = None  # lines of an open | or > block scalar
    folded = False
    for line in lines[1:]:
        if line.strip() == "---":
            return data
        if block is not None and key is not None and (line.startswith((" ", "\t")) or not line.strip()):
            block.append(line.strip())
            data[key] = (" " if folded else "\n").join(b for b in block if b) or None
            continue
        block = None
        item = re.match(r"^\s+-\s*(.*)$", line)
        if item and key is not None:
            current = data.get(key)
            values = current if isinstance(current, list) else []
            values.append(item.group(1).strip().strip("\"'"))
            data[key] = values
            continue
        if re.match(r"^[A-Za-z_][\w-]*:", line):
            key, value = (part.strip() for part in line.split(":", 1))
            if re.fullmatch(r"[|>][+-]?", value):
                block, folded = [], value.startswith(">")
                data[key] = None
            elif value.startswith("[") and value.endswith("]"):
                data[key] = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
            else:
                data[key] = value.strip("\"'") if value else None
    return None  # no closing fence


def _looks_secret(key: str, value: object) -> bool:
    if not isinstance(value, str):
        return False
    literal = ENV_REFERENCE.sub("", value)
    if SECRET_PREFIX.search(literal):
        return True
    return bool(SENSITIVE_KEY.search(key)) and bool(LONG_RUN.search(literal))


def _secret_in_text(value: object) -> bool:
    """A literal token inside free text such as a command argument or a URL query."""
    if not isinstance(value, str):
        return False
    literal = ENV_REFERENCE.sub("", value)
    if SECRET_PREFIX.search(literal):
        return True
    return any(LONG_RUN.search(m.group(2)) for m in QUERY_SECRET.finditer(literal))


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _load_json(path: Path, root: Path, code: str, findings: list[Finding]) -> object | None:
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        findings.append(Finding("error", code, _rel(path, root), f"not valid JSON: {exc}"))
        return None


def check_settings(root: Path, findings: list[Finding]) -> None:
    for name in ("settings.json", "settings.local.json"):
        path = root / ".claude" / name
        if not path.is_file():
            continue
        data = _load_json(path, root, "ST001", findings)
        if data is None:
            continue
        rel = _rel(path, root)
        if not isinstance(data, dict):
            findings.append(Finding("error", "ST001", rel, "top level must be a JSON object"))
            continue
        permissions = data.get("permissions", {})
        if not isinstance(permissions, dict):
            findings.append(Finding("error", "ST001", rel, "'permissions' must be an object"))
            permissions = {}
        for kind in PERMISSION_LISTS:
            rules = permissions.get(kind, [])
            if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
                findings.append(Finding("error", "ST001", rel,
                                        f"permissions.{kind} must be a list of rule strings such as \"Bash(npm test *)\""))
        env = data.get("env", {})
        if name == "settings.json" and isinstance(env, dict):
            for key, value in env.items():
                if _looks_secret(key, value):
                    findings.append(Finding("error", "ST007", rel,
                                            f"env.{key} holds a literal secret in a committed file; "
                                            "move it to settings.local.json or the shell environment"))


def _git_root(root: Path) -> bool:
    result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root,
                            capture_output=True, text=True, check=False)
    return result.returncode == 0


def check_gitignore(root: Path, findings: list[Finding]) -> None:
    if not _git_root(root):
        return
    for rel in (".claude/settings.local.json", "CLAUDE.local.md"):
        if not (root / rel).exists():
            continue
        result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=root, check=False)
        if result.returncode != 0:
            findings.append(Finding("error", "ST002", rel,
                                    f"personal file is not gitignored; add '{rel}' to .gitignore"))


def strip_code(text: str) -> str:
    """Drop fenced blocks (a fence closes only on the same character, at least as long) and code spans."""
    kept: list[str] = []
    fence = ""
    for line in text.splitlines():
        opening = FENCE.match(line)
        if fence:
            if opening and opening.group(1)[0] == fence[0] and len(opening.group(1)) >= len(fence) \
                    and not line.strip()[len(opening.group(1)):].strip():
                fence = ""
            continue
        if opening:
            fence = opening.group(1)
            continue
        kept.append(line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(kept))


def check_claude_md(root: Path, findings: list[Finding]) -> None:
    candidates = [p for p in (root / "CLAUDE.md", root / ".claude" / "CLAUDE.md") if p.is_file()]
    if not candidates:
        findings.append(Finding("warn", "ST010", "CLAUDE.md",
                                "no project CLAUDE.md (./CLAUDE.md or ./.claude/CLAUDE.md)"))
    local = root / "CLAUDE.local.md"
    for path in candidates + ([local] if local.is_file() else []):
        text = read_text(path)
        rel = _rel(path, root)
        lines = len(text.splitlines())
        if lines > CLAUDE_MD_LINES_MAX:
            findings.append(Finding("warn", "ST004", rel,
                                    f"{lines} lines; the memory docs target under {CLAUDE_MD_LINES_MAX}. "
                                    "Move procedures to skills and area rules to .claude/rules"))
        for raw in IMPORT.findall(strip_code(text)):
            target = raw.rstrip(".,;:")  # sentence punctuation after an import is not part of the path
            resolved = Path(target).expanduser() if target.startswith("~") else path.parent / target
            if not resolved.is_file():
                findings.append(Finding("error", "ST003", rel, f"@{target} does not resolve to a file"))


def check_rules(root: Path, findings: list[Finding]) -> None:
    rules = root / ".claude" / "rules"
    for path in sorted(rules.rglob("*.md")) if rules.is_dir() else []:
        meta = parse_frontmatter(read_text(path))
        if meta is None:
            continue
        rel = _rel(path, root)
        if "paths" in meta and not meta["paths"]:
            findings.append(Finding("error", "ST009", rel, "'paths' is empty; give a glob or a list of globs"))
        unknown = sorted(k for k in meta if k != "paths")
        if unknown:
            findings.append(Finding("warn", "ST009", rel,
                                    f"unknown frontmatter {unknown}; 'paths' is the only field read from a rule"))


def check_agents(root: Path, findings: list[Finding]) -> None:
    agents = root / ".claude" / "agents"
    for path in sorted(agents.rglob("*.md")) if agents.is_dir() else []:
        meta = parse_frontmatter(read_text(path)) or {}
        missing = [k for k in ("name", "description") if not meta.get(k)]
        if missing:
            findings.append(Finding("error", "ST005", _rel(path, root),
                                    f"subagent frontmatter is missing {missing}"))


def check_skills(root: Path, findings: list[Finding]) -> None:
    skills = root / ".claude" / "skills"
    for path in sorted(skills.glob("*/SKILL.md")) if skills.is_dir() else []:
        meta = parse_frontmatter(read_text(path)) or {}
        rel = _rel(path, root)
        if not meta.get("description"):
            findings.append(Finding("warn", "ST006", rel,
                                    "no description; Claude decides when to load a skill from it "
                                    "(the skills docs call it recommended)"))


def check_mcp(root: Path, findings: list[Finding]) -> None:
    path = root / ".mcp.json"
    if not path.is_file():
        return
    data = _load_json(path, root, "ST008", findings)
    if data is None:
        return
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        findings.append(Finding("error", "ST008", ".mcp.json", "needs a top-level 'mcpServers' object"))
        return
    for server, config in servers.items():
        if not isinstance(config, dict):
            findings.append(Finding("error", "ST008", ".mcp.json", f"server '{server}' must be an object"))
            continue
        texts = [config.get("url")] + (config["args"] if isinstance(config.get("args"), list) else [])
        if any(_secret_in_text(t) for t in texts):
            findings.append(Finding("error", "ST007", ".mcp.json",
                                    f"{server} has a literal secret in args or url; use ${{VAR}} expansion"))
        for block in ("env", "headers"):
            values = config.get(block, {})
            for key, value in values.items() if isinstance(values, dict) else []:
                if _looks_secret(key, value):
                    findings.append(Finding("error", "ST007", ".mcp.json",
                                            f"{server}.{block}.{key} holds a literal secret; use ${{VAR}} expansion"))


CHECKS = (check_settings, check_gitignore, check_claude_md, check_rules, check_agents, check_skills, check_mcp)


def run(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for check in CHECKS:
        check(root, findings)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", nargs="?", default=".", help="project root (default: current directory)")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    args = parser.parse_args(argv)
    root = Path(args.project).resolve()
    if not root.is_dir():
        print(f"cannot check: {root} is not a directory", file=sys.stderr)
        return 2
    try:
        findings = run(root)
    except CannotCheck as exc:
        print(f"cannot check: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"project": str(root), "findings": [asdict(f) for f in findings]}, indent=2))
    else:
        for f in findings:
            print(f"{'ERROR' if f.severity == 'error' else 'WARN '} {f.code} {f.path}: {f.message}")
        errors = sum(f.severity == "error" for f in findings)
        print(f"{errors} error(s), {len(findings) - errors} warning(s)")
    failing = [f for f in findings if f.severity == "error" or args.strict]
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
