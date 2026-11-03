#!/usr/bin/env python3
"""Resolve every checkable claim in a Markdown doc against the actual source tree.

A "claim" is anything the doc asserts about the code that can be mechanically
checked: a file path, a link, a heading anchor, an environment variable, a CLI
flag, a command, an imported symbol, or a code block that should parse.

Every claim comes back with a verdict and, when verified, a `file:line`
citation. That citation ledger is the point: it turns "the README looks fine"
into evidence a reviewer can spot-check.

Usage:
    python3 verify_claims.py README.md
    python3 verify_claims.py README.md --repo /path/to/repo
    python3 verify_claims.py docs/*.md --json > ledger.json
    python3 verify_claims.py README.md --fail-on missing,syntax-error

Exit codes:
    0  no claims at or above the --fail-on severity
    1  at least one claim at or above the --fail-on severity
    2  bad invocation (missing file, unreadable repo, unknown --fail-on value,
       or only non-Markdown files given)

--fail-on takes one or more of: info, warn, missing, syntax-error. The lowest
one listed is the threshold, so `--fail-on warn` also fails on missing and
syntax-error.

Markdown only. reStructuredText, AsciiDoc, Org, and plain-text files are
skipped with a note on stderr: an RST heading underline such as `~~~~` is a
code fence in Markdown, and reading RST as Markdown turns prose into "commands".

Stdlib only. No network calls: external URLs are reported, never fetched.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import textwrap
import unicodedata
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import unquote

# Directories that never contain the source a doc is describing. Walking them
# costs seconds and produces bogus "found it over here" evidence.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".nox", "venv", ".venv", "env",
    "dist", "build", "target", "vendor", "site-packages", ".next", ".nuxt",
    "coverage", ".gradle", "Pods", ".terraform", ".idea", ".vscode",
}

# Extensions worth indexing as "source". Keeps the grep index small enough to
# hold in memory for a normal repo.
SOURCE_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
    ".go", ".rs", ".rb", ".java", ".kt", ".swift", ".c", ".h", ".cc", ".cpp",
    ".hpp", ".cs", ".php", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".env", ".example",
    ".tf", ".sql", ".graphql", ".proto", ".dockerfile", ".mk", ".gradle",
}

# A path-looking code span needs one of these to be worth resolving, otherwise
# `foo/bar` in prose generates noise.
PATHY_EXT = SOURCE_EXT | {
    ".md", ".rst", ".txt", ".lock", ".log", ".csv", ".tsv", ".html", ".css",
    ".scss", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".xml",
}

# Environment variables the shell or CI provides. Absent from the repo by
# design, so flagging them is a false positive every time.
AMBIENT_ENV = {
    "PATH", "HOME", "PWD", "OLDPWD", "USER", "SHELL", "LANG", "LC_ALL", "TERM",
    "TMPDIR", "TMP", "TEMP", "EDITOR", "PAGER", "HOSTNAME", "PS1", "IFS",
    "CI", "GITHUB_TOKEN", "GITHUB_ACTIONS", "GITHUB_WORKSPACE", "GITHUB_SHA",
    "GITHUB_REF", "GITHUB_REPOSITORY", "RUNNER_OS", "PYTHONPATH", "NODE_ENV",
    "VIRTUAL_ENV", "CONDA_PREFIX", "DISPLAY", "SSH_AUTH_SOCK", "XDG_CONFIG_HOME",
    "AWS_REGION", "AWS_PROFILE", "DOCKER_HOST", "GOPATH", "GOROOT", "CARGO_HOME",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
}

# Uppercase code spans that are protocols, formats, or HTTP verbs. A doc that
# writes `POST` or `JSON` is not naming an environment variable.
NOT_ENV = {
    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "JSON", "YAML",
    "TOML", "HTML", "HTTP", "HTTPS", "REST", "GRPC", "UTF8", "UTF-8", "ASCII",
    "NULL", "TRUE", "FALSE", "NONE", "README", "LICENSE", "LICENCE", "TODO",
    "FIXME", "NOTE", "WARNING", "STDIN", "STDOUT", "STDERR", "EOF", "SIGINT",
    "SIGTERM", "SIGKILL", "SIGHUP", "UUID", "CSV", "TSV", "XML", "SQL", "CRUD",
}

# Shell keywords and builtins. A line that starts with one is shell control
# flow or a builtin, not a program the project has to provide.
SHELL_BUILTINS = {
    "if", "then", "else", "elif", "fi", "case", "esac", "for", "while", "until",
    "do", "done", "function", "alias", "unalias", "set", "unset", "exit",
    "return", "true", "false", "read", "eval", "exec", "shift", "trap", "wait",
    "time", "type", "local", "declare", "printf", "pushd", "popd", "test",
}

# Wrappers that run another program: `uv run widgetkit --flag` is a claim
# about `widgetkit`, so strip the wrapper and check what it runs.
RUNNERS = [("uv", "run"), ("uvx",), ("pipx", "run"), ("poetry", "run"),
           ("pdm", "run"), ("hatch", "run"), ("conda", "run"), ("npx",), ("bunx",)]

# Runner options that consume the following token as their value.
RUNNER_VALUE_OPTS = {"--python", "-p", "--with", "--from", "--project", "--directory",
                     "--package", "--extra", "--group", "--env-file", "-n", "--name",
                     "--spec", "-e", "--environment", "--index", "--index-url"}

# `python -m X` for these is a tool the reader installs, not project code.
KNOWN_TOOL_MODULES = {
    "pip", "pytest", "build", "twine", "uv", "mypy", "ruff", "black", "isort",
    "coverage", "tox", "nox", "sphinx", "mkdocs", "flake8", "pylint", "pipx",
    "virtualenv", "ipykernel", "jupyter", "pre_commit", "hatch", "poetry", "pdm",
}

# File types this script cannot parse as Markdown.
NON_MARKDOWN_SUFFIXES = {".rst", ".rest", ".txt", ".adoc", ".asciidoc", ".org"}

# A code span like `application/json` is a media type, not a path.
MIME_RE = re.compile(r"(?:application|text|image|audio|video|multipart|font|model|message)/[\w.+-]+")

# Package managers and VCS tools. Their subcommands and flags belong to them,
# not to the project, so we check that the tool exists and stop there.
FOREIGN_TOOLS = {
    "git", "docker", "docker-compose", "kubectl", "helm", "terraform", "aws",
    "gcloud", "az", "curl", "wget", "ssh", "scp", "rsync", "tar", "unzip",
    "brew", "apt", "apt-get", "yum", "dnf", "pacman", "choco", "winget",
    "nix", "nix-env", "nix-shell", "guix", "flox", "snap", "flatpak", "port",
    "pkg", "pkg_add", "pkgin", "zypper", "emerge", "apk", "xbps-install",
    "scoop", "doas", "asdf", "mise", "eget", "curlie", "just", "task", "bazel",
    "cd", "ls", "mkdir", "rm", "cp", "mv", "cat", "echo", "export", "source",
    "sudo", "chmod", "chown", "grep", "sed", "awk", "find", "which", "open",
    "node", "deno", "bun", "python", "python3", "pip", "pip3", "uv", "uvx",
    "pipx", "poetry", "pdm", "hatch", "conda", "mamba", "npm", "npx", "yarn",
    "pnpm", "cargo", "rustup", "go", "gofmt", "java", "mvn", "gradle", "dotnet",
    "ruby", "gem", "bundle", "rails", "php", "composer", "make", "cmake",
    "pytest", "ruff", "mypy", "black", "tox", "nox", "pre-commit", "twine",
    "sphinx-build", "mkdocs", "coverage", "flake8", "pylint", "isort",
    "jupyter", "ipython",
}

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)")
LINK_RE = re.compile(r"(!?)\[([^\]\n]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFDEF_RE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*<?(\S+)>?")
CODESPAN_RE = re.compile(r"`([^`\n]+)`")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ENVVAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
SHELL_LANGS = {"sh", "bash", "shell", "zsh", "console", "terminal", "shell-session", ""}
PY_LANGS = {"python", "py", "python3", "pycon", "doctest"}
JS_LANGS = {"js", "javascript", "jsx", "ts", "typescript", "tsx", "node"}

SEVERITY_ORDER = ["ok", "info", "warn", "missing", "syntax-error"]
VERDICT_SEVERITY = {
    "verified": "ok",
    "external": "info",
    "skipped": "info",
    "tool-absent": "info",
    "unresolved": "warn",
    "moved": "warn",
    "missing": "missing",
    "syntax-error": "syntax-error",
}


@dataclass
class Claim:
    kind: str           # path | link | anchor | env | flag | command | symbol | code-block
    value: str          # the literal thing the doc asserts
    line: int           # 1-indexed line in the doc
    verdict: str        # see VERDICT_SEVERITY
    evidence: str = ""  # file:line that proves it, or the reason it failed
    note: str = ""

    @property
    def severity(self) -> str:
        return VERDICT_SEVERITY.get(self.verdict, "warn")


@dataclass
class Repo:
    """An index of the source tree, built once and queried per claim."""

    root: Path
    files: list[Path] = field(default_factory=list)
    rel_set: set[str] = field(default_factory=set)
    by_basename: dict[str, list[str]] = field(default_factory=dict)
    text_index: dict[str, list[str]] = field(default_factory=dict)
    py_modules: dict[str, dict] = field(default_factory=dict)
    entrypoints: set[str] = field(default_factory=set)
    npm_scripts: set[str] = field(default_factory=set)
    make_targets: set[str] = field(default_factory=set)
    package_name: str = ""

    @classmethod
    def build(cls, root: Path) -> "Repo":
        repo = cls(root=root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".egg")]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    continue
                repo.files.append(p)
                repo.rel_set.add(rel)
                repo.by_basename.setdefault(fn, []).append(rel)
                # Index directories too, so `src/` resolves.
            for dn in dirnames:
                d = Path(dirpath) / dn
                try:
                    repo.rel_set.add(d.relative_to(root).as_posix())
                except ValueError:
                    pass
        repo._index_text()
        repo._index_python()
        repo._read_manifests()
        return repo

    def _index_text(self) -> None:
        for p in self.files:
            if p.suffix.lower() not in SOURCE_EXT and p.name not in {"Makefile", "Dockerfile", "Justfile"}:
                continue
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            self.text_index[p.relative_to(self.root).as_posix()] = lines

    def _index_python(self) -> None:
        """Map dotted module path -> {names defined at module level}."""
        for p in self.files:
            if p.suffix != ".py":
                continue
            rel = p.relative_to(self.root)
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            # Drop a leading src/ layout wrapper so `src/pkg/mod.py` is `pkg.mod`.
            if parts and parts[0] in {"src", "lib"}:
                parts = parts[1:]
            if not parts:
                continue
            dotted = ".".join(parts)
            names: set[str] = set()
            reexports: set[str] = set()
            try:
                with warnings.catch_warnings():
                    # Old string escapes in the repo's own source are not our finding.
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(self.text_from(rel.as_posix()) or "")
            except (SyntaxError, ValueError):
                self.py_modules.setdefault(dotted, {"names": names, "file": rel.as_posix(), "all": None})
                continue
            dunder_all = None
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                            if t.id == "__all__":
                                dunder_all = _literal_str_list(node.value)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        reexports.add(a.asname or a.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        reexports.add(a.asname or a.name.split(".")[0])
            self.py_modules[dotted] = {
                "names": names | reexports,
                "defined": names,
                "file": rel.as_posix(),
                "all": dunder_all,
            }

    def _read_manifests(self) -> None:
        pkg = self.root / "package.json"
        if pkg.exists():
            try:
                data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
                self.package_name = data.get("name", "") or self.package_name
                self.npm_scripts |= set((data.get("scripts") or {}).keys())
                binv = data.get("bin")
                if isinstance(binv, str):
                    self.entrypoints.add(data.get("name", "").split("/")[-1])
                elif isinstance(binv, dict):
                    self.entrypoints |= set(binv.keys())
            except (json.JSONDecodeError, OSError):
                pass
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            text = self.text_from("pyproject.toml") or ""
            # The name lives in [project] (PEP 621) or, for Poetry 1.x, [tool.poetry].
            for table in (r"project", r"tool\.poetry"):
                block = re.search(rf"^\[{table}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
                m = block and re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', block.group(1), re.M)
                if m and not self.package_name:
                    self.package_name = m.group(1)
            # [project.scripts] is the standard; [tool.poetry.scripts] is Poetry's
            # older table, still common in existing repositories.
            for table in (r"project\.scripts", r"project\.gui-scripts", r"tool\.poetry\.scripts"):
                block = re.search(rf"^\[{table}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
                if block:
                    self.entrypoints |= set(re.findall(r"^\s*[\"']?([\w.-]+)[\"']?\s*=", block.group(1), re.M))
        setup_cfg = self.text_from("setup.cfg") or ""
        if "console_scripts" in setup_cfg:
            # Only the indented continuation lines belong to console_scripts; the
            # next unindented key (dev =, test = under extras_require) does not.
            tail = setup_cfg.split("console_scripts", 1)[1].split("\n")[1:]
            for ln in tail:
                if ln.strip() and not ln[:1].isspace():
                    break
                m = re.match(r"^\s+([\w.-]+)\s*=", ln)
                if m:
                    self.entrypoints.add(m.group(1))
        cargo = self.text_from("Cargo.toml") or ""
        if cargo:
            m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', cargo, re.M)
            if m:
                self.entrypoints.add(m.group(1))
                if not self.package_name:
                    self.package_name = m.group(1)
        for mk in ("Makefile", "makefile", "GNUmakefile"):
            text = self.text_from(mk)
            if text:
                self.make_targets |= set(re.findall(r"^([A-Za-z0-9_.-]+):(?!=)", text, re.M))
                break

    def text_from(self, rel: str) -> str | None:
        lines = self.text_index.get(rel)
        if lines is not None:
            return "\n".join(lines)
        p = self.root / rel
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def grep(self, needle: str, limit: int = 1) -> list[str]:
        """Return up to `limit` `file:line` citations containing `needle`."""
        hits: list[str] = []
        for rel, lines in self.text_index.items():
            for i, line in enumerate(lines, 1):
                if needle in line:
                    hits.append(f"{rel}:{i}")
                    if len(hits) >= limit:
                        return hits
        return hits

    def grep_multiline(self, pattern: str, limit: int = 1) -> list[str]:
        """Like grep_re, but the pattern may span lines (export blocks, decorators)."""
        rx = re.compile(pattern, re.S)
        hits: list[str] = []
        for rel, lines in self.text_index.items():
            text = "\n".join(lines)
            m = rx.search(text)
            if m:
                hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}")
                if len(hits) >= limit:
                    return hits
        return hits

    def grep_re(self, pattern: str, limit: int = 1) -> list[str]:
        """Same as grep, but the needle is a regular expression."""
        rx = re.compile(pattern)
        hits: list[str] = []
        for rel, lines in self.text_index.items():
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}")
                    if len(hits) >= limit:
                        return hits
        return hits


def _literal_str_list(node: ast.AST) -> list[str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def frontmatter_end(text: str) -> int:
    """Line index just past a leading YAML frontmatter block, or 0 if absent."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return 0
    for i, line in enumerate(lines[1:], 1):
        if line.strip() in ("---", "..."):
            return i + 1
    return 0


def blank_frontmatter(text: str) -> str:
    """Replace frontmatter with blank lines so it is not linted as prose.

    Blanking rather than removing keeps every later line at its original number,
    which is what makes a finding's location usable.
    """
    end = frontmatter_end(text)
    if not end:
        return text
    lines = text.splitlines()
    return "\n".join([""] * end + lines[end:])


def normalize_setext(text: str) -> str:
    """Rewrite setext headings as ATX so one heading regex handles both.

    `Title` followed by `=====` is an h1 and `-----` is an h2. Repositories
    still use this style, and a linter that only knows `#` sees such a document
    as having no headings at all.
    """
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    marker = ""
    start = frontmatter_end(text)
    if start:
        out.extend(lines[:start])
    for i, line in enumerate(lines[start:], start):
        fm = FENCE_RE.match(line)
        if fm and not in_fence:
            in_fence, marker = True, fm.group(2)[0] * 3
        elif in_fence and line.strip().startswith(marker):
            in_fence = False
        if (not in_fence and out and re.fullmatch(r"\s*(=+|-+)\s*", line)
                and out[-1].strip()
                and not out[-1].lstrip().startswith(("#", "-", "*", "|", ">", "="))
                and not FENCE_RE.match(out[-1])
                and i > 0):
            level = "#" if line.strip()[0] == "=" else "##"
            out[-1] = f"{level} {out[-1].strip()}"
            # Blank rather than removed, so every later line keeps its number.
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def slugify(heading: str) -> str:
    """GitHub's heading-anchor algorithm (github-slugger 2.x).

    GitHub lowercases the rendered heading text, deletes every character that
    is not a letter, mark, number, connector punctuation (`_`), hyphen, or
    space, then turns each space into a hyphen. It does not collapse runs:
    "Install & configure" is `install--configure`, and "my_var option" keeps
    its underscore as `my_var-option`.
    """
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = LINK_RE.sub(r"\2", text)
    # Emphasis markers vanish when the heading renders. An underscore inside a
    # word (snake_case) is literal text in GFM, so only strip delimiter runs.
    text = re.sub(r"[*~]+", "", text)
    text = re.sub(r"(?<![^\W_])_+|_+(?![^\W_])", "", text)
    text = text.strip().lower()
    kept = "".join(
        ch for ch in text
        if ch in "- " or unicodedata.category(ch)[0] in "LMN" or unicodedata.category(ch) == "Pc"
    )
    return kept.replace(" ", "-")


def heading_slugs(markdown: str) -> set[str]:
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    # Explicit anchors: <a name="x">, <h3 id="x">, and the {#x} attribute syntax.
    for m in re.finditer(r'(?:\bname|\bid)\s*=\s*["\']([^"\']+)["\']', markdown):
        slugs.add(m.group(1).lower())
    for m in re.finditer(r"\{#([\w-]+)\}", markdown):
        slugs.add(m.group(1).lower())
    markdown = normalize_setext(markdown)
    in_fence = False
    fence_marker = ""
    for line in markdown.splitlines():
        fm = FENCE_RE.match(line)
        if fm:
            marker = fm.group(2)
            if not in_fence:
                in_fence, fence_marker = True, marker[0] * 3
            elif line.strip().startswith(fence_marker):
                in_fence = False
            continue
        if in_fence:
            continue
        hm = HEADING_RE.match(line)
        if hm:
            base = slugify(hm.group(2))
            n = seen.get(base, 0)
            seen[base] = n + 1
            slugs.add(base if n == 0 else f"{base}-{n}")
    return slugs


@dataclass
class Block:
    lang: str
    line: int
    body: str


def split_doc(text: str) -> tuple[list[tuple[int, str]], list[Block]]:
    """Return (prose lines as (lineno, text), fenced code blocks)."""
    prose: list[tuple[int, str]] = []
    blocks: list[Block] = []
    in_fence = False
    marker = ""
    lang = ""
    start = 0
    buf: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        fm = FENCE_RE.match(line)
        if fm and not in_fence:
            in_fence = True
            marker = fm.group(2)[0] * len(fm.group(2))
            lang = fm.group(3).lower().split(",")[0].strip()
            start = i
            buf = []
            continue
        if in_fence:
            if line.strip().startswith(marker[0] * 3) and set(line.strip()) <= {marker[0]}:
                blocks.append(Block(lang=lang, line=start, body="\n".join(buf)))
                in_fence = False
                continue
            buf.append(line)
            continue
        prose.append((i, line))
    if in_fence:
        blocks.append(Block(lang=lang, line=start, body="\n".join(buf)))
    return prose, blocks


def resolve_path(repo: Repo, doc_dir: Path, target: str) -> tuple[str, str]:
    """Return (verdict, evidence) for a filesystem path claim."""
    cleaned = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not cleaned:
        return "skipped", "fragment only"
    candidates = []
    if cleaned.startswith("/"):
        candidates.append(repo.root / cleaned.lstrip("/"))
    else:
        candidates.append((doc_dir / cleaned))
        candidates.append(repo.root / cleaned)
    for c in candidates:
        try:
            resolved = c.resolve()
            rel = resolved.relative_to(repo.root.resolve()).as_posix()
        except (ValueError, OSError):
            if c.exists():
                return "verified", str(c)
            continue
        if resolved.exists():
            return "verified", rel or "."
    base = Path(cleaned).name
    others = repo.by_basename.get(base, [])
    if others:
        return "moved", f"not at {cleaned}; found {others[0]}"
    return "missing", f"no such path under {repo.root.name}/"


def check_links(repo: Repo, doc: Path, text: str, prose: list[tuple[int, str]]) -> list[Claim]:
    claims: list[Claim] = []
    own_slugs = heading_slugs(text)
    refdefs = {}
    for lineno, line in prose:
        m = REFDEF_RE.match(line)
        if m:
            refdefs[m.group(1).lower()] = (lineno, m.group(2))
    targets: list[tuple[int, str, str]] = []
    for lineno, line in prose:
        for m in LINK_RE.finditer(line):
            targets.append((lineno, m.group(3), m.group(2)))
    for lineno, (_, url) in ((v[0], (k, v[1])) for k, v in refdefs.items()):
        targets.append((lineno, url, ""))

    for lineno, target, label in targets:
        if target.startswith(("http://", "https://")):
            claims.append(Claim("link", target, lineno, "external", "not fetched (no network calls)"))
            continue
        if target.startswith(("mailto:", "tel:", "data:")):
            claims.append(Claim("link", target, lineno, "skipped", "non-filesystem scheme"))
            continue
        if target.startswith("#"):
            frag = unquote(target[1:]).lower()
            if frag in own_slugs:
                claims.append(Claim("anchor", target, lineno, "verified", f"{doc.name} heading"))
            else:
                claims.append(Claim("anchor", target, lineno, "missing", "no heading with that slug in this doc"))
            continue
        verdict, evidence = resolve_path(repo, doc.parent, target)
        claims.append(Claim("link", target, lineno, verdict, evidence))
        if verdict == "verified" and "#" in target:
            frag = unquote(target.split("#", 1)[1]).lower()
            path = repo.root / evidence
            if path.suffix.lower() in {".md", ".markdown"}:
                slugs = heading_slugs(path.read_text(encoding="utf-8", errors="replace"))
                ok = frag in slugs
                claims.append(Claim(
                    "anchor", f"{target}", lineno,
                    "verified" if ok else "missing",
                    f"{evidence} headings" if ok else f"{evidence} has no heading with that slug",
                ))
    return claims


def check_code_spans(repo: Repo, doc: Path, prose: list[tuple[int, str]]) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    for lineno, line in prose:
        for m in CODESPAN_RE.finditer(line):
            span = m.group(1).strip()
            if not span or " " in span or len(span) > 200:
                continue
            if span.startswith(("http://", "https://", "$", "-")):
                continue
            looks_pathy = "/" in span or Path(span).suffix.lower() in PATHY_EXT
            if not looks_pathy or not re.fullmatch(r"[\w./*@~-]+", span):
                continue
            if span.count("/") and span.split("/")[0] in {"", "..", "~"}:
                continue
            if MIME_RE.fullmatch(span):
                continue
            if ("path", span) in seen:
                continue
            seen.add(("path", span))
            if "*" in span:
                claims.append(Claim("path", span, lineno, "skipped", "glob pattern"))
                continue
            verdict, evidence = resolve_path(repo, doc.parent, span)
            claims.append(Claim("path", span, lineno, verdict, evidence))
    return claims


def check_env_vars(repo: Repo, prose: list[tuple[int, str]], blocks: list[Block]) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[str] = set()

    def consider(name: str, lineno: int) -> None:
        if name in AMBIENT_ENV or name in NOT_ENV or name in seen or len(name) < 4:
            return
        seen.add(name)
        # Whole-word match: WIDGETKIT_TIME must not verify against WIDGETKIT_TIMEOUT.
        hits = repo.grep_re(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        if hits:
            claims.append(Claim("env", name, lineno, "verified", hits[0]))
        else:
            claims.append(Claim("env", name, lineno, "missing", "name never appears in source"))

    for lineno, line in prose:
        for m in CODESPAN_RE.finditer(line):
            span = m.group(1).strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", span):
                consider(span, lineno)
    for b in blocks:
        if b.lang not in SHELL_LANGS and b.lang not in {"env", "dotenv", "ini", "properties"}:
            continue
        for offset, line in enumerate(b.body.splitlines(), 1):
            for m in re.finditer(r"(?:^|\s|\$\{?)([A-Z][A-Z0-9_]{3,})(?=[=}\s]|$)", line):
                consider(m.group(1), b.line + offset)
    return claims


def check_commands(repo: Repo, blocks: list[Block]) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[str] = set()
    for b in blocks:
        if b.lang not in SHELL_LANGS:
            continue
        raw_lines = b.body.splitlines()
        # A block that shows a prompt is a transcript: the unprompted lines are
        # program output, not commands. Checking them produces pure noise.
        prompted = any(re.match(r"^\s*[$>#]\s+\S", ln) for ln in raw_lines)
        for offset, raw in enumerate(raw_lines, 1):
            line = raw.strip()
            lineno = b.line + offset
            if not line:
                continue
            if prompted:
                if not re.match(r"^[$>]\s+\S", line):
                    continue
                line = re.sub(r"^[$>]\s+", "", line)
            elif line.startswith("#"):
                continue
            # A command starts with a bare program name. Anything else in a
            # fence -- "Your name: Click", "3 passed in 0.2s" -- is output.
            if not re.match(r"^[a-zA-Z_][\w./-]*(\s|$)", line):
                continue
            tokens = line.split()
            if not tokens:
                continue
            # `uv run widgetkit --flag` is a claim about widgetkit, not uv.
            for runner in RUNNERS:
                if tuple(tokens[:len(runner)]) == runner and len(tokens) > len(runner):
                    rest = tokens[len(runner):]
                    # Skip the runner's own options. `--opt=value` is one token;
                    # `--python 3.12` and `--with pkg` take the next token too.
                    while rest and rest[0].startswith("-"):
                        takes_value = "=" not in rest[0] and rest[0] in RUNNER_VALUE_OPTS
                        rest = rest[2:] if takes_value else rest[1:]
                    if rest and re.fullmatch(r"[a-zA-Z_][\w./-]*", rest[0]):
                        tokens = rest
                        line = " ".join(rest)
                    break
            prog = tokens[0]
            if "=" in prog and not prog.startswith("-"):
                continue
            if prog in SHELL_BUILTINS:
                continue
            # Commands are lowercase by convention, and a colon in the first
            # two tokens means the line is labelled output ("Your name: Click").
            if prog != prog.lower() or any(t.endswith(":") for t in tokens[:2]):
                continue
            key = " ".join(tokens[:3])
            if key in seen:
                continue
            seen.add(key)

            if prog in {"npm", "pnpm", "yarn"} and len(tokens) >= 2:
                sub = tokens[2] if (tokens[1] == "run" and len(tokens) > 2) else tokens[1]
                if tokens[1] in {"install", "i", "add", "ci", "publish", "init", "exec", "create"}:
                    claims.append(Claim("command", key, lineno, "skipped", "package registry operation"))
                elif sub in repo.npm_scripts:
                    claims.append(Claim("command", key, lineno, "verified", f"package.json scripts.{sub}"))
                else:
                    claims.append(Claim("command", key, lineno, "missing",
                                        f"no scripts.{sub} in package.json"))
                continue

            if prog == "make":
                target = next((t for t in tokens[1:] if not t.startswith("-")), "")
                if not target:
                    claims.append(Claim("command", key, lineno, "skipped", "default make target"))
                elif target in repo.make_targets:
                    claims.append(Claim("command", key, lineno, "verified", f"Makefile target {target}"))
                else:
                    claims.append(Claim("command", key, lineno, "missing", f"no `{target}:` target in Makefile"))
                continue

            if prog in {"python", "python3", "py"} and len(tokens) >= 3 and tokens[1] == "-m":
                mod = tokens[2]
                top = mod.split(".")[0]
                info = repo.py_modules.get(mod)
                if info is not None:
                    # `python -m pkg` runs pkg/__main__.py, not pkg/__init__.py.
                    if info["file"].endswith("__init__.py"):
                        main_mod = repo.py_modules.get(mod + ".__main__")
                        if main_mod:
                            claims.append(Claim("command", key, lineno, "verified", main_mod["file"]))
                        else:
                            claims.append(Claim("command", key, lineno, "missing",
                                                f"{info['file']} is a package with no __main__.py; "
                                                f"`python -m {mod}` fails"))
                    else:
                        claims.append(Claim("command", key, lineno, "verified", info["file"]))
                elif top in KNOWN_TOOL_MODULES or top in getattr(sys, "stdlib_module_names", ()):
                    claims.append(Claim("command", key, lineno, "skipped",
                                        "standard-library or third-party tool module"))
                elif top in repo.py_modules or any(k.startswith(top + ".") for k in repo.py_modules):
                    claims.append(Claim("command", key, lineno, "missing", f"module {mod} not in this repo"))
                else:
                    claims.append(Claim("command", key, lineno, "unresolved",
                                        f"module {mod} is not in this repo; confirm it is a tool the reader installs"))
                continue

            if prog in repo.entrypoints or prog == repo.package_name:
                claims.append(Claim("command", key, lineno, "verified", "declared entrypoint"))
                for flag in dict.fromkeys(FLAG_RE.findall(line)):
                    # Whole-token match: `--dry` must not verify against `--dry-run`.
                    hits = repo.grep_re(rf"(?<![\w-]){re.escape(flag)}(?![\w-])")
                    claims.append(Claim(
                        "flag", flag, lineno,
                        "verified" if hits else "missing",
                        hits[0] if hits else "flag string never appears in source",
                    ))
                continue

            if prog in {"python", "python3", "node", "bash", "sh", "ruby"} and len(tokens) >= 2 \
                    and not tokens[1].startswith("-"):
                script = tokens[1]
                verdict, evidence = resolve_path(repo, repo.root, script)
                if verdict == "missing":
                    # A tutorial often tells the reader to create this file. That
                    # is legitimate, but only if the doc says so -- hence warn.
                    verdict = "unresolved"
                    evidence = ("not in the repo; if the reader is meant to create it, "
                                "the doc must say so before this command")
                claims.append(Claim("path", script, lineno, verdict, evidence,
                                    note="script named in a shell example"))
                continue

            if prog in FOREIGN_TOOLS:
                claims.append(Claim("command", key, lineno, "skipped", "third-party tool"))
                continue

            local = repo.by_basename.get(prog) or repo.by_basename.get(Path(prog).name)
            if local:
                claims.append(Claim("command", key, lineno, "verified", local[0]))
            elif shutil.which(prog):
                claims.append(Claim("command", key, lineno, "tool-absent",
                                    "resolves on this machine but is not declared by the project"))
            else:
                # It may be a package manager or an OS tool this script has
                # never heard of, so this is a prompt to look, not a failure.
                claims.append(Claim("command", key, lineno, "unresolved",
                                    "not a project entrypoint, npm script, make target, or repo "
                                    "file; confirm it is a third-party tool the reader has"))
    return claims


def check_python_blocks(repo: Repo, blocks: list[Block]) -> list[Claim]:
    claims: list[Claim] = []
    for b in blocks:
        if b.lang not in PY_LANGS:
            continue
        source = b.body
        if ">>>" in source:
            source = "\n".join(
                ln[4:] for ln in source.splitlines()
                if ln.startswith((">>> ", "... "))
            )
            if not source.strip():
                continue
        # A snippet indented as a whole, or a decorator shown on its own, is a
        # fragment of valid code rather than a syntax error.
        source = textwrap.dedent(source)
        if re.search(r"^@[\w.]+(?:\(.*\))?\s*\Z", source.rstrip().splitlines()[-1] if source.strip() else "", re.S):
            source += "\ndef _fragment(): pass\n"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except SyntaxError as exc:
            claims.append(Claim("code-block", f"python block at line {b.line}", b.line,
                                "syntax-error", f"{exc.msg} (block line {exc.lineno})"))
            continue

        imported: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    imported[a.asname or root] = a.name
                    if root in repo.py_modules or any(k.startswith(root + ".") for k in repo.py_modules):
                        claims.append(Claim("symbol", a.name, b.line, "verified",
                                            repo.py_modules.get(a.name, repo.py_modules.get(root, {})).get("file", root)))
                    else:
                        claims.append(Claim("symbol", a.name, b.line, "external",
                                            "third-party or stdlib import"))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mod = node.module
                info = repo.py_modules.get(mod)
                if info is None:
                    claims.append(Claim("symbol", f"from {mod}", b.line, "external",
                                        "third-party or stdlib import"))
                    continue
                exported = set(info["names"])
                if info.get("all"):
                    exported |= set(info["all"])
                for a in node.names:
                    if a.name == "*":
                        continue
                    if a.name in exported:
                        claims.append(Claim("symbol", f"{mod}.{a.name}", b.line, "verified", info["file"]))
                    else:
                        hits = repo.grep(f"def {a.name}") or repo.grep(f"class {a.name}")
                        if hits:
                            claims.append(Claim("symbol", f"{mod}.{a.name}", b.line, "moved",
                                                f"not exported by {info['file']}; defined at {hits[0]}"))
                        else:
                            claims.append(Claim("symbol", f"{mod}.{a.name}", b.line, "missing",
                                                f"not defined in {info['file']}"))
    return claims


def check_json_blocks(blocks: list[Block]) -> list[Claim]:
    claims: list[Claim] = []
    for b in blocks:
        if b.lang not in {"json", "jsonc"} or not b.body.strip():
            continue
        if b.lang == "jsonc" or "//" in b.body:
            continue
        try:
            json.loads(b.body)
            claims.append(Claim("code-block", f"json block at line {b.line}", b.line, "verified", "parses"))
        except json.JSONDecodeError as exc:
            claims.append(Claim("code-block", f"json block at line {b.line}", b.line,
                                "syntax-error", f"{exc.msg} at block line {exc.lineno}"))
    return claims


def check_js_blocks(repo: Repo, blocks: list[Block]) -> list[Claim]:
    claims: list[Claim] = []
    own = repo.package_name
    for b in blocks:
        if b.lang not in JS_LANGS:
            continue
        for m in re.finditer(
            r"""(?:import\s+(?:(\{[^}]*\})|[\w*\s,]+)\s+from\s*|require\(\s*)['"]([^'"]+)['"]""",
            b.body,
        ):
            names_blob, mod = m.group(1), m.group(2)
            offset = b.body[: m.start()].count("\n") + 1
            lineno = b.line + offset
            is_own = bool(own) and (mod == own or mod.startswith(own + "/"))
            if not (mod.startswith(".") or is_own):
                claims.append(Claim("symbol", mod, lineno, "external", "third-party module"))
                continue
            if not names_blob:
                claims.append(Claim("symbol", mod, lineno, "skipped", "default import; target not resolved"))
                continue
            for name in [n.strip().split(" as ")[0].strip() for n in names_blob.strip("{}").split(",")]:
                if not name or not re.fullmatch(r"\w+", name):
                    continue
                esc = re.escape(name)
                hits = (
                    # export function|const|let|class|type|interface|enum NAME
                    repo.grep_re(rf"\bexport\s+(?:default\s+)?(?:async\s+)?"
                                 rf"(?:function\*?|const|let|var|class|type|interface|enum)\s+{esc}\b")
                    # export {NAME}, export {x, NAME as y} -- these blocks wrap
                    # across lines in real source, so match on file text.
                    or repo.grep_multiline(rf"\bexport\s*\{{[^}}]*\b{esc}\b[^}}]*\}}")
                    # CommonJS
                    or repo.grep_re(rf"\bexports\.{esc}\s*=")
                    or repo.grep_re(rf"\bmodule\.exports\s*=\s*\{{[^}}]*\b{esc}\b")
                )
                if hits:
                    claims.append(Claim("symbol", f"{mod}:{name}", lineno, "verified", hits[0]))
                else:
                    claims.append(Claim("symbol", f"{mod}:{name}", lineno, "missing",
                                        "no matching export found in source"))
    return claims


def verify(doc: Path, repo: Repo) -> list[Claim]:
    text = doc.read_text(encoding="utf-8", errors="replace")
    prose, blocks = split_doc(text)
    claims = []
    claims += check_links(repo, doc, text, prose)
    claims += check_code_spans(repo, doc, prose)
    claims += check_env_vars(repo, prose, blocks)
    claims += check_commands(repo, blocks)
    claims += check_python_blocks(repo, blocks)
    claims += check_js_blocks(repo, blocks)
    claims += check_json_blocks(blocks)
    return sorted(claims, key=lambda c: (c.line, c.kind, c.value))


def find_repo_root(start: Path) -> Path:
    cur = start if start.is_dir() else start.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
        if any((candidate / m).exists() for m in
               ("package.json", "pyproject.toml", "Cargo.toml", "go.mod", "setup.py")):
            return candidate
    return cur


def render(doc: Path, claims: list[Claim], show_all: bool) -> str:
    lines = [f"# Claim ledger: {doc}", ""]
    counts: dict[str, int] = {}
    for c in claims:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1
    shown = [c for c in claims if show_all or c.severity not in ("ok", "info")]
    if not shown:
        lines.append("No unresolved claims. Everything checkable resolved to source.")
    else:
        width = max((len(c.kind) for c in shown), default=4)
        for c in shown:
            mark = {"ok": "OK  ", "info": "--  ", "warn": "WARN", "missing": "FAIL", "syntax-error": "FAIL"}[c.severity]
            lines.append(f"{mark} L{c.line:<4} {c.kind:<{width}}  {c.value}")
            if c.evidence:
                lines.append(f"              -> {c.evidence}")
    lines.append("")
    total = len(claims)
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    lines.append(f"{total} claims checked: {summary or 'none'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", nargs="+", help="Markdown file(s) to check")
    ap.add_argument("--repo", help="Repository root (default: nearest ancestor with .git or a manifest)")
    ap.add_argument("--json", action="store_true", help="Emit the ledger as JSON")
    ap.add_argument("--show-all", "--show-ok", dest="show_all", action="store_true",
                    help="Include verified, external, and skipped claims in text output")
    ap.add_argument("--fail-on", default="missing,syntax-error",
                    help="Severities that cause exit 1: info, warn, missing, syntax-error. "
                         "The lowest one listed is the threshold (default: missing,syntax-error)")
    args = ap.parse_args(argv)

    requested = [s.strip() for s in args.fail_on.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SEVERITY_ORDER[1:]]
    if unknown or not requested:
        print(f"error: --fail-on accepts info, warn, missing, syntax-error; got {args.fail_on!r}",
              file=sys.stderr)
        return 2
    floor = min(SEVERITY_ORDER.index(s) for s in requested)
    fail_on = set(SEVERITY_ORDER[floor:])

    doc_paths = [Path(d) for d in args.docs]
    for d in doc_paths:
        if not d.is_file():
            print(f"error: {d} is not a readable file", file=sys.stderr)
            return 2
    skipped = [d for d in doc_paths if d.suffix.lower() in NON_MARKDOWN_SUFFIXES]
    for d in skipped:
        print(f"note: skipped {d}: verify_claims.py reads Markdown only", file=sys.stderr)
    doc_paths = [d for d in doc_paths if d not in skipped]
    if not doc_paths:
        print("error: no Markdown files to check", file=sys.stderr)
        return 2

    root = Path(args.repo).resolve() if args.repo else find_repo_root(doc_paths[0].resolve())
    if not root.is_dir():
        print(f"error: repo root {root} is not a directory", file=sys.stderr)
        return 2

    repo = Repo.build(root)
    payload = []
    worst = False
    for d in doc_paths:
        claims = verify(d, repo)
        if any(c.severity in fail_on for c in claims):
            worst = True
        if args.json:
            payload.append({"doc": str(d), "repo": str(root),
                            "claims": [asdict(c) | {"severity": c.severity} for c in claims]})
        else:
            print(render(d, claims, args.show_all))
            print()
    if args.json:
        print(json.dumps(payload, indent=2))
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
