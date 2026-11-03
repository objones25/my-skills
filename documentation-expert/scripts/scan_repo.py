#!/usr/bin/env python3
"""Survey a repository before writing or auditing any of its documentation.

Answers the questions that decide what to write and what not to write:
what the project actually is, which docs already exist, whether a generator
already owns the API reference, what the public surface is, and how far the
docs have fallen behind the code.

The most useful output is usually the last section: public names the source
exports that no document mentions, and documented commands that no longer
exist. That is the gap list. Everything above it is context for reading it.

Usage:
    python3 scan_repo.py                 # scan the current directory
    python3 scan_repo.py /path/to/repo
    python3 scan_repo.py --json

Exit codes:
    0  scan completed
    2  path is not a readable directory

Stdlib only. Reads git metadata by shelling out to `git`; degrades cleanly
when git is absent or the directory is not a repository.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Importing the sibling module would otherwise write scripts/__pycache__/,
# and these scripts promise to write nothing to disk.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_claims import Repo, SKIP_DIRS  # noqa: E402

DOC_NAMES = [
    "README.md", "README.rst", "README.txt", "README",
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "SUPPORT.md",
    "CHANGELOG.md", "CHANGES.md", "HISTORY.md", "AGENTS.md", "CLAUDE.md",
    "ARCHITECTURE.md", "INSTALL.md", "USAGE.md", "LICENSE", "LICENSE.md",
    "LICENSE.txt", "llms.txt",
]

# Marker file -> what owns the reference docs if it is present.
GENERATOR_MARKERS = {
    "docs/conf.py": "Sphinx (autodoc): improve docstrings, do not write parallel reference pages",
    "doc/conf.py": "Sphinx (autodoc): improve docstrings, do not write parallel reference pages",
    "mkdocs.yml": "MkDocs: add pages to the configured nav, do not create an orphan docs tree",
    "mkdocs.yaml": "MkDocs: add pages to the configured nav, do not create an orphan docs tree",
    "docusaurus.config.js": "Docusaurus: docs live under the configured docs dir with sidebar entries",
    "docusaurus.config.ts": "Docusaurus: docs live under the configured docs dir with sidebar entries",
    "typedoc.json": "TypeDoc: reference comes from TSDoc comments in source",
    "typedoc.config.js": "TypeDoc: reference comes from TSDoc comments in source",
    ".readthedocs.yaml": "Read the Docs build: docs are published from this config",
    ".readthedocs.yml": "Read the Docs build: docs are published from this config",
    "openapi.yaml": "OpenAPI spec: endpoint reference is generated from it",
    "openapi.json": "OpenAPI spec: endpoint reference is generated from it",
    "docs/openapi.yaml": "OpenAPI spec: endpoint reference is generated from it",
    "book.toml": "mdBook: docs live under the configured src dir",
    "Doxyfile": "Doxygen: reference comes from source comments",
    ".vitepress/config.js": "VitePress: docs live under the configured srcDir",
}

CLI_HINTS = {
    "argparse": r"\bargparse\.ArgumentParser\b",
    "click": r"@click\.(?:command|group)\b",
    "typer": r"\btyper\.Typer\(",
    "commander": r"\brequire\(['\"]commander['\"]\)|from\s+['\"]commander['\"]",
    "yargs": r"\brequire\(['\"]yargs['\"]\)|from\s+['\"]yargs['\"]",
    "clap": r"#\[derive\([^)]*Parser",
    "cobra": r"cobra\.Command\{",
}


def run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def last_commit_epoch(root: Path, rel: str) -> int | None:
    out = run_git(root, "log", "-1", "--format=%ct", "--", rel)
    return int(out) if out and out.isdigit() else None


def last_commit_sha(root: Path, rel: str) -> str | None:
    return run_git(root, "log", "-1", "--format=%H", "--", rel) or None


def source_paths(repo: Repo) -> list[str]:
    """Where the code lives. Flat layouts (pkg/ at the root) have no src/."""
    conventional = [d for d in ("src", "lib", "source", "app", "pkg") if d in repo.rel_set]
    if conventional:
        return conventional
    # Top-level Python packages, Go/Rust/JS roots: any first-level directory
    # holding source files, minus docs and tests.
    skip = {"docs", "doc", "documentation", "tests", "test", "spec", "examples",
            "example", "benchmarks", "scripts", "extras", ".github", "site", "website"}
    heads = sorted({rel.split("/")[0] for rel in repo.text_index
                    if "/" in rel and rel.split("/")[0] not in skip
                    and Path(rel).suffix in {".py", ".go", ".rs", ".ts", ".js", ".rb", ".java"}})
    return heads


def find_docs(repo: Repo) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    # Repositories spell these every way: README.md, readme.md, Readme.md.
    lower_index: dict[str, str] = {}
    for rel in repo.rel_set:
        lower_index.setdefault(rel.lower(), rel)
    for name in DOC_NAMES:
        # GitHub's own precedence: .github/, then the root, then docs/.
        for prefix in (".github/", "", "docs/"):
            actual = lower_index.get(f"{prefix}{name}".lower())
            if actual and actual not in seen:
                seen.add(actual)
                docs.append({"path": actual, "role": "root doc"})
    for rel in sorted(repo.rel_set):
        if rel in seen or not rel.lower().endswith((".md", ".rst", ".mdx")):
            continue
        if rel.split("/")[0] in {"docs", "doc", "documentation", "website", "site"}:
            docs.append({"path": rel, "role": "docs tree"})
            seen.add(rel)
    for d in docs:
        p = repo.root / d["path"]
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        d["bytes"] = len(text.encode("utf-8"))
        d["lines"] = text.count("\n") + 1 if text else 0
        d["headings"] = re.findall(r"^#{1,3}\s+(.+?)\s*#*$", text, re.M)[:25]
        d["code_blocks"] = len(re.findall(r"^\s*(?:```|~~~)", text, re.M)) // 2
    return docs


def detect_generators(repo: Repo) -> list[dict]:
    found = []
    for marker, meaning in GENERATOR_MARKERS.items():
        if marker in repo.rel_set:
            found.append({"marker": marker, "implication": meaning})
    # Rust and Go generate reference from source comments with no marker file.
    if "Cargo.toml" in repo.rel_set:
        found.append({"marker": "Cargo.toml",
                      "implication": "rustdoc: reference comes from /// comments in source"})
    if "go.mod" in repo.rel_set:
        found.append({"marker": "go.mod",
                      "implication": "pkg.go.dev: reference comes from doc comments in source"})
    return found


def python_surface(repo: Repo) -> list[dict]:
    """Public names a Python package exports, preferring __all__."""
    surface: list[dict] = []
    packages = sorted(
        (m for m, info in repo.py_modules.items()
         if info["file"].endswith("__init__.py") and "." not in m),
        key=len,
    )
    for pkg in packages:
        info = repo.py_modules[pkg]
        names = info.get("all")
        if names is None:
            names = sorted(n for n in info.get("defined", set()) if not n.startswith("_"))
            if not names:
                names = sorted(n for n in info["names"] if not n.startswith("_"))
        for n in names:
            surface.append({"name": n, "module": pkg, "file": info["file"],
                            "declared_in_all": info.get("all") is not None})
    return surface


def js_surface(repo: Repo) -> list[dict]:
    entry = None
    pkg = repo.root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return []
        exports = data.get("exports")
        if isinstance(exports, str):
            entry = exports
        elif isinstance(exports, dict):
            root_export = exports.get(".", exports)
            entry = root_export if isinstance(root_export, str) else (
                root_export.get("import") or root_export.get("default")
                or root_export.get("require") if isinstance(root_export, dict) else None
            )
        entry = entry or data.get("main") or data.get("module")
    if not entry:
        return []
    rel = entry.lstrip("./")
    text = repo.text_from(rel) or ""
    names = set(re.findall(
        r"\bexport\s+(?:default\s+)?(?:async\s+)?"
        r"(?:function\*?|const|let|var|class|type|interface|enum)\s+(\w+)", text))
    for block in re.findall(r"\bexport\s*\{([^}]*)\}", text, re.S):
        for piece in block.split(","):
            piece = piece.strip()
            if not piece:
                continue
            name = piece.split(" as ")[-1].strip() if " as " in piece else piece
            if re.fullmatch(r"\w+", name):
                names.add(name)
    return [{"name": n, "module": rel, "file": rel, "declared_in_all": True}
            for n in sorted(names)]


def detect_cli(repo: Repo) -> list[str]:
    found = []
    for label, pattern in CLI_HINTS.items():
        hits = repo.grep_re(pattern)
        if hits:
            found.append(f"{label} ({hits[0]})")
    return found


def source_env_vars(repo: Repo) -> dict[str, str]:
    """Environment variables the source reads. These are config a user must know."""
    out: dict[str, str] = {}
    patterns = [
        r"""os\.environ(?:\.get)?\[?\(?['"]([A-Z][A-Z0-9_]{2,})['"]""",
        r"""os\.getenv\(\s*['"]([A-Z][A-Z0-9_]{2,})['"]""",
        r"""process\.env\.([A-Z][A-Z0-9_]{2,})""",
        r"""process\.env\[['"]([A-Z][A-Z0-9_]{2,})['"]\]""",
        r'env::var\(\s*"([A-Z][A-Z0-9_]{2,})"',
        r'os\.Getenv\(\s*"([A-Z][A-Z0-9_]{2,})"',
        r'ENV\[[\'"]([A-Z][A-Z0-9_]{2,})[\'"]\]',
    ]
    compiled = [re.compile(p) for p in patterns]
    # Test fixtures set variables the real program never reads, so a variable
    # found only under a test directory is not project configuration.
    test_roots = {"tests", "test", "spec", "__tests__", "e2e", "fixtures", "benchmarks"}
    for rel, lines in repo.text_index.items():
        head = rel.split("/")[0]
        if head in SKIP_DIRS or head in test_roots:
            continue
        for i, line in enumerate(lines, 1):
            for rx in compiled:
                for m in rx.finditer(line):
                    out.setdefault(m.group(1), f"{rel}:{i}")
    return out


def doc_corpus(repo: Repo, docs: list[dict]) -> str:
    return "\n".join((repo.root / d["path"]).read_text(encoding="utf-8", errors="replace")
                     for d in docs if (repo.root / d["path"]).exists())


def scan(root: Path) -> dict:
    repo = Repo.build(root)
    docs = find_docs(repo)
    corpus = doc_corpus(repo, docs)
    surface = python_surface(repo) or js_surface(repo)
    envs = source_env_vars(repo)

    undocumented_names = [
        s for s in surface
        if not re.search(rf"(?<![\w.]){re.escape(s['name'])}(?![\w])", corpus)
    ]
    undocumented_envs = {k: v for k, v in envs.items()
                         if not re.search(rf"\b{re.escape(k)}\b", corpus)}
    undocumented_scripts = sorted(
        s for s in repo.npm_scripts
        if not re.search(rf"\b{re.escape(s)}\b", corpus)
    )
    undocumented_entrypoints = sorted(
        e for e in repo.entrypoints if e and not re.search(rf"\b{re.escape(e)}\b", corpus)
    )

    now = int(time.time())
    src_dirs = source_paths(repo)
    for d in docs:
        e = last_commit_epoch(root, d["path"])
        d["last_commit_days_ago"] = round((now - e) / 86400, 1) if e else None
        sha = last_commit_sha(root, d["path"]) if e else None
        if sha and src_dirs:
            # Commits after the doc's last commit, not "since its timestamp":
            # --since includes the doc's own commit and any same-second commit.
            behind = run_git(root, "rev-list", "--count", f"{sha}..HEAD", "--", *src_dirs)
            d["source_commits_since_doc_touched"] = int(behind) if behind and behind.isdigit() else None

    return {
        "root": str(root),
        "identity": {
            "package_name": repo.package_name,
            "entrypoints": sorted(repo.entrypoints),
            "npm_scripts": sorted(repo.npm_scripts),
            "make_targets": sorted(repo.make_targets),
            "manifests": [m for m in ("package.json", "pyproject.toml", "setup.py", "setup.cfg",
                                      "Cargo.toml", "go.mod", "Gemfile", "pom.xml", "build.gradle")
                          if m in repo.rel_set],
            "cli_frameworks": detect_cli(repo),
            "test_dirs": [d for d in ("tests", "test", "spec", "__tests__", "e2e", "examples")
                          if d in repo.rel_set],
        },
        "docs": docs,
        "generators": detect_generators(repo),
        "public_surface_count": len(surface),
        "gaps": {
            "undocumented_public_names": [f"{s['name']} ({s['file']})" for s in undocumented_names],
            "undocumented_env_vars": [f"{k} ({v})" for k, v in sorted(undocumented_envs.items())],
            "undocumented_npm_scripts": undocumented_scripts,
            "undocumented_entrypoints": undocumented_entrypoints,
        },
    }


def render(data: dict) -> str:
    out: list[str] = [f"# Repository scan: {data['root']}", ""]
    ident = data["identity"]
    out.append("## Identity")
    out.append(f"- package name: {ident['package_name'] or '(none declared)'}")
    out.append(f"- manifests: {', '.join(ident['manifests']) or 'none found'}")
    out.append(f"- entrypoints: {', '.join(ident['entrypoints']) or 'none declared'}")
    if ident["cli_frameworks"]:
        out.append(f"- CLI framework: {', '.join(ident['cli_frameworks'])}")
    if ident["npm_scripts"]:
        out.append(f"- npm scripts: {', '.join(ident['npm_scripts'])}")
    if ident["make_targets"]:
        out.append(f"- make targets: {', '.join(ident['make_targets'][:20])}")
    if ident["test_dirs"]:
        out.append(f"- tests and examples: {', '.join(ident['test_dirs'])} "
                   f"(read these for real user goals before inventing any)")

    out.append("")
    out.append("## Generated reference")
    if data["generators"]:
        for g in data["generators"]:
            out.append(f"- {g['marker']}: {g['implication']}")
    else:
        out.append("- none detected; hand-written reference is the only reference")

    out.append("")
    out.append("## Existing documents")
    if not data["docs"]:
        out.append("- none found")
    for d in data["docs"][:40]:
        age = f"{d['last_commit_days_ago']}d ago" if d.get("last_commit_days_ago") is not None else "unknown age"
        behind = d.get("source_commits_since_doc_touched")
        drift = f", {behind} source commits since" if behind is not None else ""
        out.append(f"- {d['path']}  ({d['lines']} lines, {d['code_blocks']} code blocks, "
                   f"last touched {age}{drift})")
    if len(data["docs"]) > 40:
        out.append(f"- ... and {len(data['docs']) - 40} more")

    out.append("")
    out.append(f"## Gaps  (public surface: {data['public_surface_count']} names)")
    gaps = data["gaps"]
    any_gap = False
    for label, key in [
        ("Public names no document mentions", "undocumented_public_names"),
        ("Environment variables the source reads but no document explains", "undocumented_env_vars"),
        ("Declared entrypoints no document mentions", "undocumented_entrypoints"),
        ("npm scripts no document mentions", "undocumented_npm_scripts"),
    ]:
        items = gaps[key]
        if not items:
            continue
        any_gap = True
        out.append(f"- {label} ({len(items)}):")
        for item in items[:30]:
            out.append(f"    {item}")
        if len(items) > 30:
            out.append(f"    ... and {len(items) - 30} more")
    if not any_gap:
        out.append("- none: every public name, env var, and entrypoint appears in some document")
    out.append("")
    out.append("A gap is a candidate, not an obligation. Some names are public by "
               "accident and some scripts are internal. Decide, then say why in the plan.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="Repository root (default: current directory)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    data = scan(root)
    print(json.dumps(data, indent=2) if args.json else render(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
