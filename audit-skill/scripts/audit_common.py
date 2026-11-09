"""Constants and helpers shared by audit-skill's scripts. Stdlib only.

The vocabularies here are the contract between the auditor, the reviewer and the
scripts; references/formats.md documents them for people.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VERDICTS = frozenset({
    "correct", "wrong", "stale", "version-specific", "couldnt-verify", "disputed", "not-checked",
})
DEFINITE_VERDICTS = frozenset({"correct", "wrong", "stale", "version-specific"})
KINDS = frozenset({
    "tool-behavior", "empirical", "recommendation", "citation", "self-claim", "script", "example", "test",
})
STRENGTHS = ("A", "B", "C", "D", "E")
TIERS = frozenset({1, 2, 3, 4, 5, 6})
SEARCH_PURPOSES = frozenset({"support", "disconfirm", "known-item", "snowball"})
MAX_REVIEW_ROUNDS = 3

# Junk that viewers and interpreters create on their own. Everything else is hashed.
IGNORED_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hypothesis"})
IGNORED_FILES = frozenset({".DS_Store"})
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
# Not copied into the workspace (large, machine-specific), but still hashed.
NOT_COPIED_DIRS = frozenset({".venv", "venv", "node_modules"})


HASH_IGNORED_DIRS = frozenset({".git"})


def _hash_ignored(rel: Path) -> bool:
    return any(part in HASH_IGNORED_DIRS for part in rel.parts) or rel.name in IGNORED_FILES \
        or rel.suffix in IGNORED_SUFFIXES


def _ignored(rel: Path) -> bool:
    return any(part in IGNORED_DIRS for part in rel.parts) or rel.name in IGNORED_FILES \
        or rel.suffix in IGNORED_SUFFIXES


def iter_skill_files(root: Path) -> list[Path]:
    """Every file in a skill, relative to its root, sorted, without caches, OS junk or virtualenvs."""
    return sorted(rel for rel in (p.relative_to(root) for p in root.rglob("*") if p.is_file())
                  if not _ignored(rel) and not any(part in NOT_COPIED_DIRS for part in rel.parts))


def tree_hash(root: Path) -> str:
    """SHA-256 over every path, file mode and file content, including empty directories and virtualenvs.

    Used to prove the audit was read-only, so it skips only .git, .DS_Store and Python bytecode.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _hash_ignored(rel) or (path.is_dir() and path.name in IGNORED_DIRS):
            continue
        mode = path.lstat().st_mode
        digest.update(f"{rel.as_posix()}\0{mode:o}\0".encode())
        if path.is_file() and not path.is_symlink():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        elif path.is_symlink():
            digest.update(str(path.readlink()).encode())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: each line must be a JSON object")  # noqa: TRY004 - bad file content, reported like a parse error
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a JSON object")  # noqa: TRY004 - bad file content, reported like a parse error
    return data


def is_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def item_hash(item: dict[str, Any]) -> str:
    """Hash of everything the auditor controls in an item. A changed hash means the reviewer must see it again."""
    content = {key: value for key, value in item.items() if key != "dispute"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_snapshot(run: Path) -> dict[str, Any]:
    """Hashes of everything the report is built from, frozen when the review loop ends."""

    def file_hash(name: str) -> str:
        path = run / name
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""

    sources = sorted(p.relative_to(run).as_posix() for p in (run / "sources").rglob("*") if p.is_file()) \
        if (run / "sources").is_dir() else []
    names = ["meta.json", "candidates.jsonl", "dismissed.jsonl", "suggestions.jsonl", "search_log.jsonl", *sources]
    return {
        "items": {str(i.get("id")): item_hash(i) for i in read_jsonl(run / "items.jsonl")},
        "files": {name: file_hash(name) for name in names if not name.endswith(".lock")},
    }


def load_meta(run: Path) -> dict[str, Any]:
    meta_path = run / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; run start_audit.py first")
    return load_json(meta_path)
