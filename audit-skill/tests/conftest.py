"""Shared fixtures for audit-skill's script tests. Run: python -m pytest tests -q"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

SAMPLE_SKILL_MD = """---
name: sample
description: "A sample skill. Use when testing. Not for production."
---

# Sample

Python 3.13 added a REPL with colour, see https://docs.python.org/3/whatsnew/3.13.html.
Always run the tests before release.

```bash
python -m pytest
```

Plain prose with nothing checkable in it
"""

SOURCE_TEXT = (
    "What's New In Python 3.13\n"
    "Python now uses a new interactive shell by default, based on code from the PyPy project.\n"
    "It supports colour by default and multi-line editing.\n"
)


def run_script(name: str, *args: str | Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        capture_output=True, text=True, timeout=120, cwd=cwd, check=False,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def sample_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "repo" / "sample"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
    (skill / "scripts" / "tool.py").write_text("print('hi')\n", encoding="utf-8")
    return skill


@pytest.fixture
def run_dir(sample_skill: Path, tmp_path: Path) -> Path:
    """A started audit with candidates found, but no items yet."""
    proc = run_script("start_audit.py", sample_skill, "--scope", "full", "--root", tmp_path / "repo",
                      "--date", "2026-11-07")
    assert proc.returncode == 0, proc.stderr
    run = Path(proc.stdout.strip().splitlines()[-1])
    proc = run_script("find_candidates.py", run)
    assert proc.returncode == 0, proc.stderr
    return run


def add_source(run: Path, text: str = SOURCE_TEXT, url: str = "https://docs.python.org/3/whatsnew/3.13.html",
               source_id: str = "S001", dest: str = "sources", fetched_at: str | None = None) -> str:
    import datetime as dt

    fetched_at = fetched_at or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    folder = run / dest
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{source_id}.txt").write_text(text, encoding="utf-8")
    index = folder / "index.jsonl"
    rows = read_jsonl(index) if index.exists() else []
    rows.append({"id": source_id, "url": url, "final_url": url, "status": 200, "content_type": "text/html",
                 "sha256": "0" * 64, "fetched_at": fetched_at, "path": f"{source_id}.txt"})
    write_jsonl(index, rows)
    return source_id


def good_items() -> list[dict]:
    """Items that cover every candidate in SAMPLE_SKILL_MD and pass validation."""
    return [
        {"id": "I001", "file": "SKILL.md", "line": 3, "kind": "self-claim", "text": "Description states use and scope.",
         "in_scope": True, "verdict": "correct", "strength": "A",
         "evidence": [{"type": "command", "command": "head -4 SKILL.md", "output": "description: ...", "exit_code": 0}],
         "searches": [], "note": "Read the frontmatter."},
        {"id": "I002", "file": "SKILL.md", "line": 8, "kind": "tool-behavior",
         "text": "Python 3.13's interactive shell supports colour.", "in_scope": True, "verdict": "correct",
         "strength": "B",
         "evidence": [{"type": "source", "source_id": "S001", "tier": 1, "interest": "none",
                       "quote": "It supports colour by default and multi-line editing."}],
         "searches": ["Q001", "Q002"], "note": "Official what's-new page."},
        {"id": "I003", "file": "SKILL.md", "line": 9, "kind": "recommendation", "text": "Run tests before release.",
         "in_scope": False, "verdict": "not-checked", "evidence": [], "searches": []},
        {"id": "I004", "file": "SKILL.md", "line": 11, "end_line": 13, "kind": "example",
         "text": "The pytest invocation runs the suite.", "in_scope": False, "verdict": "not-checked",
         "evidence": [], "searches": []},
        {"id": "I005", "file": "scripts/tool.py", "line": 1, "kind": "script", "text": "tool.py prints hi.",
         "in_scope": True, "verdict": "correct", "strength": "A",
         "evidence": [{"type": "command", "command": "python scripts/tool.py", "output": "hi", "exit_code": 0}],
         "searches": [], "note": "Ran it."},
    ]


def good_searches() -> list[dict]:
    return [
        {"id": "Q001", "engine": "WebSearch", "query": "python 3.13 new repl colour", "date": "2026-11-07",
         "hits": 10, "opened": ["https://docs.python.org/3/whatsnew/3.13.html"], "purpose": "support",
         "item_ids": ["I002"]},
        {"id": "Q002", "engine": "WebSearch", "query": "python 3.13 repl no colour bug", "date": "2026-11-07",
         "hits": 10, "opened": [], "purpose": "disconfirm", "item_ids": ["I002"]},
    ]


@pytest.fixture
def complete_run(run_dir: Path) -> Path:
    """A run whose items validate cleanly."""
    add_source(run_dir)
    write_jsonl(run_dir / "items.jsonl", good_items())
    write_jsonl(run_dir / "search_log.jsonl", good_searches())
    write_jsonl(run_dir / "dismissed.jsonl", [{"file": "SKILL.md", "lines": [15], "reason": "no checkable statement"}])
    return run_dir
