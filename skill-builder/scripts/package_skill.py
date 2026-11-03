#!/usr/bin/env python3
"""Zip a skill without junk, unpack the zip somewhere clean, and run its tests there.

Usage: package_skill.py SKILL_DIR [--out DIR] [--json]
The test command comes from metadata.test-command in SKILL.md, and it runs as a shell
command, so package only skills you trust.
Exit codes: 0 packaged, 1 checker errors or failing tests (no zip is left), 2 could not check.
Testing the unpacked copy catches files that only exist in your working folder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_skill import JUNK_NAMES, Checker, parse_frontmatter  # noqa: E402

TEST_TIMEOUT_SECONDS = 1800


def build_zip(skill: Path, target: Path) -> int:
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(skill.rglob("*")):
            relative = path.relative_to(skill)
            if path.is_file() and not JUNK_NAMES.intersection(relative.parts):
                archive.write(path, Path(skill.name) / relative)
                count += 1
    return count


def run_tests(zip_path: Path, name: str, command: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as clean:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(clean)
        return subprocess.run(command, shell=True, cwd=Path(clean) / name, capture_output=True,
                              text=True, timeout=TEST_TIMEOUT_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skill_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    skill = args.skill_dir.resolve()
    if not (skill / "SKILL.md").is_file():
        print(f"could not package {skill}: no SKILL.md", file=sys.stderr)
        return 2
    report: dict = {"skill": skill.name, "zip": None, "files": 0, "test_command": None, "tests": "not run"}

    def finish(code: int, message: str) -> int:
        print(json.dumps(report, indent=2) if args.json else message)
        return code

    # Junk is left out of the zip, so it is not an error here.
    errors = [f for f in Checker(skill).run() if f.severity == "error" and f.code != "SB012"]
    if errors:
        report["tests"] = "not run: checker errors"
        return finish(1, "not packaged, checker errors:\n" + "\n".join(f"  {f.code} {f.path}: {f.message}" for f in errors))
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{skill.name}.zip"
    report["files"] = build_zip(skill, target)
    fields, _ = parse_frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
    metadata = (fields or {}).get("metadata")
    command = metadata.get("test-command") if isinstance(metadata, dict) else None
    report["test_command"] = command
    if not command:
        report["zip"] = str(target)
        return finish(0, f"packaged {target} ({report['files']} files); tests not run: no metadata.test-command")
    result = run_tests(target, skill.name, command)
    if result.returncode != 0:
        target.unlink()
        report["tests"] = "failed"
        tail = (result.stdout + result.stderr).strip()[-2000:]
        return finish(1, f"not packaged, tests failed in the unpacked copy:\n{tail}")
    report["zip"] = str(target)
    report["tests"] = "passed"
    return finish(0, f"packaged {target} ({report['files']} files); tests passed in the unpacked copy")


if __name__ == "__main__":
    raise SystemExit(main())
