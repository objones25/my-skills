#!/usr/bin/env python3
"""Zip a skill without junk, unpack the zip somewhere clean, and run its tests there.

Usage: package_skill.py SKILL_DIR [--out DIR] [--json]
The test command comes from metadata.test-command in SKILL.md, and it runs as a shell
command, so package only skills you trust.
Exit codes: 0 packaged, 1 the skill failed (checker errors, a link leaving the skill, or
failing tests), 2 could not check. An earlier zip of this skill is removed first, and a zip
is only left on exit 0. Exit 0 with "tests not run" means there is no metadata.test-command.
Testing the unpacked copy catches files that only exist in your working folder.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_skill import Checker, NeedsPyYAML, is_excluded, parse_frontmatter, walk_files  # noqa: E402

TEST_TIMEOUT_SECONDS = 1800
BLOCKING_WARNINGS = {"SB014"}  # a TODO means the skill is not finished


def bad_links(skill: Path) -> list[str]:
    """Links that would leak an excluded or outside file, or vanish from the zip."""
    found = []
    for folder, subfolders, names in os.walk(skill):
        for name in [*subfolders, *names]:
            path = Path(folder) / name
            relative = path.relative_to(skill)
            if not path.is_symlink() or is_excluded(relative):
                continue
            try:
                target = path.resolve(strict=True)
            except (OSError, RuntimeError):
                found.append(str(relative))
                continue
            outside = skill != target and skill not in target.parents
            if path.is_dir() or outside or is_excluded(target.relative_to(skill)):
                found.append(str(relative))
    return sorted(found)


def build_zip(skill: Path, target: Path) -> int:
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
        for path in walk_files(skill):
            relative = path.relative_to(skill)
            if not is_excluded(relative):
                archive.write(path, Path(skill.name) / relative)
                count += 1
    return count


def move_into_place(built: Path, target: Path) -> None:
    built.replace(target)


def extract_keeping_modes(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            extracted = Path(archive.extract(info, destination))
            mode = info.external_attr >> 16
            if mode:
                extracted.chmod(mode & 0o777)


def run_tests(zip_path: Path, name: str, command: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as clean:
        extract_keeping_modes(zip_path, Path(clean))
        try:
            result = subprocess.run(command, shell=True, cwd=Path(clean) / name, capture_output=True,
                                    timeout=TEST_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return 1, f"timed out after {TEST_TIMEOUT_SECONDS} s"
        return result.returncode, (result.stdout + result.stderr).decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skill_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    skill = args.skill_dir.resolve()
    out = args.out.resolve()
    report: dict = {"skill": skill.name, "zip": None, "files": 0, "test_command": None, "tests": "not run"}

    def finish(code: int, message: str) -> int:
        print(json.dumps(report, indent=2) if args.json else message)
        return code

    if not (skill / "SKILL.md").is_file():
        print(f"could not package {skill}: no SKILL.md", file=sys.stderr)
        return 2
    if out == skill or skill in out.parents:
        print(f"could not package {skill}: --out must be outside the skill", file=sys.stderr)
        return 2
    target = out / f"{skill.name}.zip"
    try:
        out.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        # Junk is left out of the zip, so it is not an error here.
        findings = Checker(skill).run()
        fields, _ = parse_frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, NeedsPyYAML) as exc:
        print(f"could not package {skill}: {exc}", file=sys.stderr)
        return 2
    blocking = [f for f in findings if f.code != "SB012" and (f.severity == "error" or f.code in BLOCKING_WARNINGS)]
    if blocking:
        report["tests"] = "not run: checker errors"
        return finish(1, "not packaged, checker errors:\n" + "\n".join(f"  {f.code} {f.path}: {f.message}" for f in blocking))
    links = bad_links(skill)
    if links:
        report["tests"] = "not run: links"
        return finish(1, "not packaged, these links leave the skill or would vanish from the zip:\n  " + "\n  ".join(links))
    metadata = (fields or {}).get("metadata")
    command = metadata.get("test-command") if isinstance(metadata, dict) else None
    if command is not None and not isinstance(command, str):
        print(f"could not package {skill}: metadata.test-command must be text", file=sys.stderr)
        return 2
    report["test_command"] = command
    # Build beside the target, because a rename cannot cross volumes.
    with tempfile.TemporaryDirectory(dir=out) as building:
        built = Path(building) / target.name
        try:
            report["files"] = build_zip(skill, built)
        except OSError as exc:
            print(f"could not package {skill}: {exc}", file=sys.stderr)
            return 2
        if command:
            code, output = run_tests(built, skill.name, command)
            if code != 0:
                report["tests"] = "failed"
                return finish(1, f"not packaged, tests failed in the unpacked copy:\n{output.strip()[-2000:]}")
            report["tests"] = "passed"
        try:
            move_into_place(built, target)
        except OSError as exc:
            print(f"could not package {skill}: {exc}", file=sys.stderr)
            return 2
    report["zip"] = str(target)
    if command:
        return finish(0, f"packaged {target} ({report['files']} files); tests passed in the unpacked copy")
    return finish(0, f"packaged {target} ({report['files']} files); tests not run: no metadata.test-command")


if __name__ == "__main__":
    raise SystemExit(main())
