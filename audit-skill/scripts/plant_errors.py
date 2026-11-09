#!/usr/bin/env python3
"""Plant known errors in a copy of a skill, and score an audit of that copy against them.

Usage:
    python plant_errors.py plant SOURCE_SKILL DEST PLANTS_JSON
    python plant_errors.py score RUN_DIR PLANTS_JSON [--json]

plant   Copies SOURCE_SKILL to DEST (which must not exist), then replaces each plant's
        "old" string with "new" in the copy. Every "old" must occur exactly once, or
        nothing is written. Writes DEST's sibling <name>.plants.json mapping each plant
        id to the file and line it landed on. SOURCE_SKILL is never written.
        Exit 0 planted, 2 refused.

score   Reads RUN_DIR/items.jsonl of an audit of the planted copy. A plant is detected
        when an item spanning at most 10 lines overlaps the planted lines and has a
        verdict in the plant's "expect" list, or is disputed with the auditor or the
        reviewer giving such a verdict. Other items flagged wrong, stale or version-specific are listed
        as other_flags: each is a false flag or a real error in the base skill, which a
        person decides. Exit 0 all detected, 1 some missed, 2 cannot read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from audit_common import load_json, read_jsonl
from start_audit import ignore_junk

FLAG_VERDICTS = frozenset({"wrong", "stale", "version-specific"})


def cmd_plant(source: Path, dest: Path, plants: dict[str, Any]) -> int:
    if dest.exists():
        print(f"plant_errors: {dest} already exists", file=sys.stderr)
        return 2
    if dest.resolve().is_relative_to(source.resolve()):
        print("plant_errors: DEST must be outside SOURCE_SKILL", file=sys.stderr)
        return 2
    texts: dict[str, str] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for plant in plants["plants"]:
        rel = plant["file"]
        if rel not in texts:
            path = source / rel
            if not path.is_file():
                print(f"plant_errors: {plant['id']}: {rel} not in {source}", file=sys.stderr)
                return 2
            texts[rel] = path.read_text(encoding="utf-8")
        count = texts[rel].count(plant["old"])
        if count != 1:
            print(f"plant_errors: {plant['id']}: 'old' occurs {count} times in {rel}; it must occur once",
                  file=sys.stderr)
            return 2
        texts[rel] = texts[rel].replace(plant["old"], plant["new"])
        line = texts[rel][: texts[rel].index(plant["new"])].count("\n") + 1
        manifest[plant["id"]] = {"file": rel, "line": line}
    shutil.copytree(source, dest, ignore=ignore_junk)
    for rel, text in texts.items():
        (dest / rel).write_text(text, encoding="utf-8")
    (dest.parent / f"{dest.name}.plants.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"planted {len(manifest)} error(s) in {dest}")
    return 0


MAX_ITEM_SPAN = 10


def locate(skill: Path, plant: dict[str, Any]) -> tuple[int, int] | None:
    """First and last line of the planted text in the copy."""
    path = skill / plant["file"]
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    index = text.find(plant["new"])
    if index < 0:
        return None
    first = text[:index].count("\n") + 1
    return first, first + plant["new"].count("\n")


def flagged(item: dict[str, Any], expect: set[str]) -> bool:
    if item.get("verdict") == "disputed":
        dispute = item.get("dispute") or {}
        return any((dispute.get(side) or {}).get("verdict") in expect for side in ("auditor", "reviewer"))
    return item.get("verdict") in expect


def cmd_score(run: Path, plants: dict[str, Any], as_json: bool) -> int:
    skill = Path(load_json(run / "meta.json")["skill_path"])
    items = read_jsonl(run / "items.jsonl")
    detected, missed, planted_items = [], [], set()
    for plant in plants["plants"]:
        span = locate(skill, plant)
        hits = []
        for i in items:
            start, end = i.get("line", 0), i.get("end_line") or i.get("line", 0)
            if span and i.get("file") == plant["file"] and end - start <= MAX_ITEM_SPAN \
                    and start <= span[1] and end >= span[0]:
                hits.append(i)
        planted_items.update(i["id"] for i in hits)
        ok = any(flagged(i, set(plant["expect"])) for i in hits)
        (detected if ok else missed).append(plant["id"])
    other = [i["id"] for i in items if i.get("verdict") in FLAG_VERDICTS and i["id"] not in planted_items]
    result = {"detected": detected, "missed": missed, "other_flags": other,
              "detection_rate": f"{len(detected)}/{len(plants['plants'])}"}
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"detected {result['detection_rate']}: {', '.join(detected) or 'none'}")
        print(f"missed: {', '.join(missed) or 'none'}")
        print(f"other items flagged (false flag or real error, check by hand): {', '.join(other) or 'none'}")
    return 0 if not missed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    plant = sub.add_parser("plant")
    plant.add_argument("source", type=Path)
    plant.add_argument("dest", type=Path)
    plant.add_argument("plants", type=Path)
    score = sub.add_parser("score")
    score.add_argument("run_dir", type=Path)
    score.add_argument("plants", type=Path)
    score.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        plants = load_json(args.plants)
        if args.action == "plant":
            return cmd_plant(args.source, args.dest, plants)
        return cmd_score(args.run_dir, plants, args.json)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"plant_errors: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
