#!/usr/bin/env python3
"""Fetch, cache, and slice the current Claude Code hooks documentation.

Usage:
  fetch_reference.py --index                      list headings with line numbers
  fetch_reference.py --section "Stop input"       print one section (heading match is case-insensitive)
  fetch_reference.py --grep "requires Claude Code v"   print matching lines with line numbers
  fetch_reference.py --doc guide --index          use the hooks guide instead of the reference
  fetch_reference.py --refresh ...                ignore the cache
  fetch_reference.py --file hooks.md ...          use a local copy (if the download fails)

The cache lives in $XDG_CACHE_HOME/hook-writer or ~/.cache/hook-writer and is
refreshed when older than one day. Standard library only.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DOCS = {
    "reference": "https://code.claude.com/docs/en/hooks.md",
    "guide": "https://code.claude.com/docs/en/hooks-guide.md",
}
MAX_AGE_S = 24 * 3600
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    path = Path(base) / "hook-writer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url, dest):
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:  # urllib can fail on macOS python.org builds without certs
        err = exc
    if shutil.which("curl"):
        result = subprocess.run(["curl", "-fsSL", "-m", "30", "-o", str(dest), url], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        err = result.stderr.strip() or err
    print(f"download failed for {url}: {err}", file=sys.stderr)
    return False


def load(doc, refresh, local):
    if local:
        return Path(local).read_text(encoding="utf-8")
    dest = cache_dir() / f"{doc}.md"
    stale = not dest.exists() or time.time() - dest.stat().st_mtime > MAX_AGE_S
    if refresh or stale:
        tmp = dest.with_suffix(".tmp")
        if download(DOCS[doc], tmp) and tmp.stat().st_size > 1000:
            tmp.replace(dest)
        elif dest.exists():
            print("[download failed; using the older cached copy]", file=sys.stderr)
        else:
            sys.exit("no cached copy either; download the page yourself and pass --file")
    age_h = (time.time() - dest.stat().st_mtime) / 3600
    print(f"[{doc}: {DOCS[doc]} cached {age_h:.1f} h ago at {dest}]", file=sys.stderr)
    return dest.read_text(encoding="utf-8")


def print_index(lines):
    in_code = False
    for n, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            depth = len(m.group(1))
            print(f"{n:5d}  {'  ' * (depth - 1)}{m.group(2)}")


def print_section(lines, name):
    target = name.strip().lower().strip("`")
    in_code = False
    start = level = None
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if not m:
            continue
        title = m.group(2).strip().lower().replace("`", "")
        if start is None and title == target:
            start, level = i, len(m.group(1))
        elif start is not None and len(m.group(1)) <= level:
            print("\n".join(lines[start:i]))
            return True
    if start is not None:
        print("\n".join(lines[start:]))
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", choices=sorted(DOCS), default="reference")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--file", help="read this local markdown file instead of downloading")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", action="store_true")
    group.add_argument("--section")
    group.add_argument("--grep")
    args = ap.parse_args()

    lines = load(args.doc, args.refresh, args.file).splitlines()
    if args.index:
        print_index(lines)
    elif args.section:
        if not print_section(lines, args.section):
            sys.exit(f"no heading named {args.section!r}; run --index to see headings")
    else:
        pattern = re.compile(args.grep, re.IGNORECASE)
        for n, line in enumerate(lines, 1):
            if pattern.search(line):
                print(f"{n:5d}: {line}")


if __name__ == "__main__":
    main()
