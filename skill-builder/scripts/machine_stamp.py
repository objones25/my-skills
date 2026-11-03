#!/usr/bin/env python3
"""Print the date, machine and versions to state beside a machine-specific result.

Usage: machine_stamp.py [--package NAME ...]
Example output: On 2026-11-03 on macOS arm64 with Python 3.13.2, torch 2.15.0
Exit codes: 0 always, unless the arguments are wrong (2).
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import platform

SYSTEM_NAMES = {"Darwin": "macOS"}


def package_version(name: str) -> str:
    try:
        return f"{name} {importlib.metadata.version(name)}"
    except importlib.metadata.PackageNotFoundError:
        return f"{name} not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", action="append", default=[], metavar="NAME", help="also state this package's version")
    args = parser.parse_args()
    system = SYSTEM_NAMES.get(platform.system(), platform.system())
    versions = [f"Python {platform.python_version()}", *map(package_version, args.package)]
    print(f"On {datetime.date.today().isoformat()} on {system} {platform.machine()} with {', '.join(versions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
