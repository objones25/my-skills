"""Execute a reference file's Walkthrough section and diff its printed output.

Every ```python block runs in one namespace, in order. When the next fence is a bare
``` block, that is the documented output and must match stdout exactly (memory
addresses and trailing spaces normalised). Blocks that time things are run but not
compared, because timings vary by machine.
"""
from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import torch

TIMING_MARKERS = ("torch.profiler", "bench.Timer")


@dataclass
class Mismatch:
    file: str
    first_line: str
    documented: str
    actual: str


def _normalise(text: str) -> str:
    text = re.sub(r"0x[0-9a-f]+", "0xADDR", text)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def walkthrough_blocks(path: Path) -> list[tuple[str, str | None]]:
    text = path.read_text()
    section = text.split("## Walkthrough", 1)[1].split("\n## ", 1)[0]
    fences = re.findall(r"```(\w*)\n(.*?)```", section, re.S)
    blocks, i = [], 0
    while i < len(fences):
        lang, code = fences[i]
        expected = None
        if lang == "python" and i + 1 < len(fences) and fences[i + 1][0] == "":
            expected = fences[i + 1][1]
            i += 1
        if lang == "python":
            blocks.append((code, expected))
        i += 1
    return blocks


def run_walkthrough(path: Path, skip_modules: tuple[str, ...] = ()) -> list[Mismatch]:
    torch.manual_seed(0)
    namespace: dict = {}
    mismatches = []
    for code, expected in walkthrough_blocks(path):
        if any(f"import {m}" in code or f"from {m}" in code for m in skip_modules):
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, str(path), "exec"), namespace)
        if expected is None or any(marker in code for marker in TIMING_MARKERS):
            continue
        if _normalise(buf.getvalue()) != _normalise(expected):
            mismatches.append(Mismatch(path.name, code.strip().splitlines()[0],
                                       expected.strip(), buf.getvalue().strip()))
    return mismatches
