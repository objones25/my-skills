#!/usr/bin/env python3
"""Score the [auto] expectations of evals/evals.json against one reply. Stdlib only.

Usage:
    python3 evals/grade.py ID ANSWER_FILE [--transcript FILE]

Prints JSON: {"id": ID, "expectations": [{"text": ..., "passed": bool}, ...]}.
The other expectations in evals.json need a human (or model) grader; these checks catch
the specific misclaims the skill's audits found, so a regression shows up as a False.
Exit 0 when every check passes, 1 when any fails, 2 for an unknown id or unreadable file.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

Check = Callable[[str, str], bool]

_CODE_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
_MOODS = re.compile(
    r"\b(cleaner|more maintainable|more flexible|more extensible|more decoupled|"
    r"better separation of concerns|improves testability)\b", re.IGNORECASE)


def _code_blocks(answer: str) -> list[str]:
    return _CODE_BLOCK.findall(answer)


def _search(pattern: str) -> Check:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda answer, _transcript: bool(rx.search(answer))


def _absent(pattern: str) -> Check:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda answer, _transcript: not rx.search(answer)


def _absent_in_code(pattern: str) -> Check:
    rx = re.compile(pattern)
    return lambda answer, _transcript: not any(rx.search(b) for b in _code_blocks(answer))


def _code_parses(answer: str, _transcript: str) -> bool:
    for block in _code_blocks(answer):
        try:
            ast.parse(block)
        except SyntaxError:
            return False
    return True


def _export_report_has_no_smtp(answer: str, _transcript: str) -> bool:
    """True only if some code block defines export_report and its body never touches SMTP."""
    found = False
    for block in _code_blocks(answer):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "export_report":
                found = True
                if re.search(r"smtplib|\bSMTP\(", ast.unparse(node)):
                    return False
    return found


def _scanner_run(_answer: str, transcript: str) -> bool:
    return "design_smells.py" in transcript


NO_MOODS: tuple[str, Check] = ("[auto] No unfalsifiable mood claims", _absent(_MOODS.pattern))
CODE_PARSES: tuple[str, Check] = ("[auto] All Python code blocks parse", _code_parses)

CHECKS: dict[int, list[tuple[str, Check]]] = {
    1: [
        ("[auto] Adds no Protocol or ABC", _absent_in_code(r"\((?:typing\.)?Protocol\b|\bABC\b|ABCMeta")),
        NO_MOODS,
    ],
    2: [
        ("[auto] Keeps the name export_report", _search(r"def export_report\s*\(")),
        ("[auto] export_report no longer constructs SMTP", _export_report_has_no_smtp),
        ("[auto] Scanner was run", _scanner_run),
        CODE_PARSES,
    ],
    3: [
        ("[auto] No ~100x or ~160x isinstance claim", _absent(r"\b1[06]0\s*(?:x|times)\b")),
        ("[auto] Does not say Protocols cannot share code",
         _absent(r"\b(?:cannot|can't|can not|don't|do not)\s+share\s+(?:any\s+)?(?:code|implementation)")),
        ("[auto] Mentions @abstractmethod or a type checker", _search(r"abstractmethod|\bmypy\b|\bpyright\b")),
    ],
    4: [
        ("[auto] Does not say never retry 429", _absent(r"never\s+retry\s+(?:the\s+)?(?:429|throttl)")),
        ("[auto] Mentions Retry-After", _search(r"retry-after")),
        ("[auto] Mentions an idempotency key", _search(r"idempoten\w*[ -]key")),
        CODE_PARSES,
        NO_MOODS,
    ],
    5: [
        ("[auto] Names import-linter or lint-imports", _search(r"import-linter|lint-imports|importlinter")),
        ("[auto] Does not claim check_layers.py catches transitive imports",
         _absent(r"check_layers\.py[^.\n]*\b(?:catches|detects|finds|sees|handles)\b[^.\n]*\btransitive")),
    ],
    6: [
        ("[auto] No Singleton class", _absent_in_code(r"def __new__\s*\(\s*cls|metaclass\s*=\s*\w*Singleton")),
        ("[auto] Mentions a lock or the lifespan", _search(r"\block\b|Lock\(|lifespan")),
        ("[auto] Mentions that @cache can run more than once under threads",
         lambda a, t: bool(re.search(r"\bcache\b", a, re.IGNORECASE) and re.search(r"thread", a, re.IGNORECASE))),
    ],
    7: [
        ("[auto] Cites the published CQRS warning", _search(r"risky complexity|Fowler")),
        NO_MOODS,
    ],
}


def grade(eval_id: int, answer: str, transcript: str = "") -> dict:
    if eval_id not in CHECKS:
        raise KeyError(f"no eval with id {eval_id}; known ids: {sorted(CHECKS)}")
    return {"id": eval_id,
            "expectations": [{"text": text, "passed": check(answer, transcript)}
                             for text, check in CHECKS[eval_id]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the [auto] expectations for one eval reply.")
    parser.add_argument("eval_id", type=int)
    parser.add_argument("answer_file", type=Path)
    parser.add_argument("--transcript", type=Path, help="tool-call transcript, for checks on what was run")
    args = parser.parse_args(argv)
    try:
        answer = args.answer_file.read_text()
        transcript = args.transcript.read_text() if args.transcript else ""
        result = grade(args.eval_id, answer, transcript)
    except (OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if all(e["passed"] for e in result["expectations"]) else 1


if __name__ == "__main__":
    sys.exit(main())
