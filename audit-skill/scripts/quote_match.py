#!/usr/bin/env python3
"""Check that a quoted passage really appears in a fetched source's text.

Usage:
    python quote_match.py SOURCE_TXT "quoted passage"

Matching is exact after normalization: Unicode NFKC (ligatures), curly quotes to
straight, dashes to '-', whitespace collapsed, case folded, and PDF line-break
hyphenation tried both joined and kept. Superscript and subscript digits keep a
marker (10^5 is not 105). A quote may join fragments with '...' or '[...]': each
fragment must appear in order, at most MAX_GAP_CHARS apart, and the elided gap must
not contain a negation or hedge such as not, never, rarely, only, removed or deprecated
(so '... improves' cannot stand for 'rarely improves').
A model never decides this.

A quote must be long enough to be specific: at least MIN_QUOTE_CHARS normalized
characters overall and MIN_FRAGMENT_CHARS per fragment.

Exit 0 found, 1 not found or too short, 2 source file missing.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

MIN_QUOTE_CHARS = 25
MIN_FRAGMENT_CHARS = 12
MAX_GAP_CHARS = 300
TRANSLATE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00ad": None, "\u200b": None,
})
# Superscripts and subscripts keep a marker, so 10^5 never reads as 105 and a footnote mark stays apart.
SCRIPTS = str.maketrans({
    **{sup: f"^{d}" for d, sup in enumerate("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")},
    **{sub: f"_{d}" for d, sub in enumerate("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")},
})
ELLIPSIS = re.compile(r"\s*(?:\[\s*\.\.\.\s*\]|\.\.\.|\u2026)\s*")
HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\r?\n[ \t]*(\w)")
NEGATION = re.compile(r"\b(?:not|no|never|none|neither|nor|cannot|without|fails?|failed|rarely|seldom|hardly|barely|"
                      r"scarcely|only|except|unless|removed|deprecated|unsupported)\b|n't\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.translate(SCRIPTS)).translate(TRANSLATE)
    return re.sub(r"\s+", " ", text).strip().casefold()


def fragments(quote: str) -> list[str]:
    quote = quote.replace("\u2026", "...")
    return [normalize(part) for part in ELLIPSIS.split(quote) if normalize(part)]


def _in_order(parts: list[str], haystack: str) -> bool:
    """Each fragment appears after the previous one, with a short gap that holds no negation."""

    def search(index: int, start: int) -> bool:
        if index == len(parts):
            return True
        position = haystack.find(parts[index], start)
        while position >= 0:
            if index > 0:
                gap = haystack[start:position]
                if len(gap) > MAX_GAP_CHARS:
                    return False
                if NEGATION.search(gap):
                    position = haystack.find(parts[index], position + 1)
                    continue
            if search(index + 1, position + len(parts[index])):
                return True
            if index == 0:
                position = haystack.find(parts[index], position + 1)
            else:
                return False
        return False

    return search(0, 0)


def quote_found(quote: str, text: str) -> bool:
    parts = fragments(quote)
    if not parts:
        return False
    plain = normalize(text)
    joined = normalize(HYPHEN_BREAK.sub(r"\1\2", text))
    kept = normalize(HYPHEN_BREAK.sub(r"\1-\2", text))
    return any(_in_order(parts, haystack) for haystack in (plain, joined, kept))


def quote_problem(quote: str) -> str | None:
    """Why a quote is too weak to count, or None."""
    parts = fragments(quote)
    if sum(len(p) for p in parts) < MIN_QUOTE_CHARS:
        return f"quote shorter than {MIN_QUOTE_CHARS} characters"
    short = [p for p in parts if len(p) < MIN_FRAGMENT_CHARS]
    if short:
        return f"fragment shorter than {MIN_FRAGMENT_CHARS} characters: {short[0]!r}"
    return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    source, quote = Path(args[0]), args[1]
    if not source.is_file():
        print(f"quote_match: {source} not found", file=sys.stderr)
        return 2
    problem = quote_problem(quote)
    if problem:
        print(f"TOO SHORT: {problem}")
        return 1
    if quote_found(quote, source.read_text(encoding="utf-8", errors="replace")):
        print("FOUND")
        return 0
    print("NOT FOUND")
    return 1


if __name__ == "__main__":
    sys.exit(main())
