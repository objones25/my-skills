#!/usr/bin/env python3
"""Deterministic style and structure lints for Markdown documentation.

Every rule here is mechanical, so it produces the same finding every time and
can be cited by ID in an audit report. Judgement calls -- is this example
useful, is this the right level of detail -- are deliberately not in here.
Those belong to the reviewer, and mixing them in makes the machine findings
less trustworthy.

Rule sources are named in RULES below: mostly the Google developer
documentation style guide, its accessibility page, and standard-readme.

Usage:
    python3 lint_prose.py README.md
    python3 lint_prose.py docs/*.md --json
    python3 lint_prose.py README.md --min-severity warn
    python3 lint_prose.py README.md --disable TONE003,A11Y003
    python3 lint_prose.py --list-rules

Exit codes:
    0  nothing at or above --fail-on (and always for --list-rules)
    1  at least one finding at or above --fail-on (default: error)
    2  bad invocation, or only non-Markdown files given

Markdown only: .rst, .adoc, .org, and .txt files are skipped with a note on
stderr, because their heading and code syntax differs from Markdown's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

SEVERITIES = ["info", "warn", "error"]

RULES = {
    # id: (severity, message template, source)
    "DOC001": ("error", "Document has {n} level-1 headings; it should have exactly one",
               "Google style: headings"),
    "DOC002": ("warn", "Heading level jumps from h{a} to h{b}; do not skip levels",
               "Google style: headings"),
    "DOC003": ("warn", "Heading has no content and no subsections under it",
               "Google style: headings"),
    "DOC004": ("warn", "Heading contains a link; move the link into the body",
               "Google style: headings"),
    "DOC005": ("info", "Heading looks like Title Case; use sentence case",
               "Google style: headings"),
    "DOC006": ("info", "Duplicate heading text makes anchors ambiguous",
               "Google style: headings"),
    "CODE001": ("warn", "Fenced code block has no language tag",
                "Google style: code samples"),
    "CODE002": ("warn", "Elision uses '...' or an ellipsis; use a comment in the block's language",
                "Google style: code samples"),
    "CODE003": ("info", "Code line is {n} characters; wrap at 80",
                "Google style: code samples"),
    "LINK001": ("error", "Link text {text!r} is meaningless out of context",
                "Google style: link text"),
    "LINK002": ("warn", "Link text is a bare URL; use a descriptive phrase",
                "Google style: link text"),
    "LINK003": ("warn", "Link text {text!r} points at two different targets",
                "Google style: link text"),
    "A11Y001": ("error", "Image has no alt text", "Google style: accessibility"),
    "A11Y002": ("warn", "Directional reference {word!r}; use 'preceding' or 'following'",
                "Google style: accessibility"),
    "A11Y003": ("info", "Sentence is {n} words; keep sentences under 26",
                "Google style: accessibility"),
    "TONE001": ("warn", "{word!r} judges the reader's experience; state the step count or the time instead",
                "Google style: word list"),
    "TONE002": ("warn", "{word!r} is marketing, not documentation; give the measurement",
                "Diataxis: reference must be neutral and factual"),
    "TONE003": ("info", "{word!r} is filler; delete it", "Google style: word list"),
    "TONE004": ("info", "Prefer {fix!r} over {word!r}", "Google style: word list"),
    "TONE005": ("warn", "{word!r} is exclusionary or ableist; replace it",
                "Google style: word list"),
    "TONE006": ("info", "Exclamation mark", "Google style: voice and tone"),
    "HONEST001": ("error", "Unfinished marker {word!r} left in published documentation",
                  "Write the Docs: incorrect docs are worse than missing docs"),
    "HONEST002": ("error", "Unfilled template placeholder {word!r}",
                  "Write the Docs: incorrect docs are worse than missing docs"),
    "HONEST003": ("warn", "{word!r} tells the reader the docs are unfinished; say what is missing instead",
                  "Write the Docs: incorrect docs are worse than missing docs"),
    "RM001": ("error", "README has no fenced code block; a reader cannot copy anything",
              "standard-readme: Install and Usage must contain a code block"),
    "RM002": ("error", "README has no section matching {want}", "standard-readme spec"),
    "RM003": ("warn", "{n} badges above the first heading of substance; badge soup buries the description",
              "standard-readme: badges are optional and unlabelled"),
    "RM004": ("info", "Opening description is {n} characters; standard-readme caps the short description at 120",
              "standard-readme spec"),
}

# word -> (rule, replacement or "")
TONE_WORDS: dict[str, tuple[str, str]] = {}
for _w in ["easy", "easily", "simple", "simply", "trivial", "trivially",
           "obviously", "of course", "straightforward", "painless"]:
    TONE_WORDS[_w] = ("TONE001", "")
for _w in ["blazing fast", "blazingly fast", "lightning fast", "lightning-fast",
           "seamless", "seamlessly", "effortless", "effortlessly", "cutting-edge",
           "state-of-the-art", "revolutionary", "game-changing", "world-class",
           "best-in-class", "supercharge", "supercharged", "magical", "magic sauce",
           "battle-tested", "rock-solid", "bulletproof", "unparalleled"]:
    TONE_WORDS[_w] = ("TONE002", "")
for _w in ["just", "really", "very", "actually", "basically", "simply put",
           "needless to say", "please note", "note that", "it should be noted"]:
    TONE_WORDS[_w] = ("TONE003", "")
for _w, _fix in {
    "utilize": "use", "leverage": "use", "in order to": "to", "e.g.": "for example",
    "i.e.": "that is", "click on": "click", "allows you to": "lets you",
    "enables you to": "lets you", "currently": "(delete; it is implied)",
    "please": "(delete from instructions)", "terminate": "stop", "abort": "cancel",
    "whitelist": "allowlist", "blacklist": "denylist", "master": "primary",
    "slave": "replica", "dummy": "placeholder", "sanity check": "quick check",
    "and/or": "or", "via": "with, or through",
}.items():
    TONE_WORDS[_w] = ("TONE004", _fix)
for _w in ["dumb down", "cripple", "crippled", "lame", "you guys", "blind to",
           "grandfathered"]:
    TONE_WORDS[_w] = ("TONE005", "")

VAGUE_LINK_TEXT = {
    "here", "click here", "this", "this link", "link", "read more", "more",
    "this document", "this article", "this page", "see here", "learn more",
    "click", "info", "documentation", "docs",
}

UNFINISHED = ["todo", "tbd", "fixme", "coming soon", "to be written",
              "documentation pending", "lorem ipsum", "under construction"]
UNFINISHED_SOFT = ["work in progress", "wip", "xxx", "not yet documented"]
# Case-sensitive on the SCREAMING_CASE forms: "your project" is ordinary
# English, while YOUR_PROJECT is a scaffold nobody filled in.
PLACEHOLDERS = [
    re.compile(r"YOUR[_ ](?:NAME|PROJECT|TOKEN|KEY|API|USERNAME|ORG|REPO)"),
    re.compile(r"PROJECT[_ ]NAME"),
    re.compile(r"REPLACE[_ ]ME"),
    re.compile(r"CHANGE[_ ]ME"),
    re.compile(r"\{\{\s*\w+\s*\}\}"),
    re.compile(r"<your[ _\-][\w \-]+>", re.I),
    re.compile(r"example\.com/your", re.I),
]

README_SECTIONS = {
    "install": ["install", "installation", "getting started", "setup", "quick start",
                "quickstart", "get started"],
    "usage": ["usage", "using", "example", "examples", "quick start", "quickstart",
              "how to use", "getting started"],
    "license": ["license", "licence", "licensing"],
    "contributing": ["contributing", "contribute", "development", "developing",
                     "how to contribute"],
}

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK_RE = re.compile(r"(!?)\[([^\]\n]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
CODESPAN_RE = re.compile(r"`[^`\n]+`")
# Only flag "above"/"below" when they refer to a position in the document.
# "Put a line below the text" describes rendering, not navigation, and flagging
# it would train the reader to ignore this rule.
DOC_NOUNS = (r"diagram|table|image|figure|example|snippet|command|section|output|"
             r"list|code|screenshot|block|steps?|instructions?|paragraph|chapter")
DIRECTIONAL_RE = re.compile(
    rf"(?:the|this)\s+(?:{DOC_NOUNS})\s+(above|below)\b"
    rf"|\b(?:{DOC_NOUNS})\s+(?:shown|listed|described)\s+(above|below)\b"
    r"|\bas\s+(?:shown|described|mentioned|noted)\s+(above|below)\b"
    r"|\bsee\s+(above|below)\b"
    r"|\b(above|below)\s*[,:]\s*$",
    re.I,
)


# Importing the sibling module would otherwise write scripts/__pycache__/,
# and these scripts promise to write nothing to disk.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_claims import blank_frontmatter, normalize_setext  # noqa: E402


@dataclass
class Finding:
    rule: str
    severity: str
    line: int
    message: str
    source: str
    excerpt: str = ""


def emit(out: list[Finding], rule: str, line: int, excerpt: str = "", **fmt) -> None:
    sev, template, source = RULES[rule]
    out.append(Finding(rule, sev, line, template.format(**fmt), source, excerpt[:120]))


def strip_code(line: str) -> str:
    """Remove inline code spans so lints do not fire on identifiers."""
    return CODESPAN_RE.sub(" ", line)


def parse(text: str):
    """Yield (lineno, line, in_fence, fence_lang) for every line."""
    in_fence = False
    marker = ""
    lang = ""
    for i, line in enumerate(text.splitlines(), 1):
        fm = FENCE_RE.match(line)
        if fm and not in_fence:
            in_fence, marker, lang = True, fm.group(2)[0] * 3, fm.group(3).lower()
            yield i, line, False, lang
            continue
        if in_fence and line.strip().startswith(marker) and set(line.strip()) <= set(marker):
            in_fence = False
            yield i, line, False, ""
            continue
        yield i, line, in_fence, lang if in_fence else ""


def check_structure(text: str, out: list[Finding]) -> None:
    headings: list[tuple[int, int, str]] = []
    content_after: dict[int, bool] = {}
    last_heading_line = None
    for lineno, line, in_fence, _ in parse(text):
        if in_fence:
            if last_heading_line is not None:
                content_after[last_heading_line] = True
            continue
        hm = HEADING_RE.match(line)
        if hm:
            headings.append((lineno, len(hm.group(1)), hm.group(2).strip()))
            last_heading_line = lineno
            content_after.setdefault(lineno, False)
        elif line.strip() and last_heading_line is not None:
            content_after[last_heading_line] = True

    h1s = [h for h in headings if h[1] == 1]
    # Some projects set the title with raw HTML for centering. It still counts.
    html_h1 = len(re.findall(r"<h1[\s>]", text, re.I))
    total_h1 = len(h1s) + html_h1
    if total_h1 != 1:
        emit(out, "DOC001", h1s[1][0] if len(h1s) > 1 else 1, n=total_h1)

    prev_level = None
    seen_text: dict[str, int] = {}
    for idx, (lineno, level, htext) in enumerate(headings):
        if prev_level is not None and level > prev_level + 1:
            emit(out, "DOC002", lineno, htext, a=prev_level, b=level)
        prev_level = level
        # A heading with no prose is fine when it introduces subsections. It is
        # only a defect when the next heading is a sibling or an uncle.
        nxt_level = headings[idx + 1][1] if idx + 1 < len(headings) else level
        if not content_after.get(lineno, False) and nxt_level <= level:
            emit(out, "DOC003", lineno, htext)
        if LINK_RE.search(htext):
            emit(out, "DOC004", lineno, htext)
        words = [w for w in re.sub(r"[`*_]", "", htext).split() if w.isalpha()]
        if len(words) >= 3:
            capped = sum(1 for w in words[1:] if w[0].isupper() and not w.isupper())
            if capped >= len(words[1:]) * 0.6:
                emit(out, "DOC005", lineno, htext)
        key = htext.lower()
        if key in seen_text:
            emit(out, "DOC006", lineno, htext)
        seen_text[key] = lineno


def check_code(text: str, out: list[Finding]) -> None:
    in_fence = False
    marker = ""
    fence_line = 0
    lang = ""
    for i, line in enumerate(text.splitlines(), 1):
        fm = FENCE_RE.match(line)
        if fm and not in_fence:
            in_fence, marker = True, fm.group(2)[0] * 3
            lang, fence_line = fm.group(3).lower(), i
            if not lang:
                emit(out, "CODE001", i, line.strip())
            continue
        if in_fence:
            if line.strip().startswith(marker) and set(line.strip()) <= set(marker):
                in_fence = False
                continue
            stripped = line.strip()
            if stripped in {"...", "…"} or stripped.endswith(("  ...", " …")):
                emit(out, "CODE002", i, stripped)
            if len(line.rstrip()) > 80:
                emit(out, "CODE003", i, stripped[:60], n=len(line.rstrip()))


def check_links(text: str, out: list[Finding]) -> None:
    text_to_targets: dict[str, set[str]] = {}
    for lineno, line, in_fence, _ in parse(text):
        if in_fence:
            continue
        for m in LINK_RE.finditer(line):
            is_image, label, target = m.group(1), m.group(2).strip(), m.group(3)
            if is_image:
                if not label:
                    emit(out, "A11Y001", lineno, m.group(0))
                continue
            plain = re.sub(r"[`*_]", "", label).strip().rstrip(".").lower()
            if plain in VAGUE_LINK_TEXT:
                emit(out, "LINK001", lineno, m.group(0), text=label)
            elif plain.startswith(("http://", "https://", "www.")):
                emit(out, "LINK002", lineno, m.group(0))
            if plain and plain not in VAGUE_LINK_TEXT:
                text_to_targets.setdefault(plain, set()).add(target)
    for label, targets in text_to_targets.items():
        if len(targets) > 1:
            emit(out, "LINK003", 1, ", ".join(sorted(targets))[:100], text=label)


def check_prose(text: str, out: list[Finding]) -> None:
    for lineno, line, in_fence, _ in parse(text):
        if in_fence or line.startswith(("    ", "\t")):
            continue
        clean = strip_code(line)
        clean_nolink = LINK_RE.sub(r"\2", clean)
        low = clean_nolink.lower()

        for phrase, (rule, fix) in TONE_WORDS.items():
            pattern = r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])"
            if re.search(pattern, low):
                if rule == "TONE004":
                    emit(out, rule, lineno, clean.strip(), word=phrase, fix=fix)
                else:
                    emit(out, rule, lineno, clean.strip(), word=phrase)

        m = DIRECTIONAL_RE.search(clean_nolink)
        if m:
            emit(out, "A11Y002", lineno, clean.strip(), word=next(g for g in m.groups() if g))

        if "!" in clean_nolink and not clean_nolink.lstrip().startswith(("!", "|")):
            for sentence_end in re.finditer(r"\w!(?:\s|$)", clean_nolink):
                emit(out, "TONE006", lineno, clean.strip())
                break

        for marker in UNFINISHED:
            if re.search(r"(?<![\w-])" + re.escape(marker) + r"(?![\w-])", low):
                emit(out, "HONEST001", lineno, clean.strip(), word=marker.upper())
        for marker in UNFINISHED_SOFT:
            if re.search(r"(?<![\w-])" + re.escape(marker) + r"(?![\w-])", low):
                emit(out, "HONEST003", lineno, clean.strip(), word=marker.upper())
        for pat in PLACEHOLDERS:
            pm = pat.search(clean)
            if pm:
                emit(out, "HONEST002", lineno, clean.strip(), word=pm.group(0))



def check_sentence_length(text: str, out: list[Finding]) -> None:
    """A11Y003 over whole paragraphs, so hard-wrapped sentences are measured
    as one sentence rather than one line at a time."""
    para: list[tuple[int, str]] = []

    def flush() -> None:
        if not para:
            return
        # Join the lines, remembering where each one starts, so a finding is
        # reported on the line where its sentence begins.
        pieces = [LINK_RE.sub(r"\2", strip_code(t)).strip() for _, t in para]
        starts, pos = [], 0
        for piece in pieces:
            starts.append(pos)
            pos += len(piece) + 1
        joined = " ".join(pieces)
        for m in re.finditer(r"\S.*?(?:[.?!](?=\s|$)|$)", joined):
            sentence = m.group(0)
            words = [w for w in sentence.split() if any(ch.isalpha() for ch in w)]
            # Google: "fewer than 26 words per sentence", so 26 is already too many.
            if len(words) >= 26:
                idx = max(i for i, st in enumerate(starts) if st <= m.start())
                emit(out, "A11Y003", para[idx][0], sentence.strip()[:80], n=len(words))
        para.clear()

    for lineno, line, in_fence, _ in parse(text):
        stripped = line.strip()
        is_break = (in_fence or not stripped or HEADING_RE.match(line) or FENCE_RE.match(line)
                    or line.startswith(("    ", "\t")) or stripped.startswith(("|", ">", "<")))
        # A list item starts a new paragraph of its own.
        is_item = bool(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line))
        if is_break:
            flush()
            continue
        if is_item:
            flush()
        para.append((lineno, line))
    flush()


def check_readme(text: str, out: list[Finding]) -> None:
    lines = text.splitlines()
    headings = []
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            headings.append(m.group(2).lower())
    has_fence = any(FENCE_RE.match(ln) for ln in lines)
    if not has_fence:
        emit(out, "RM001", 1)
    for key, wanted in README_SECTIONS.items():
        if not any(any(w in h for w in wanted) for h in headings):
            emit(out, "RM002", 1, want=key)

    badge_count = 0
    for line in lines[:40]:
        badge_count += len(re.findall(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", line))
        badge_count += len(re.findall(r"!\[[^\]]*\]\(https://(?:img\.shields\.io|badgen\.net|badge\.fury\.io)[^)]*\)", line))
    if badge_count > 6:
        emit(out, "RM003", 1, n=badge_count)

    for i, line in enumerate(lines, 1):
        if HEADING_RE.match(line):
            body = []
            for nxt in lines[i:]:
                if HEADING_RE.match(nxt) or FENCE_RE.match(nxt):
                    break
                if nxt.strip() and not nxt.strip().startswith(("[!", "!", "<", "|", ">")):
                    body.append(nxt.strip())
                elif body:
                    break
            para = " ".join(body)
            if len(para) > 120:
                emit(out, "RM004", i + 1, para[:80], n=len(para))
            break


def lint(path: Path, readme_checks: bool | None = None) -> list[Finding]:
    # Setext headings become ATX first so line numbers stay stable (the rewrite
    # replaces the title line and drops the underline, keeping earlier lines put).
    text = normalize_setext(blank_frontmatter(
        path.read_text(encoding="utf-8", errors="replace")))
    out: list[Finding] = []
    check_structure(text, out)
    check_code(text, out)
    check_links(text, out)
    check_prose(text, out)
    check_sentence_length(text, out)
    is_readme = readme_checks if readme_checks is not None else path.stem.lower() == "readme"
    if is_readme:
        check_readme(text, out)
    return sorted(out, key=lambda f: (f.line, f.rule))


def render(path: Path, findings: list[Finding], min_sev: str) -> str:
    floor = SEVERITIES.index(min_sev)
    shown = [f for f in findings if SEVERITIES.index(f.severity) >= floor]
    lines = [f"# Style report: {path}", ""]
    if not shown:
        lines.append(f"No findings at severity {min_sev} or above.")
    for f in shown:
        tag = {"error": "ERROR", "warn": "WARN ", "info": "INFO "}[f.severity]
        lines.append(f"{tag} L{f.line:<4} {f.rule}  {f.message}")
        if f.excerpt:
            lines.append(f"              {f.excerpt}")
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    lines.append("")
    lines.append("Totals: " + (", ".join(f"{counts[s]} {s}" for s in SEVERITIES if s in counts) or "clean"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-severity", choices=SEVERITIES, default="info",
                    help="Hide findings below this severity in text output")
    ap.add_argument("--fail-on", choices=SEVERITIES, default="error",
                    help="Exit 1 if any finding is at or above this severity")
    ap.add_argument("--disable", default="", help="Comma-separated rule IDs to skip")
    ap.add_argument("--readme", action="store_true", help="Force README-specific checks on")
    ap.add_argument("--list-rules", action="store_true", help="Print the rule table and exit")
    args = ap.parse_args(argv)

    if args.list_rules:
        for rid, (sev, msg, src) in sorted(RULES.items()):
            print(f"{rid}  {sev:<5}  {msg}\n          source: {src}")
        return 0
    if not args.docs:
        ap.error("the following arguments are required: docs")

    unknown_rules = {r.strip().upper() for r in args.disable.split(",") if r.strip()} - set(RULES)
    if unknown_rules:
        print(f"error: unknown rule ID(s) for --disable: {', '.join(sorted(unknown_rules))}; "
              f"see --list-rules", file=sys.stderr)
        return 2

    disabled = {r.strip().upper() for r in args.disable.split(",") if r.strip()}
    fail_floor = SEVERITIES.index(args.fail_on)
    failed = False
    payload = []
    checked = 0
    for name in args.docs:
        path = Path(name)
        if not path.is_file():
            print(f"error: {path} is not a readable file", file=sys.stderr)
            return 2
        if path.suffix.lower() in {".rst", ".rest", ".txt", ".adoc", ".asciidoc", ".org"}:
            print(f"note: skipped {path}: lint_prose.py reads Markdown only", file=sys.stderr)
            continue
        checked += 1
        findings = [f for f in lint(path, True if args.readme else None) if f.rule not in disabled]
        if any(SEVERITIES.index(f.severity) >= fail_floor for f in findings):
            failed = True
        if args.json:
            payload.append({"doc": str(path), "findings": [asdict(f) for f in findings]})
        else:
            print(render(path, findings, args.min_severity))
            print()
    if not checked:
        print("error: no Markdown files to lint", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
