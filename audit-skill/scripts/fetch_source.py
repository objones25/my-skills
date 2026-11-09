#!/usr/bin/env python3
"""Fetch a source's raw text into the run, so quotes are matched against the real page.

Usage:
    python fetch_source.py URL --run RUN_DIR [--dest sources|review/round-N/sources]

Why this exists: WebFetch returns a model-processed rendering of a page, so a quote
"found" in it proves nothing. This script downloads the bytes itself (urllib), turns
HTML into visible text (stdlib HTMLParser, dropping script, style and hidden elements) and PDFs into
text with `pdftotext` (poppler), refuses login, consent and error pages (by the
redirect target and by short pages that read like one), and saves DEST/S###.txt plus a row in DEST/index.jsonl
(id, url, final_url, status, content_type, sha256 of the raw bytes, fetched_at, path).

Fetching a URL already in DEST's index reuses it. Index updates hold a file lock,
so parallel workers can fetch at once (POSIX only: fcntl). The reviewer fetches into its own
folder so its copy is independent of the auditor's. Only http and https are allowed.

Prints the source id as the last line. Exit 0 on success, 2 when the fetch, the
conversion or the arguments fail (nothing is recorded then).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path

from audit_common import read_jsonl, write_jsonl

USER_AGENT = "audit-skill/0.1 (source verification; +https://claude.com/claude-code)"
TIMEOUT_SECONDS = 30
MAX_BYTES = 25_000_000
SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template"})
BLOCK_TAGS = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
                        "pre", "blockquote", "table", "ul", "ol", "dd", "dt", "figcaption", "title"})


class FetchError(Exception):
    """The source could not be fetched or converted to text."""


VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
                       "track", "wbr"})
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
LOGIN_PATH = re.compile(r"login|log-in|signin|sign-in|sso|/auth|cookie|consent|captcha|paywall", re.IGNORECASE)
SOFT_ERROR = re.compile(r"page not found|404 not found|not be found|sign in to continue|log in to continue|"
                        r"must (?:log|sign) in|access denied|enable javascript|are you a robot", re.IGNORECASE)
SOFT_ERROR_MAX_CHARS = 3000
META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)


def is_hidden(attrs: list[tuple[str, str | None]]) -> bool:
    values = {name.lower(): (value or "") for name, value in attrs}
    return "hidden" in values or values.get("aria-hidden", "").lower() == "true" \
        or bool(HIDDEN_STYLE.search(values.get("style", "")))


class TextExtractor(HTMLParser):
    """Visible text only: script, style and elements hidden by attribute or inline style are dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.hidden:
            if tag not in VOID_TAGS:
                self.hidden += 1
            return
        if tag not in VOID_TAGS and is_hidden(attrs):
            self.hidden = 1
            return
        if tag in SKIP_TAGS:
            self.skipping += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            self.hidden -= 1
            return
        if tag in SKIP_TAGS and self.skipping:
            self.skipping -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping and not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line) + "\n"


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def pdf_to_text(raw: bytes) -> str:
    if shutil.which("pdftotext") is None:
        raise FetchError("pdftotext not installed (brew install poppler / apt install poppler-utils)")
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(raw)
        proc = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, check=False,
                              timeout=120)
    if proc.returncode != 0:
        raise FetchError(f"pdftotext failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout.decode("utf-8", errors="replace")


def download(url: str) -> tuple[bytes, str, str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_BYTES + 1)
            ctype = response.headers.get("Content-Type", "")
            return raw, ctype, response.geturl(), response.status
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc


def refuse_non_source(url: str, final_url: str, text: str) -> None:
    """Raise when the fetch landed on a login, consent or error page instead of the source."""
    if final_url != url and LOGIN_PATH.search(final_url.split("?")[0]):
        raise FetchError(f"redirected to a login, sign-in or consent page ({final_url}), not the source; "
                         "find an open copy (author PDF, arXiv, PubMed Central)")
    if len(text) < SOFT_ERROR_MAX_CHARS and SOFT_ERROR.search(text):
        raise FetchError(f"the page looks like an error, login or bot check, not the source ({final_url}); "
                         "find an open copy")


def to_text(raw: bytes, ctype: str, url: str) -> str:
    if len(raw) > MAX_BYTES:
        raise FetchError(f"larger than {MAX_BYTES} bytes")
    lowered = ctype.lower()
    if "pdf" in lowered or raw.startswith(b"%PDF") or url.lower().endswith(".pdf"):
        return pdf_to_text(raw)
    charset = "utf-8"
    if "charset=" in lowered:
        charset = lowered.split("charset=")[-1].split(";")[0].strip() or "utf-8"
    elif (meta := META_CHARSET.search(raw[:4096])) is not None:
        charset = meta.group(1).decode("ascii")
    try:
        decoded = raw.decode(charset, errors="replace")
    except LookupError:
        decoded = raw.decode("utf-8", errors="replace")
    if "html" in lowered or decoded.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
        return html_to_text(decoded)
    return decoded


@contextlib.contextmanager
def index_lock(dest: Path) -> Iterator[None]:
    """Serialize index updates, so parallel auditor workers never reuse a source id."""
    with (dest / ".index.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def find_existing(index_path: Path, url: str) -> dict | None:
    return next((row for row in read_jsonl(index_path) if row["url"] == url), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dest", default="sources")
    args = parser.parse_args(argv)

    if not args.url.startswith(("http://", "https://")):
        print("fetch_source: only http and https URLs are allowed", file=sys.stderr)
        return 2
    run = args.run.resolve()
    dest = (run / args.dest).resolve()
    if not (run / "meta.json").is_file() or not dest.is_relative_to(run):
        print("fetch_source: --run must be a started audit and --dest must be inside it", file=sys.stderr)
        return 2
    index_path = dest / "index.jsonl"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with index_lock(dest):
            existing = find_existing(index_path, args.url)
    except ValueError as exc:
        print(f"fetch_source: {index_path} is corrupt: {exc}", file=sys.stderr)
        return 2
    if existing:
        print(f"already fetched: {dest / existing['path']}")
        print(existing["id"])
        return 0
    try:
        raw, ctype, final_url, status = download(args.url)
        text = to_text(raw, ctype, args.url)
        refuse_non_source(args.url, final_url, text)
    except FetchError as exc:
        print(f"fetch_source: {exc}", file=sys.stderr)
        return 2
    if not text.strip():
        print(f"fetch_source: no text extracted from {args.url} (scanned PDF or script-rendered page?)",
              file=sys.stderr)
        return 2
    with index_lock(dest):
        existing = find_existing(index_path, args.url)
        if existing:
            print(existing["id"])
            return 0
        rows = read_jsonl(index_path)
        source_id = f"S{len(rows) + 1:03d}"
        (dest / f"{source_id}.txt").write_text(text, encoding="utf-8")
        rows.append({
            "id": source_id, "url": args.url, "final_url": final_url, "status": status, "content_type": ctype,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "path": f"{source_id}.txt",
        })
        write_jsonl(index_path, rows)
    print(f"saved {len(text)} characters to {dest / (source_id + '.txt')}")
    print(source_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
