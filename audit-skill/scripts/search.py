"""Run one scholarly search and log it, only if it actually ran.

Usage:
    python search.py ENGINE "QUERY" --run RUN --worker W --purpose P --item ID [--item ID ...] [--max N]

ENGINE is crossref (Crossref works API), s2 (Semantic Scholar graph API) or arxiv (arXiv API).
On success it prints one {"title", "url"} JSON line per result, appends one row to
RUN/parts/search-W.jsonl (id Q-W-NNN, the engine's name, the query exactly as run, today's
UTC date, the real hit count, purpose, item_ids, opened: []) and prints the new id last.

Why this exists: a search log is evidence only if every row is a search that ran. Agents
logged blocked or failed searches as "0 hits" by hand; this script writes the row itself and
writes nothing when the search fails, so that cannot happen. A rate limit (HTTP 429) is retried
at most 3 times, honouring Retry-After (capped at 30 s): limits are respected, not evaded.
These engines index research papers, so they do not replace a web search for style guides
or tool docs; use them alongside WebSearch, not as a way around its budget.

--endpoint replaces the engine's API URL (for tests or a mirror). Parallel workers can log at
once: rows are appended under a file lock (POSIX only: fcntl).

Exit codes: 0 searched and logged, 2 the search did not run or the arguments are bad
(nothing is logged then). It never exits 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from audit_common import SEARCH_PURPOSES, read_jsonl
from fetch_source import index_lock

TIMEOUT_S = 20
ATTEMPTS_MAX = 3
BACKOFF_S_MAX = 30
RESULTS_MAX = 25
USER_AGENT = "audit-skill-search/0.1"
WORKER_RE = re.compile(r"[A-Za-z0-9]+")
ENGINES = {
    "crossref": ("Crossref API", "https://api.crossref.org/works"),
    "s2": ("Semantic Scholar API", "https://api.semanticscholar.org/graph/v1/paper/search"),
    "arxiv": ("arXiv API", "https://export.arxiv.org/api/query"),
}


class SearchFailed(Exception):
    """The search did not run to completion; nothing may be logged."""


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, ATTEMPTS_MAX + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == ATTEMPTS_MAX:
                raise SearchFailed(f"HTTP {exc.code} from {url}") from exc
            retry_after = exc.headers.get("Retry-After", "")
            time.sleep(min(int(retry_after) if retry_after.isdigit() else 5 * attempt, BACKOFF_S_MAX))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SearchFailed(f"{type(exc).__name__}: {exc}") from exc
    raise AssertionError("unreachable: the loop returns or raises on its last attempt")


def _parse(engine: str, body: bytes) -> list[dict[str, str]]:
    try:
        if engine == "crossref":
            items = json.loads(body)["message"]["items"]
            return [{"title": (i.get("title") or [""])[0], "url": "https://doi.org/" + i["DOI"]} for i in items]
        if engine == "s2":
            return [{"title": p.get("title") or "", "url": p.get("url") or ""} for p in json.loads(body)["data"]]
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(body)
        return [{"title": " ".join((e.findtext("a:title", "", ns)).split()), "url": e.findtext("a:id", "", ns)}
                for e in root.findall("a:entry", ns)]
    except (ValueError, KeyError, TypeError, ET.ParseError) as exc:
        raise SearchFailed(f"could not parse the {engine} response: {exc}") from exc


def _url(engine: str, endpoint: str, query: str, limit: int) -> str:
    params = {"crossref": {"query": query, "rows": limit},
              "s2": {"query": query, "limit": limit, "fields": "title,url"},
              "arxiv": {"search_query": "all:" + query, "max_results": limit}}[engine]
    return endpoint + "?" + urllib.parse.urlencode(params)


def _next_id(rows: list[dict], worker: str) -> str:
    prefix = f"Q-{worker}-"
    numbers = [int(r["id"][len(prefix):]) for r in rows
               if isinstance(r.get("id"), str) and r["id"].startswith(prefix) and r["id"][len(prefix):].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", choices=sorted(ENGINES))
    parser.add_argument("query")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--worker", required=True, help="letters and digits; names the log part and the ids")
    parser.add_argument("--purpose", required=True, choices=sorted(SEARCH_PURPOSES))
    parser.add_argument("--item", action="append", default=[], dest="items", help="an item id; repeatable")
    parser.add_argument("--max", type=int, default=10, dest="limit")
    parser.add_argument("--endpoint", help="override the engine's API URL")
    args = parser.parse_args(argv)  # argparse exits 2 on bad arguments
    if not WORKER_RE.fullmatch(args.worker):
        parser.error("--worker must be letters and digits only")
    if not args.items:
        parser.error("name at least one --item the search is for")
    if not args.query.strip():
        parser.error("the query is empty")
    if not 1 <= args.limit <= RESULTS_MAX:
        parser.error(f"--max must be 1..{RESULTS_MAX}")
    parts = args.run / "parts"
    if not parts.is_dir():
        parser.error(f"{parts} does not exist; is --run the run folder?")

    name, default_endpoint = ENGINES[args.engine]
    try:
        results = _parse(args.engine, _get(_url(args.engine, args.endpoint or default_endpoint,
                                                args.query, args.limit)))
    except SearchFailed as exc:
        print(f"search: did not run, nothing logged: {exc}", file=sys.stderr)
        return 2

    log_path = parts / f"search-{args.worker}.jsonl"
    with index_lock(parts):
        search_id = _next_id(read_jsonl(log_path) if log_path.exists() else [], args.worker)
        row = {"id": search_id, "engine": name, "query": args.query,
               "date": datetime.now(UTC).date().isoformat(), "hits": len(results), "opened": [],
               "purpose": args.purpose, "item_ids": args.items}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    for result in results:
        print(json.dumps(result))
    print(search_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
