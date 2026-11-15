"""search.py: runs one scholarly search and logs it only if it actually ran."""

from __future__ import annotations

import http.server
import json
import threading

import pytest
from conftest import read_jsonl, run_script
from validate_audit import check_searches  # conftest puts scripts/ on sys.path

CROSSREF = json.dumps({"message": {"items": [
    {"DOI": "10.1109/ms.2014.80", "title": ["How API Documentation Fails"]},
    {"DOI": "10.1/x", "title": ["Another paper"]},
]}}).encode()
ARXIV = (b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
         b"<entry><id>http://arxiv.org/abs/1</id><title>A  README study</title></entry></feed>")


class Server:
    """A local HTTP server that answers each request with the next queued (status, headers, body)."""

    def __init__(self, responses: list[tuple[int, dict, bytes]]) -> None:
        queue = list(responses)
        self.requests: list[str] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append(self.path)
                status, headers, body = queue.pop(0) if len(queue) > 1 else queue[0]
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/search"

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def run(tmp_path):
    (tmp_path / "parts").mkdir()
    return tmp_path


def search(run, server, *extra, engine="crossref", purpose="support", items=("I001",), worker="W1"):
    args = [engine, "api documentation problems", "--run", run, "--worker", worker, "--purpose", purpose,
            "--endpoint", server.url, *extra]
    for item in items:
        args += ["--item", item]
    return run_script("search.py", *args)


def log(run, worker="W1"):
    path = run / "parts" / f"search-{worker}.jsonl"
    return read_jsonl(path) if path.exists() else []


def test_a_search_that_ran_is_logged_with_its_real_hits(run):
    server = Server([(200, {"Content-Type": "application/json"}, CROSSREF)])
    try:
        result = search(run, server, items=("I001", "I002"))
    finally:
        server.close()
    assert result.returncode == 0, result.stderr
    assert "How API Documentation Fails" in result.stdout
    assert result.stdout.strip().splitlines()[-1] == "Q-W1-001"
    [row] = log(run)
    assert row["id"] == "Q-W1-001" and row["engine"] == "Crossref API"
    assert row["query"] == "api documentation problems" and row["hits"] == 2
    assert row["purpose"] == "support" and row["item_ids"] == ["I001", "I002"] and row["opened"] == []
    assert check_searches([row]) == []


def test_ids_continue_after_existing_rows(run):
    (run / "parts" / "search-W1.jsonl").write_text(json.dumps(
        {"id": "Q-W1-005", "engine": "Crossref API", "query": "q", "date": "2026-11-15", "hits": 0,
         "opened": [], "purpose": "support", "item_ids": ["I001"]}) + "\n")
    server = Server([(200, {}, CROSSREF)])
    try:
        assert search(run, server).returncode == 0
    finally:
        server.close()
    assert [r["id"] for r in log(run)] == ["Q-W1-005", "Q-W1-006"]


@pytest.mark.parametrize("response", [
    (500, {}, b"server error"),
    (200, {}, b"not json at all"),
    (404, {}, b"missing"),
], ids=["server-error", "unparseable", "not-found"])
def test_a_search_that_did_not_run_is_not_logged(run, response):
    server = Server([response])
    try:
        result = search(run, server)
    finally:
        server.close()
    assert result.returncode == 2
    assert log(run) == []


def test_rate_limit_is_retried_a_bounded_number_of_times_then_refused(run):
    server = Server([(429, {"Retry-After": "0"}, b"slow down")])
    try:
        result = search(run, server)
    finally:
        server.close()
    assert result.returncode == 2
    assert len(server.requests) == 3
    assert log(run) == []


def test_rate_limit_then_success_is_logged_once(run):
    server = Server([(429, {"Retry-After": "0"}, b""), (200, {}, CROSSREF)])
    try:
        result = search(run, server)
    finally:
        server.close()
    assert result.returncode == 0, result.stderr
    assert len(log(run)) == 1


def test_arxiv_atom_feed_is_parsed(run):
    server = Server([(200, {}, ARXIV)])
    try:
        result = search(run, server, engine="arxiv", purpose="disconfirm")
    finally:
        server.close()
    assert result.returncode == 0, result.stderr
    [row] = log(run)
    assert row["engine"] == "arXiv API" and row["hits"] == 1 and row["purpose"] == "disconfirm"
    assert "A README study" in result.stdout


@pytest.mark.parametrize("bad", [
    {"purpose": "vibes"},
    {"items": ()},
    {"worker": "../x"},
], ids=["purpose", "no-item", "worker-path"])
def test_bad_arguments_exit_2_and_log_nothing(run, bad):
    server = Server([(200, {}, CROSSREF)])
    try:
        result = search(run, server, **bad)
    finally:
        server.close()
    assert result.returncode == 2
    assert not list((run / "parts").glob("search-*.jsonl")) or log(run) == []
    assert server.requests == []
