"""fetch_source.py: saves the raw text of a source so quotes can be string-matched."""

from __future__ import annotations

import http.server
import shutil
import threading
from pathlib import Path

import pytest
from conftest import read_jsonl, run_script

HTML = (b"<html><head><title>Doc</title><style>.x{color:red}</style><script>var a=1;</script></head>"
        b"<body><h1>Fixtures</h1><p>Fixtures are created when first requested &amp; torn down after.</p>"
        b"<p>Second paragraph.</p></body></html>")


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF with one line of Helvetica text, offsets computed so the xref is valid."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        routes = {"/doc.html": ("text/html; charset=utf-8", HTML),
                  "/paper.pdf": ("application/pdf", minimal_pdf("Mutants are a valid substitute for real faults")),
                  "/plain.txt": ("text/plain", b"plain text body that is long enough"),
                  "/login": ("text/html", b"<html><body><h1>Sign in</h1><p>Page not found or you must log in.</p></body></html>"),
                  "/hidden.html": ("text/html", (b"<html><body><p>Visible claim text here.</p>"
                                   b"<div style='display:none'>Hidden planted text here.</div>"
                                   b"<span hidden>Also hidden text.</span></body></html>")),
                  "/latin1.html": ("text/html", '<html><head><meta charset="iso-8859-1"></head><body><p>Caf\u00e9 '
                                   'na\u00efve r\u00e9sum\u00e9 text.</p></body></html>'.encode("latin-1"))}
        path = self.path.split("?")[0]
        if path == "/paper":
            self.send_response(302)
            self.send_header("Location", "/login")
            self.end_headers()
            return
        if path not in routes:
            self.send_error(404)
            return
        ctype, body = routes[path]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_html_is_saved_as_visible_text(run_dir, server):
    proc = run_script("fetch_source.py", f"{server}/doc.html", "--run", run_dir)
    assert proc.returncode == 0, proc.stderr
    row = read_jsonl(run_dir / "sources" / "index.jsonl")[0]
    text = (run_dir / "sources" / row["path"]).read_text()
    assert "Fixtures are created when first requested & torn down after." in text
    assert "color:red" not in text and "var a" not in text
    assert row["id"] == "S001" and row["status"] == 200 and len(row["sha256"]) == 64
    assert proc.stdout.strip().endswith("S001")


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext (poppler) not installed")
def test_pdf_is_converted_with_pdftotext(run_dir, server):
    proc = run_script("fetch_source.py", f"{server}/paper.pdf", "--run", run_dir)
    assert proc.returncode == 0, proc.stderr
    row = read_jsonl(run_dir / "sources" / "index.jsonl")[0]
    assert "Mutants are a valid substitute" in (run_dir / "sources" / row["path"]).read_text()


def test_same_url_is_reused_not_refetched(run_dir, server):
    run_script("fetch_source.py", f"{server}/plain.txt", "--run", run_dir)
    proc = run_script("fetch_source.py", f"{server}/plain.txt", "--run", run_dir)
    assert proc.stdout.strip().endswith("S001")
    assert len(read_jsonl(run_dir / "sources" / "index.jsonl")) == 1


def test_reviewer_fetches_go_to_their_own_folder(run_dir, server):
    proc = run_script("fetch_source.py", f"{server}/plain.txt", "--run", run_dir, "--dest", "review/round-1/sources")
    assert proc.returncode == 0, proc.stderr
    assert (run_dir / "review" / "round-1" / "sources" / "index.jsonl").exists()
    assert not (run_dir / "sources" / "index.jsonl").exists()


def test_http_error_exits_2_and_records_nothing(run_dir, server):
    proc = run_script("fetch_source.py", f"{server}/missing", "--run", run_dir)
    assert proc.returncode == 2
    assert "404" in proc.stderr
    assert not (run_dir / "sources" / "index.jsonl").exists()


def test_dest_outside_run_is_refused(run_dir, server):
    assert run_script("fetch_source.py", f"{server}/plain.txt", "--run", run_dir, "--dest", "../x").returncode == 2


def test_local_file_url_is_rejected(run_dir, tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    proc = run_script("fetch_source.py", target.as_uri(), "--run", run_dir)
    assert proc.returncode == 2
    assert "http" in proc.stderr


def test_parallel_fetches_get_unique_ids(run_dir, server):
    from concurrent.futures import ThreadPoolExecutor

    urls = [f"{server}/plain.txt?n={n}" for n in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        procs = list(pool.map(lambda u: run_script("fetch_source.py", u, "--run", run_dir), urls))
    assert all(p.returncode == 0 for p in procs), [p.stderr for p in procs]
    rows = read_jsonl(run_dir / "sources" / "index.jsonl")
    assert len(rows) == 8
    assert len({r["id"] for r in rows}) == 8
    assert all((run_dir / "sources" / r["path"]).is_file() for r in rows)


def test_redirect_to_a_login_page_is_refused(run_dir, server):
    proc = run_script("fetch_source.py", f"{server}/paper", "--run", run_dir)
    assert proc.returncode == 2
    assert "login" in proc.stderr or "sign-in" in proc.stderr
    assert not (run_dir / "sources" / "index.jsonl").exists() or not read_jsonl(run_dir / "sources" / "index.jsonl")


def test_hidden_html_is_dropped(run_dir, server):
    run_script("fetch_source.py", f"{server}/hidden.html", "--run", run_dir)
    text = (run_dir / "sources" / "S001.txt").read_text()
    assert "Visible claim text here." in text
    assert "Hidden planted" not in text and "Also hidden" not in text


def test_meta_charset_is_honoured(run_dir, server):
    run_script("fetch_source.py", f"{server}/latin1.html", "--run", run_dir)
    assert "Caf\u00e9 na\u00efve r\u00e9sum\u00e9" in (run_dir / "sources" / "S001.txt").read_text()


def test_corrupt_index_exits_2(run_dir, server):
    (run_dir / "sources" / "index.jsonl").write_text("not json\n")
    assert run_script("fetch_source.py", f"{server}/plain.txt", "--run", run_dir).returncode == 2
