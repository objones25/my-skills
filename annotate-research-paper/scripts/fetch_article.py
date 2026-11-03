#!/usr/bin/env python3
"""Turn a web article the user gave you into a PDF that annotate.py can mark up.

Usage (pick one source):
    python fetch_article.py --url  https://example.com/post   out.pdf
    python fetch_article.py --html saved_page.html            out.pdf [--source-url URL]
    python fetch_article.py --text page_text.md               out.pdf --source-url URL [--title T]

--url   renders the live page in headless Chromium and prints it to PDF (needs
        network access to that host; TLS is verified, never bypassed).
--html  renders a saved .html file the user gave you (no network needed).
--text  fallback when the page can't be printed (blocked, paywall-free but
        JS-only, bot wall): give the page's VERBATIM text (e.g., a browser's
        "page text" extraction or the user's copy) as Markdown or plain text.
        Never feed it a summarizer's output: the annotation must be of the
        author's words, not a paraphrase.

Every output starts with a provenance header: the source URL, the retrieval
date, and how it was produced (printed page / saved HTML / verbatim text), so
the annotated PDF always says what it is a copy of.
"""
import argparse
import datetime as dt
import html
import re
import sys
from pathlib import Path


def header_html(title, url, how):
    today = dt.date.today().isoformat()
    u = html.escape(url or "(no URL given)")
    return (f"<div style='font:9pt sans-serif;color:#555;border-bottom:1px solid #ccc;padding:4pt 0;"
            f"margin-bottom:10pt'><b>{html.escape(title or 'Article')}</b><br>Source: {u}<br>"
            f"Retrieved {today}; {html.escape(how)}.</div>")


def text_to_html(text):
    try:
        import markdown
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    except Exception:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        body = "".join(f"<p>{html.escape(p)}</p>" for p in paras)
    return body


PAGE_CSS = """
body { font: 11pt/1.5 Georgia, 'DejaVu Serif', serif; max-width: 6.3in; margin: 0 auto; color: #111; }
h1, h2, h3 { font-family: 'DejaVu Sans', sans-serif; line-height: 1.25; }
pre, code { font: 9pt 'DejaVu Sans Mono', monospace; white-space: pre-wrap; }
img { max-width: 100%; }
"""


def render(page_html, out, base_url=None):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        if base_url:
            page_html = page_html.replace("<head>", f"<head><base href='{html.escape(base_url)}'>", 1)
        pg.set_content(page_html, wait_until="load", timeout=60000)
        pg.pdf(path=out, format="Letter", margin={"top": "0.7in", "bottom": "0.7in", "left": "0.8in",
                                                  "right": "0.8in"}, print_background=True)
        b.close()


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--html")
    src.add_argument("--text")
    ap.add_argument("out")
    ap.add_argument("--source-url")
    ap.add_argument("--title")
    args = ap.parse_args()

    if args.url:
        from playwright.sync_api import sync_playwright, Error as PWError
        try:
            with sync_playwright() as p:
                b = p.chromium.launch()
                pg = b.new_page()
                resp = pg.goto(args.url, wait_until="networkidle", timeout=60000)
                status = resp.status if resp else None
                title = args.title or pg.title()
                text = pg.inner_text("body")[:3000].lower()
                if status and status >= 400:
                    sys.exit(f"HTTP {status} for {args.url}; use --html with a saved copy or --text with the "
                             "page's verbatim text.")
                if re.search(r"captcha|are you a robot|enable javascript|access denied", text) and len(text) < 2000:
                    sys.exit("The page served a bot check or JS wall instead of the article; use --html or --text.")
                pg.evaluate("""([h]) => { const d = document.createElement('div'); d.innerHTML = h;
                               document.body.insertBefore(d, document.body.firstChild); }""",
                            [header_html(title, args.url, "printed from the live page")])
                pg.pdf(path=args.out, format="Letter", print_background=True,
                       margin={"top": "0.6in", "bottom": "0.6in", "left": "0.6in", "right": "0.6in"})
                b.close()
        except PWError as e:
            sys.exit(f"Could not load {args.url} in Chromium ({str(e)[:200]}). If the network blocks this host, "
                     "use --html with a saved copy or --text with the page's verbatim text.")
        print(f"wrote {args.out} (printed live page: {title})")
        return

    if args.html:
        raw = Path(args.html).read_text(errors="replace")
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
        title = args.title or (html.unescape(m.group(1)).strip() if m else Path(args.html).stem)
        hdr = header_html(title, args.source_url, f"rendered from the saved file {Path(args.html).name}")
        if re.search(r"<body[^>]*>", raw, re.I):
            raw = re.sub(r"(<body[^>]*>)", r"\1" + hdr.replace("\\", "\\\\"), raw, count=1, flags=re.I)
        else:
            raw = f"<html><head></head><body>{hdr}{raw}</body></html>"
        render(raw, args.out, base_url=args.source_url)
        print(f"wrote {args.out} (from saved HTML: {title})")
        return

    text = Path(args.text).read_text(errors="replace")
    if not args.source_url:
        sys.exit("--text needs --source-url so the PDF records what it is a copy of.")
    title = args.title or next((l.lstrip("# ").strip() for l in text.splitlines() if l.strip()), "Article")
    page = (f"<html><head><meta charset='utf-8'><style>{PAGE_CSS}</style></head><body>"
            + header_html(title, args.source_url, "converted from the page's verbatim text (layout and "
                          "images not preserved)") + text_to_html(text) + "</body></html>")
    render(page, args.out)
    print(f"wrote {args.out} (from verbatim text: {title})")


if __name__ == "__main__":
    main()
