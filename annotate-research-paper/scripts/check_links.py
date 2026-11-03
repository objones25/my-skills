#!/usr/bin/env python3
"""Open every link in an annotation spec and record what actually came back.

Usage:
    python check_links.py spec.json link_report.json [--timeout 25]

The model writing the spec is not trusted to say a link was "verified": a
language model can claim a fetch it never made. This script does the fetch
itself and writes, for every URL in the spec (notes' links and cited_works):

    {"url": ..., "ok": true/false, "http": 200, "final_url": ..., "title": ...,
     "kind": "html"|"pdf"|"oembed", "error": null, "checked_at": "2026-10-02T..."}

annotate.py --link-report link_report.json then prints, for each cited work,
exactly how its link was checked (opened by the build script, opened during
research, seen only in search results, or expected but not opened), and warns
when a page title does not match the citation.

Titles come from: HTML <title> / og:title / citation_title meta; PDF metadata
or the first lines of page 1; YouTube via its oEmbed endpoint (title and
channel). TLS is always verified (the environment's CA bundle is honored via
REQUESTS_CA_BUNDLE / SSL_CERT_FILE); never disable verification. A failure is
recorded honestly: blocked by the network proxy, HTTP error, timeout, or no
title found. A failure here does not mean the link is wrong, only that this
script could not confirm it; annotate.py labels it accordingly.
"""
import argparse
import datetime as dt
import html
import io
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 annotate-research-paper/link-check")


def collect_urls(spec):
    urls = []
    for a in spec.get("annotations", []):
        for lk in a.get("links", []) or []:
            if lk.get("url"):
                urls.append(lk["url"])
    for cw in spec.get("cited_works", []) or []:
        if cw.get("url"):
            urls.append(cw["url"])
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def html_title(text):
    for pat in (r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\']([^"\']+)',
                r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
                r"<title[^>]*>(.*?)</title>"):
        m = re.search(pat, text, re.I | re.S)
        if m:
            t = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
            if t:
                return t
    m = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.I | re.S)
    if m:
        return html.unescape(re.sub(r"<[^>]+>|\s+", " ", m.group(1))).strip()
    return None


def pdf_title(data):
    try:
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        t = (doc.metadata or {}).get("title") or ""
        first = " ".join(doc[0].get_text().split()[:30]) if len(doc) else ""
        if t and len(t) > 6 and not t.lower().endswith((".dvi", ".doc", ".tex")):
            return t.strip(), first
        return (first[:160] or None), first
    except Exception as e:  # pragma: no cover
        return None, f"(could not parse PDF: {e})"


def youtube_id(url):
    p = urlparse(url)
    if p.netloc.endswith("youtu.be"):
        return p.path.strip("/")
    m = re.search(r"[?&]v=([\w-]{6,})", url)
    return m.group(1) if m else None


def check(url, timeout):
    rec = {"url": url, "ok": False, "http": None, "final_url": None, "title": None,
           "kind": None, "detail": None, "error": None,
           "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8"})
    try:
        if "youtube.com/watch" in url or "youtu.be/" in url:
            vid = youtube_id(url)
            o = s.get("https://www.youtube.com/oembed?format=json&url="
                      + quote(f"https://www.youtube.com/watch?v={vid}", safe=""), timeout=timeout)
            rec.update(http=o.status_code, kind="oembed", final_url=url)
            if o.ok:
                j = o.json()
                rec.update(ok=True, title=j.get("title"), detail=f"channel: {j.get('author_name')}")
            else:
                rec["error"] = f"oEmbed HTTP {o.status_code} (video private, removed, or ID wrong)"
            return rec
        r = s.get(url, timeout=timeout, allow_redirects=True)
        rec.update(http=r.status_code, final_url=r.url)
        ctype = r.headers.get("content-type", "").lower()
        if not r.ok:
            rec["error"] = f"HTTP {r.status_code}"
            return rec
        if "pdf" in ctype or r.content[:5] == b"%PDF-":
            rec["kind"] = "pdf"
            t, first = pdf_title(r.content)
            rec.update(title=t, detail=first[:300] if first else None)
        else:
            rec["kind"] = "html"
            text = r.text
            rec["title"] = html_title(text)
            if re.search(r"(?i)(captcha|are you a robot|access denied|enable javascript)", text[:4000]) \
                    and not rec["title"]:
                rec["error"] = "page served a bot check / JS wall instead of content"
                return rec
        rec["ok"] = bool(rec["title"])
        if not rec["ok"]:
            rec["error"] = "fetched but no title found"
    except requests.exceptions.ProxyError as e:
        rec["error"] = f"blocked or refused by the network proxy ({str(e)[:120]})"
    except requests.exceptions.SSLError as e:
        rec["error"] = f"TLS verification failed ({str(e)[:120]}); not bypassed"
    except requests.exceptions.Timeout:
        rec["error"] = f"timed out after {timeout}s"
    except requests.exceptions.RequestException as e:
        rec["error"] = f"request failed ({type(e).__name__}: {str(e)[:120]})"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("report")
    ap.add_argument("--timeout", type=float, default=25)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    urls = collect_urls(spec)
    out = {}
    for u in urls:
        rec = check(u, args.timeout)
        out[u] = rec
        flag = "OK " if rec["ok"] else "-- "
        print(f"{flag}{rec.get('http') or '   '} {u}\n      title: {rec.get('title')!r}"
              + (f"\n      error: {rec['error']}" if rec.get("error") else ""))
    Path(args.report).write_text(json.dumps(out, indent=1))
    n_ok = sum(1 for r in out.values() if r["ok"])
    print(f"\n{n_ok}/{len(out)} links opened by this script with a readable title -> {args.report}")


if __name__ == "__main__":
    main()
