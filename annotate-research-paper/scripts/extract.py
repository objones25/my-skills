#!/usr/bin/env python3
"""Extract line-level text with stable IDs and render page images.

Usage:
    python extract.py input.pdf OUT_DIR [--dpi 110]

Writes:
    OUT_DIR/lines.txt    one line per text line:  p3 L7 [x0 y0 x1 y1] text
    OUT_DIR/pages/p3.png page renders, for reading the math visually
    OUT_DIR/meta.json    page sizes and the text-column extent per page
    OUT_DIR/inventory.txt  every numbered definition/theorem/lemma/.../proof and
                         numbered equation (annotate.py requires each to be
                         annotated or skipped with a reason), plus bold terms

Line IDs (p<page> L<n>) are 1-based and are what annotate.py uses for
"lines" anchors. annotate.py re-runs the same extraction, so IDs match as
long as the input PDF is unchanged.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf


def page_lines(page):
    """Return [(bbox, text)] for every text line, top-to-bottom, left-to-right.

    Lines on the same baseline that PyMuPDF splits into separate blocks
    (e.g. a section number and its title) are merged so IDs read naturally.
    """
    raw = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            raw.append((pymupdf.Rect(line["bbox"]), text))
    raw.sort(key=lambda t: (round(t[0].y1), t[0].x0))

    merged = []
    for rect, text in raw:
        if merged:
            prev_rect, prev_text = merged[-1]
            same_baseline = abs(prev_rect.y1 - rect.y1) < 2.5
            if same_baseline and rect.x0 >= prev_rect.x1 - 1:
                merged[-1] = (prev_rect | rect, prev_text + " " + text)
                continue
        merged.append((rect, text))
    return merged


def text_column(lines, page_rect):
    """Estimate the main text column's x-extent (ignores stray short lines)."""
    if not lines:
        return [page_rect.x0 + 72, page_rect.x1 - 72]
    xs0 = sorted(r.x0 for r, _ in lines)
    xs1 = sorted(r.x1 for r, _ in lines)
    lo = xs0[len(xs0) // 10]
    hi = xs1[int(len(xs1) * 0.9)]
    return [round(lo, 1), round(hi, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(args.pdf)

    meta = {"pages": []}
    rows = []
    thin_pages = []
    for i, page in enumerate(doc, start=1):
        lines = page_lines(page)
        for n, (r, text) in enumerate(lines, start=1):
            rows.append(f"p{i} L{n} [{r.x0:.0f} {r.y0:.0f} {r.x1:.0f} {r.y1:.0f}] {text}")
        if len(lines) < 3:
            thin_pages.append(i)
        meta["pages"].append({
            "page": i,
            "label": page.get_label() or str(i),
            "width": page.rect.width,
            "height": page.rect.height,
            "n_lines": len(lines),
            "text_column": text_column(lines, page.rect),
        })
        page.get_pixmap(dpi=args.dpi).save(out / "pages" / f"p{i}.png")

    (out / "lines.txt").write_text("\n".join(rows) + "\n")

    from annotate import find_objects, find_sections, find_citations  # annotate imports this module
    inv = ["# Figures, tables, equations, algorithms (each must be annotated or listed in the spec's 'skip',",
           "# and each needs an 'audit' entry checking it supports the claim it is cited for)"]
    for name, pno, y0, y1 in find_objects(doc):
        lines = page_lines(doc[pno - 1])
        ids = [f"L{i}" for i, (r, _) in enumerate(lines, 1) if r.y1 > y0 and r.y0 < y1]
        span = f"{ids[0]}-{ids[-1]}" if ids else "?"
        inv.append(f"p{pno} {span:>9}  {name}")
    inv.append("")
    inv.append("# Sections (the reading guide's route should mention these; not per-anchor coverage)")
    for name, pno in find_sections(doc):
        inv.append(f"p{pno}  {name}")
    inv.append("")
    cites = find_citations(doc)
    inv.append(f"# Reference-list entries found: {len(cites)} (choose the key ones to link in 'cited_works';")
    inv.append("# linking is NOT required for every entry, but any link you add must be verified)")
    for cid, text in cites[:60]:
        inv.append(f"{cid:>4}  {text}")
    inv.append("")
    inv.append("# Terms set in bold (often newly defined; consider whether each needs unpacking)")
    skip_re = re.compile(r"^(Definition|Theorem|Lemma|Proposition|Corollary|Algorithm|Remark|"
                         r"Proof|Exercise|Example|Assumption|Claim|Figure|Table|\d)")
    for i, page in enumerate(doc, start=1):
        spans = [sp for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for sp in l["spans"]]
        if not spans:
            continue
        sizes = sorted(round(sp["size"], 1) for sp in spans if sp["text"].strip())
        body = sizes[len(sizes) // 2] if sizes else 10
        terms = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                run = []  # consecutive bold spans on one line form one term
                for sp in line["spans"] + [None]:
                    bold = sp is not None and sp["flags"] & 16 and sp["size"] <= body + 0.6
                    if bold:
                        run.append(sp["text"])
                        continue
                    t = " ".join("".join(run).split()).strip(" .,:;")
                    if len(t) > 2 and not skip_re.match(t):
                        terms.append(t)
                    run = []
        if terms:
            inv.append(f"p{i}: " + "; ".join(dict.fromkeys(terms)))
    (out / "inventory.txt").write_text("\n".join(inv) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(doc)} pages, {len(rows)} lines -> {out}/lines.txt, {out}/inventory.txt, {out}/pages/")
    if thin_pages:
        print(f"WARNING: pages with almost no extractable text: {thin_pages}. "
              "If these hold real content the PDF may be scanned; use 'rect' anchors "
              "read off the page images instead of 'lines'/'quote'.", file=sys.stderr)


if __name__ == "__main__":
    main()
