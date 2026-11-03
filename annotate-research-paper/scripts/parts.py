#!/usr/bin/env python3
"""Split a paper into page chunks for worker subagents, merge their spec parts, and diff an
independent verifier's key points against the merged spec.

Usage:
    python parts.py chunks work/map.json ex/lines.txt work/chunks.json [--max 5] [--min 3]
    python parts.py merge  work/map.json work/chunks.json work/parts work/spec.json --pdf paper.pdf
                           [--globals work/globals.json]
    python parts.py verify work/spec.json work/chunks.json work/verify ex/lines.txt --pdf paper.pdf

map.json (written by the orchestrator before any page is annotated):
    {"title": str, "summary": str, "domain": str, "pages": [first, last] (only for a partial request),
     "sections": [{"id", "title", "pages": [a, b], "walkthrough": id (optional) | "references": true}],
     "reader_goal": str,                       # what the reader wants from the paper (references/walkthroughs.md)
     "walkthroughs": [{"id", "title", "pages": [a, b], "goal", "why", "anchor"}],   # explainers, usually 1-3
     "glossary": [{"term", "page"}]}            # page = where the paper first defines the term

chunks: checks the map (unique section ids; every page with body text in a section; each explainer
has a goal and spans the sections that name it; at most 4 explainers), then cuts consecutive page ranges
of at most --max pages, at section starts once a chunk has --min pages. Pages after the reference list
are chunked too (appendices often follow the references).

merge: every chunk must have work/parts/part_<a>-<b>.json with "pages": [a, b] and the map's sections
unchanged. With --pdf, each part is re-run through `annotate.py --part`, so a part that never passed its
gate cannot be merged. Writes spec.json (title, pages, sections, walkthroughs, annotations, page_review,
skip, audit, plus guide / critique / cited_works / verify_rejections from --globals) and
work/critique_notes.json.

verify: work/verify/part_<a>-<b>.json are verifier outputs, {"pages": [a, b], "points": [{"page",
"quote", "what"}]}, written from the page images without seeing the spec; one per chunk is required.
A verifier point is covered when a key point or a short note anchor on that page quotes overlapping
words (3+ word containment or a shared 4-word run) or, with --pdf, part of the same sentence. A point
the spec listed as not_key is a disagreement, not coverage. Every miss and disagreement is either fixed
or listed in globals.json "verify_rejections" as {"page", "quote", "why"} (why: 6+ words). Exit 1 while
anything is unresolved.
"""
import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

CONCEPT_WHAT = {"mechanism", "concept", "definition"}
REVIEW_MIN_WORDS = 150
ANCHOR_MAX_LINES = 6       # a note anchored on more lines than this is too broad to count as covering a point
REJECT_MIN_WORDS = 6
LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def norm(t):
    t = unicodedata.normalize("NFKC", "".join(LIG.get(c, c) for c in (t or "")))
    t = re.sub(r"-\s*\n\s*", "", t)
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return " ".join(t.split())


def grams(t, n=4):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def overlaps(a, b):
    """Same passage: one contains the other on word boundaries (3+ words; the shorter one's last word may be
    cut at a line-break hyphen, "pro-"), or they share a 4-word run."""
    a, b = norm(a), norm(b)
    short = min(len(a.split()), len(b.split()))
    if short < 3:
        return False
    if f" {a}" in f" {b} " or f" {b}" in f" {a} ":
        return True
    return short >= 4 and bool(grams(a) & grams(b))


def read_lines(path):
    """lines.txt -> {page: {"L7": text}}."""
    by_page = {}
    for ln in Path(path).read_text().splitlines():
        m = re.match(r"p(\d+) (L\d+) \[[^\]]*\] ?(.*)", ln)
        if m:
            by_page.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3)
    return by_page


def line_range(spec_lines):
    m = re.match(r"L(\d+)(?:\s*-\s*L?(\d+))?", spec_lines or "")
    return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None


def page_sentences(pdf, pno):
    """Sentences of one page in reading order (PyMuPDF's block order keeps columns together)."""
    t = " ".join(pdf[pno - 1].get_text().split())
    t = re.sub(r"(\w)- (\w)", r"\1\2", t)
    return [x for x in re.split(r"(?<=[.?!])\s+(?=[A-Z“\"(])", t) if len(x.split()) >= 5]


def check_map(mp, lines):
    errors = []
    n = max(lines) if lines else 0
    first, last = [int(x) for x in mp.get("pages", [1, n])]
    secs = mp.get("sections") or []
    if not mp.get("summary"):
        errors.append("map: add 'summary' (2-4 sentences: the claim and where the contribution is)")
    if not secs:
        errors.append("map: no 'sections'")
    ids = [s.get("id") for s in secs]
    if len(ids) != len(set(ids)):
        errors.append(f"map: duplicate section ids {sorted({i for i in ids if ids.count(i) > 1})}")
    wts = {w.get("id"): w for w in mp.get("walkthroughs") or []}
    covered = set()
    for s in secs:
        tag = f"map section {s.get('id')!r}"
        try:
            s0, s1 = int(s["pages"][0]), int(s["pages"][1])
            assert s.get("id") and s.get("title") and s0 <= s1
        except (KeyError, TypeError, ValueError, IndexError, AssertionError):
            errors.append(f"{tag}: needs 'id', 'title', and 'pages': [first, last]")
            continue
        covered |= set(range(s0, s1 + 1))
        if s.get("references"):
            continue
        wid = s.get("walkthrough")
        if wid:
            w = wts.get(wid)
            if not w:
                errors.append(f"{tag}: names explainer {wid!r}; add it to 'walkthroughs' with id, title, pages, "
                              "goal, why, anchor")
            elif any(not w.get(f) for f in ("title", "pages", "goal", "why", "anchor")):
                errors.append(f"map explainer {wid!r}: needs title, pages, goal, why, anchor")
            elif not any(int(r[0]) <= s0 and s1 <= int(r[1]) for r in (w["pages"], w.get("proof_pages") or w["pages"])):
                errors.append(f"map walkthrough {wid!r}: pages {w['pages']} do not span section {s['id']!r} "
                              f"(pp. {s0}-{s1})")
    for wid, w in wts.items():
        if len(" ".join(str(w.get("goal", "")).split()).split()) < 10:
            errors.append(f"map explainer {wid!r}: 'goal' must say what the reader will be able to do after "
                          "reading it (10+ words). See references/walkthroughs.md, 'Planning'.")
    if len(wts) > 4 and len(" ".join(str(mp.get("many_explainers_because", "")).split()).split()) < 10:
        errors.append(f"map: {len(wts)} explainers. Plan one per algorithm or central idea the reader asked to "
                      "understand (usually 1-3); for more, give 'many_explainers_because' (10+ words).")
    if wts and len(" ".join(str(mp.get("reader_goal", "")).split()).split()) < 8:
        errors.append("map: add 'reader_goal' (what the reader wants from this paper; the user's words if they "
                      "gave any, else the default in references/walkthroughs.md).")
    for p in range(first, last + 1):
        words = sum(len(t.split()) for t in lines.get(p, {}).values())
        if p not in covered and words >= REVIEW_MIN_WORDS:
            errors.append(f"map: p{p} ({words} words) is in no section")
    return errors, first, last


def cmd_chunks(a):
    mp = json.loads(Path(a.map).read_text())
    lines = read_lines(a.lines)
    errors, first, last = check_map(mp, lines)
    if errors:
        sys.exit("\n".join(errors))
    starts = {int(s["pages"][0]) for s in mp["sections"]}
    chunks, cur = [], []
    for p in range(first, last + 1):
        if cur and (len(cur) >= a.max or (p in starts and len(cur) >= a.min)):
            chunks.append([cur[0], cur[-1]])
            cur = []
        cur.append(p)
    if cur:   # a short tail: fold it into the previous chunk if that stays within --max, else split the two evenly
        prev = chunks[-1] if chunks and chunks[-1][1] == cur[0] - 1 else None
        if prev and len(cur) < a.min:
            total = cur[-1] - prev[0] + 1
            if total <= a.max:
                prev[1] = cur[-1]
            else:
                prev[1] = prev[0] + (total + 1) // 2 - 1
                chunks.append([prev[1] + 1, cur[-1]])
        else:
            chunks.append([cur[0], cur[-1]])
    Path(a.out).write_text(json.dumps({"chunks": chunks}, indent=1))
    for c in chunks:
        secs = [s["id"] for s in mp["sections"] if s["pages"][0] <= c[1] and c[0] <= s["pages"][1]]
        print(f"chunk pp. {c[0]}-{c[1]} ({c[1] - c[0] + 1} pages): {', '.join(secs)}")
    print(f"{len(chunks)} chunks for pp. {first}-{last} -> {a.out}")


def cmd_merge(a):
    mp = json.loads(Path(a.map).read_text())
    chunks = json.loads(Path(a.chunks).read_text())["chunks"]
    glob = json.loads(Path(a.globals).read_text()) if a.globals else {}
    errors, warnings = [], []
    spec = {"title": mp.get("title", ""), "sections": mp.get("sections", []),
            "walkthroughs": glob.get("walkthroughs", mp.get("walkthroughs", [])),
            "annotations": [], "page_review": [], "skip": [], "audit": []}
    if mp.get("pages"):
        spec["pages"] = mp["pages"]
    for k in ("guide", "critique", "cited_works", "verify_rejections"):
        if k in glob:
            spec[k] = glob[k]
    gate = Path(__file__).with_name("annotate.py")
    crit_notes, titles = [], {}
    for c0, c1 in chunks:
        f = Path(a.parts) / f"part_{c0}-{c1}.json"
        if not f.exists():
            errors.append(f"chunk pp. {c0}-{c1}: {f} is missing (the worker has not finished, or wrote elsewhere)")
            continue
        part = json.loads(f.read_text())
        if [int(x) for x in part.get("pages", [])] != [c0, c1]:
            errors.append(f"{f.name}: 'pages' is {part.get('pages')}, expected [{c0}, {c1}]")
        if part.get("sections") != mp.get("sections"):
            errors.append(f"{f.name}: 'sections' differs from map.json. Workers copy the map's sections unchanged; "
                          "a walkthrough decision changes only in the map, and then every chunk is re-gated.")
        if a.pdf:
            r = subprocess.run([sys.executable, str(gate), a.pdf, str(f), str(Path(a.parts) / f"gate_{c0}-{c1}.pdf"),
                                "--part"], capture_output=True, text=True)
            if r.returncode:
                tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
                errors.append(f"{f.name}: fails its gate (annotate.py --part):\n      " + "\n      ".join(tail))
        for key in ("annotations", "page_review", "skip", "audit"):
            for item in part.get(key, []):
                p = item.get("page")
                if p is not None and not (c0 <= int(p) <= c1):
                    errors.append(f"{f.name}: {key} entry on p{p} is outside the chunk")
                spec[key].append(item)
        for an in part.get("annotations", []):
            t = re.sub(r"<[^>]+>", "", an.get("title", "")).strip().lower()
            if t and an.get("kind") != "check":
                titles.setdefault(t, []).append(int(an["page"]))
        crit_notes += [dict(x, chunk=[c0, c1]) for x in part.get("critique_notes", [])]
        print(f"{f.name}: {len(part.get('annotations', []))} notes, {len(part.get('page_review', []))} pages "
              f"reviewed, {sum(len(r.get('points', [])) for r in part.get('page_review', []))} key points")
    for t, ps in titles.items():
        if len(ps) > 1 and max(ps) - min(ps) > 1:
            warnings.append(f"note title {t!r} appears on pp. {sorted(ps)}; two chunks may explain the same thing. "
                            "Keep the first and point back to it from the later one.")
    Path(a.out).write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    (Path(a.out).parent / "critique_notes.json").write_text(json.dumps(crit_notes, indent=1, ensure_ascii=False))
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)
    if errors:
        print("\nERRORS:\n" + "\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print(f"merged {len(chunks)} chunks -> {a.out}: {len(spec['annotations'])} notes, "
          f"{sum(len(r.get('points', [])) for r in spec['page_review'])} key points; "
          f"{len(crit_notes)} critique notes -> critique_notes.json")


def cmd_verify(a):
    spec = json.loads(Path(a.spec).read_text())
    chunks = json.loads(Path(a.chunks).read_text())["chunks"]
    lines = read_lines(a.lines)
    have, dismissed = {}, {}
    for r in spec.get("page_review", []):
        p = int(r["page"])
        have.setdefault(p, []).extend(pt.get("quote", "") for pt in r.get("points", []))
        dismissed.setdefault(p, []).extend((nk.get("quote", ""), nk.get("why", "")) for nk in r.get("not_key", []))
    for an in spec.get("annotations", []):
        p = int(an["page"])
        if an.get("quote"):
            have.setdefault(p, []).append(an["quote"])
        rng = line_range(an.get("lines"))
        if rng and rng[1] - rng[0] + 1 <= ANCHOR_MAX_LINES:
            have.setdefault(p, []).append(" ".join(lines.get(p, {}).get(f"L{i}", "") for i in range(rng[0], rng[1] + 1)))
    rej = [x for x in spec.get("verify_rejections", [])
           if len(" ".join(str(x.get("why", "")).split()).split()) >= REJECT_MIN_WORDS]
    short_rej = len(spec.get("verify_rejections", [])) - len(rej)
    pdf = None
    if a.pdf:
        import pymupdf
        pdf = pymupdf.open(a.pdf)
    sent_cache = {}

    def same_sentence(p, q):
        if pdf is None:
            return False
        if p not in sent_cache:
            sent_cache[p] = page_sentences(pdf, p)
        return any(overlaps(q, s_) and any(overlaps(h, s_) for h in have.get(p, [])) for s_ in sent_cache[p])

    missing_files, misses, disagree, rows = [], [], [], []
    for c0, c1 in chunks:
        f = Path(a.verify) / f"part_{c0}-{c1}.json"
        if not f.exists():
            missing_files.append(f.name)
            continue
        vp = json.loads(f.read_text())
        per_page = {}
        for pt in vp.get("points", []):
            per_page.setdefault(int(pt["page"]), []).append(pt)
        for p in range(c0, c1 + 1):
            pts = per_page.get(p, [])
            hit = 0
            for pt in pts:
                q = pt.get("quote", "")
                if any(int(x["page"]) == p and overlaps(q, x.get("quote", "")) for x in rej):
                    hit += 1
                elif any(overlaps(q, dq) for dq, _w in dismissed.get(p, [])):
                    why = next(w for dq, w in dismissed[p] if overlaps(q, dq))
                    disagree.append((p, pt.get("what", "?"), q, why))
                elif any(overlaps(q, h) for h in have.get(p, [])) or same_sentence(p, q):
                    hit += 1
                else:
                    misses.append((p, pt.get("what", "?"), q))
            rows.append((p, len(pts), hit, len(have.get(p, []))))
    print("  page  verifier  covered  spec quotes")
    for p, nv, nh, ns in rows:
        print(f"  p{p:<4} {nv:>8} {nh:>8} {ns:>12}")
    bad = False
    if missing_files:
        bad = True
        print(f"\nno verifier output for: {', '.join(missing_files)}. Every chunk gets a blind verifier.",
              file=sys.stderr)
    if short_rej:
        print(f"\n{short_rej} verify_rejections ignored: 'why' needs {REJECT_MIN_WORDS}+ words saying why the "
              "verifier is wrong.", file=sys.stderr)
    if misses:
        bad = True
        print(f"\n{len(misses)} key point(s) the verifier found that the spec does not cover:", file=sys.stderr)
        for p, w, q in misses:
            print(f"  p{p} [{w}] {q[:150]}", file=sys.stderr)
    if disagree:
        bad = True
        print(f"\n{len(disagree)} key point(s) the verifier found that the spec dismissed as not_key:", file=sys.stderr)
        for p, w, q, why in disagree:
            print(f"  p{p} [{w}] {q[:120]}  (not_key because: {why[:80]})", file=sys.stderr)
    if misses or disagree:
        print("Fix each in its chunk's part file (as a key point, with a note if it is load-bearing), or add it to "
              "globals.json 'verify_rejections' with why the verifier is wrong; then re-merge.", file=sys.stderr)
    if bad:
        sys.exit(1)
    print(f"verify: {len(rows)} pages, every verifier key point is covered or rejected with a reason")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunks")
    c.add_argument("map"), c.add_argument("lines"), c.add_argument("out")
    c.add_argument("--max", type=int, default=5)
    c.add_argument("--min", type=int, default=3)
    m = sub.add_parser("merge")
    m.add_argument("map"), m.add_argument("chunks"), m.add_argument("parts"), m.add_argument("out")
    m.add_argument("--globals")
    m.add_argument("--pdf", help="the paper; re-runs each part through annotate.py --part before merging")
    v = sub.add_parser("verify")
    v.add_argument("spec"), v.add_argument("chunks"), v.add_argument("verify"), v.add_argument("lines")
    v.add_argument("--pdf", help="the paper; lets a verifier point match a spec quote in the same sentence")
    a = ap.parse_args()
    {"chunks": cmd_chunks, "merge": cmd_merge, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
