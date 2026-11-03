#!/usr/bin/env python3
"""Build a concise explainer PDF for one algorithm or central idea of a paper.

An explainer is not a second copy of the paper. It gives the reader what the
margins cannot: the big picture, the notation in one place, the mechanism as a
diagram, one example small enough to work by hand, an outline of the proof, and
a breakdown of only the two or three hardest steps. Errors, slips, and critique
stay in the annotated PDF's margins. Read references/walkthroughs.md first; it
holds the planning protocol and the evidence behind each rule.

Usage:
    python walkthrough.py paper.pdf explainer.json out.pdf [--render DIR] [--verify-runs]
                          [--review review.json]

Spec (HTML allowed in text fields: <b> <i> <sub> <sup> <br>; no LaTeX):
{
  "title": "SimGolf, explained",
  "pages": [3, 7],                 # the paper pages this explains (a pointer note goes on the first)
  "proof_pages": [27, 30],         # optional: where the proof it outlines lives
  "goal": "what the reader can do after reading (not printed; the reviewer checks it)",
  "idea": "the whole idea in one paragraph, 40-170 words, every term in it defined or obvious",
  "notation": [{"sym": "𝒯<sub>h</sub>[g]", "means": "..."}],          # 1-16 rows, only symbols used later
  "mechanism": {
     "title": "The algorithm (Algorithm 1, p. 6)",
     "diagram": {"nodes": [{"title": "1. Pick optimistic g", "text": "highest predicted value [Line 5]"}, ...],
                 "loop": true, "highlight": [2], "highlight_label": "the only step GOLF lacks",
                 # loop: true = arrow from the last box back to the first; [i, j] = from box i back to box j
                 "caption": "repeat for t = 1..N"},
     "steps": ["one sentence per step, 2-7 steps"],
     "why": "why the mechanism works, in plain terms"},
  "example": {"title": "...", "body": "<p>...</p> with small concrete numbers the reader can check by hand",
              "verify": {"cmd": "python3 demo.py", "out": "real output"}},   # optional; never printed
  "proof": {                                                        # optional
     "goal": "the theorem, by number, in plain words: what is assumed, what is guaranteed, the bound",
     "diagram": {"nodes": [...], "loop": false, "highlight": [0, 3], "highlight_label": "...", "caption": "..."},
     "steps": [{"name": "Valid sets", "claim": "...", "where": "Lemmas C.1, C.2"}],     # 2-6 steps
     "result": "how the steps combine into the theorem's bound"},
  "hard_steps": [{"title": "...", "where": "Lemma C.1", "intro": "...",
                  "moves": ["2-5 moves, each one inference"],
                  "check": {"q": "...", "answer": "15-70 words"}}],                 # 0-3; 1-3 if proof
  "checks": [{"q": "...", "answer": "..."}],     # optional, at most 2, printed after the example
  "skim": "what the reader can safely skim on these pages, and why",
  "links": [ {link with status as in annotate.py} ],   # optional, at most 3
  "long_because": "only if over the word budget: why this explainer needs more room (10+ words)"
}

Rules the build enforces (each from references/walkthroughs.md):
  - every section above is present, in this order, within its size limits;
  - reader-visible text is at most 2,600 words (warning past 2,400), or 4,000 with 'long_because';
  - no pipeline vocabulary (chunk, part file, annotator, flagged, verifier, ...), no errata talk (typo,
    misprint, should read, slip, ...): those belong in the margins;
  - no em dashes; no code is printed (a 'verify' command checks numbers, its code stays out of the PDF);
  - the example contains concrete numbers; every check answer is 15-70 words;
  - with --verify-runs, every 'verify.cmd' is re-run and must print exactly 'out';
  - with --review, the fresh-reader review must exist, list no undefined terms, say 'pass', and record the
    spec hash the build printed ('spec_sha'), so any edit after the review needs a new review.
"""
import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
import annotate as A  # noqa: E402  (fonts, link lint)

LINK_TAG = {"paper": "Paper", "article": "Article", "video": "Video", "rfc": "RFC", "standard": "Standard",
            "docs": "Docs", "book": "Book", "repo": "Code", "course": "Course", "blog": "Blog", "cwe": "CWE"}
WORDS_WARN, WORDS_MAX, WORDS_LONG = 2400, 2600, 4000
PIPELINE = re.compile(r"(?i)\b(chunks?|part[- ]files?|annotators?|annotated by|flagged|verifiers?|"
                      r"orchestrator|map\.json|spec\.json|subagents?|gate|globals\.json|critique[- ]notes?)\b")
ERRATA = re.compile(r"(?i)\b(typos?|misprints?|misprinted|should read|errat(?:um|a)|slips?|"
                    r"wrong (?:lemma|equation|line|index|constant)|as printed)\b")
READ_WPM = 180


def ws(t):
    return " ".join((t or "").split())


def plain(t):
    return ws(re.sub(r"<[^>]+>", " ", t or ""))


def nwords(t):
    return len(plain(t).split())


def reader_texts(spec):
    """Every reader-visible text field, with a label for error messages."""
    out = [("idea", spec.get("idea", ""))]
    for i, r in enumerate(spec.get("notation", [])):
        out += [(f"notation[{i}].sym", r.get("sym", "")), (f"notation[{i}].means", r.get("means", ""))]
    m = spec.get("mechanism") or {}
    out += [("mechanism.title", m.get("title", "")), ("mechanism.why", m.get("why", ""))]
    out += [(f"mechanism.steps[{i}]", s) for i, s in enumerate(m.get("steps", []))]
    for key, d in (("mechanism.diagram", m.get("diagram")), ("proof.diagram", (spec.get("proof") or {}).get("diagram"))):
        if d:
            out += [(f"{key}.node[{i}]", n.get("title", "") + " " + n.get("text", "")) for i, n in enumerate(d.get("nodes", []))]
            out += [(f"{key}.caption", d.get("caption", "")), (f"{key}.highlight_label", d.get("highlight_label", ""))]
    e = spec.get("example") or {}
    out += [("example.title", e.get("title", "")), ("example.body", e.get("body", ""))]
    p = spec.get("proof") or {}
    out += [("proof.goal", p.get("goal", "")), ("proof.result", p.get("result", ""))]
    for i, s in enumerate(p.get("steps", [])):
        out += [(f"proof.steps[{i}]", " ".join(str(s.get(k, "")) for k in ("name", "claim", "where")))]
    for i, h in enumerate(spec.get("hard_steps", [])):
        out += [(f"hard_steps[{i}].title", h.get("title", "")), (f"hard_steps[{i}].intro", h.get("intro", ""))]
        out += [(f"hard_steps[{i}].moves[{j}]", mv) for j, mv in enumerate(h.get("moves", []))]
        ck = h.get("check") or {}
        out += [(f"hard_steps[{i}].check.q", ck.get("q", "")), (f"hard_steps[{i}].check.answer", ck.get("answer", ""))]
    for i, ck in enumerate(spec.get("checks", [])):
        out += [(f"checks[{i}].q", ck.get("q", "")), (f"checks[{i}].answer", ck.get("answer", ""))]
    out += [("skim", spec.get("skim", ""))]
    return out


def check_diagram(d, tag, errors, n_min=2, n_max=6):
    if not d or not d.get("nodes"):
        errors.append(f"{tag}: a diagram is required (nodes, plus loop/highlight/caption). One picture of how the "
                      "pieces connect is worth more than paragraphs describing it.")
        return
    nodes = d["nodes"]
    if not (n_min <= len(nodes) <= n_max):
        errors.append(f"{tag}: {len(nodes)} nodes; use {n_min}-{n_max}. Group finer steps into one box.")
    for i, n in enumerate(nodes):
        if not n.get("title"):
            errors.append(f"{tag}.node[{i}]: needs a 'title'")
        if nwords(n.get("text", "")) > 12:
            errors.append(f"{tag}.node[{i}]: box text is {nwords(n['text'])} words; keep it to 12 or fewer.")
        if len(plain(n.get("title", ""))) > 26:
            errors.append(f"{tag}.node[{i}]: title is too long for a box (<= 26 characters).")
    reg, bold = font_paths()
    if reg:
        fr, fb = pymupdf.Font(fontfile=reg), pymupdf.Font(fontfile=bold)
        for i, n in enumerate(nodes):
            for txt, f in ((plain(n.get("title", "")), fb), (plain(n.get("text", "")), fr)):
                miss = sorted({c for c in txt if not f.has_glyph(ord(c))})
                if miss:
                    errors.append(f"{tag}.node[{i}]: the diagram font has no glyph for {miss}; use plain words "
                                  "or common symbols in boxes.")
    lp = d.get("loop")
    if lp not in (None, True, False) and not (isinstance(lp, list) and len(lp) == 2
                                              and all(isinstance(v, int) for v in lp)
                                              and 0 <= lp[1] < lp[0] < len(nodes)):
        errors.append(f"{tag}.loop: use true (arrow from the last box back to the first) or [from, to], two box "
                      "indices with from > to (e.g. [2, 0] for a failed test that restarts at box 0).")
    for h in d.get("highlight", []):
        if not (isinstance(h, int) and 0 <= h < len(nodes)):
            errors.append(f"{tag}: highlight index {h!r} is not a node index")


def check_answer(ck, tag, errors):
    if not ck.get("q") or not ck.get("answer"):
        errors.append(f"{tag}: needs 'q' and 'answer'")
        return
    n = nwords(ck["answer"])
    if not (15 <= n <= 70):
        errors.append(f"{tag}: answer is {n} words; give 15-70 words that say why, not just what.")


def validate(spec, n_pages, spec_dir=None, verify_runs=False, review=None):
    errors, warnings = [], []
    for f in ("title", "pages", "goal", "idea", "notation", "mechanism", "example", "skim"):
        if not spec.get(f):
            errors.append(f"missing '{f}'")
    if errors:
        return errors, warnings
    for key in ("pages", "proof_pages"):
        if spec.get(key):
            try:
                a, b = int(spec[key][0]), int(spec[key][1])
                assert 1 <= a <= b <= n_pages
            except (TypeError, ValueError, IndexError, AssertionError):
                errors.append(f"'{key}' must be [first, last] within the paper's {n_pages} pages")
    if nwords(spec["goal"]) < 10:
        errors.append("'goal' must say what the reader can do after reading (10+ words); the reviewer tests it.")
    n = nwords(spec["idea"])
    if not (40 <= n <= 170):
        errors.append(f"'idea' is {n} words; the whole idea fits in one paragraph of 40-170 words.")
    nt = spec["notation"]
    if not (1 <= len(nt) <= 16):
        errors.append(f"'notation' has {len(nt)} rows; list 1-16, only symbols the explainer uses.")
    body_plain = " ".join(plain(t) for k, t in reader_texts(spec) if not k.startswith("notation"))
    for i, r in enumerate(nt):
        if not r.get("sym") or not r.get("means"):
            errors.append(f"notation[{i}]: needs 'sym' and 'means'")
            continue
        sym = re.split(r"[\[(,]", plain(r["sym"]).replace(" ", ""))[0]
        if sym and sym not in body_plain.replace(" ", ""):
            warnings.append(f"notation[{i}] {sym!r} is never used below; drop rows the reader will not need.")
    m = spec["mechanism"]
    if not m.get("title"):
        errors.append("mechanism: needs a 'title' that names where it lives in the paper (e.g. 'Algorithm 1, p. 6')")
    check_diagram(m.get("diagram"), "mechanism.diagram", errors)
    steps = m.get("steps") or []
    if not (2 <= len(steps) <= 7):
        errors.append(f"mechanism.steps: {len(steps)} steps; describe the mechanism in 2-7.")
    for i, s in enumerate(steps):
        if nwords(s) > 60:
            errors.append(f"mechanism.steps[{i}]: {nwords(s)} words; one step is one or two sentences (<= 60 words).")
    if not m.get("why"):
        errors.append("mechanism.why: say in plain terms why the mechanism works")
    ex = spec["example"]
    if not ex.get("title") or not ex.get("body"):
        errors.append("example: needs 'title' and 'body'")
    elif len(re.findall(r"\d+(?:\.\d+)?", plain(ex["body"]))) < 3:
        errors.append("example: give concrete small numbers the reader can check by hand (at least 3 numbers).")
    if ex.get("verify"):
        v = ex["verify"]
        if not v.get("cmd") or not v.get("out"):
            errors.append("example.verify: needs 'cmd' and real 'out'")
        elif verify_runs:
            try:
                r = subprocess.run(v["cmd"], shell=True, cwd=spec_dir, capture_output=True, text=True, timeout=120)
                if ws(r.stdout) != ws(v["out"]):
                    errors.append(f"example.verify: `{v['cmd']}` printed something else:\n{r.stdout[:500]}")
            except subprocess.TimeoutExpired:
                errors.append(f"example.verify: `{v['cmd']}` timed out")
    for i, ck in enumerate(spec.get("checks", [])):
        check_answer(ck, f"checks[{i}]", errors)
    if len(spec.get("checks", [])) > 2:
        errors.append("'checks': at most 2 after the example; put the rest on the hard steps.")
    pr = spec.get("proof")
    hs = spec.get("hard_steps", [])
    if pr:
        for f in ("goal", "steps", "result"):
            if not pr.get(f):
                errors.append(f"proof: missing '{f}'")
        check_diagram(pr.get("diagram"), "proof.diagram", errors)
        if not (2 <= len(pr.get("steps", [])) <= 6):
            errors.append(f"proof.steps: {len(pr.get('steps', []))}; outline the proof in 2-6 steps.")
        for i, s in enumerate(pr.get("steps", [])):
            if not (s.get("name") and s.get("claim") and s.get("where")):
                errors.append(f"proof.steps[{i}]: needs name, claim, and where (the lemma or page)")
            elif nwords(s["claim"]) > 70:
                errors.append(f"proof.steps[{i}]: claim is {nwords(s['claim'])} words; one step is a sentence or two.")
        if not (1 <= len(hs) <= 3):
            errors.append(f"hard_steps: {len(hs)}; break down the 1-3 steps that are both load-bearing and hard to "
                          "see. Routine steps get their one line in the proof outline.")
    elif len(hs) > 3:
        errors.append("hard_steps: at most 3")
    for i, h in enumerate(hs):
        tag = f"hard_steps[{i}]"
        for f in ("title", "where", "moves", "check"):
            if not h.get(f):
                errors.append(f"{tag}: missing '{f}'")
        if not (2 <= len(h.get("moves", [])) <= 5):
            errors.append(f"{tag}: {len(h.get('moves', []))} moves; break the step into 2-5 inferences.")
        for j, mv in enumerate(h.get("moves", [])):
            if nwords(mv) > 90:
                errors.append(f"{tag}.moves[{j}]: {nwords(mv)} words; one inference per move (<= 90 words).")
        if h.get("check"):
            check_answer(h["check"], f"{tag}.check", errors)
    if len(spec.get("links", [])) > 3:
        errors.append("'links': at most 3, the ones that teach this idea best")
    for i, lk in enumerate(spec.get("links", [])):
        if lk.get("status") not in ("opened", "search", "expected"):
            errors.append(f"links[{i}]: status must be opened, search, or expected (see links.md)")
        bad = A.lint_link(lk.get("url", ""))
        if bad and bad[0]:
            errors.append(f"links[{i}]: {bad[0]}")
    # language: what the reader sees
    total = 0
    for key, t in reader_texts(spec):
        total += nwords(t)
        if "—" in (t or ""):
            errors.append(f"{key}: contains an em dash; use a comma, colon, period, or parentheses.")
        mp = PIPELINE.search(plain(t))
        if mp:
            errors.append(f"{key}: {mp.group(0)!r} is production vocabulary the reader never sees; remove it.")
        me = ERRATA.search(plain(t))
        if me:
            errors.append(f"{key}: {me.group(0)!r} reports a paper error. Errors live in the margin notes; the "
                          "explainer explains the intended argument. Drop it, or say the idea plainly.")
        if re.search(r"(?m)^\s*(def |import |for .+:$|print\()", t or ""):
            errors.append(f"{key}: looks like code. No code is printed; state what the computation shows.")
    limit = WORDS_LONG if len(ws(spec.get("long_because", "")).split()) >= 10 else WORDS_MAX
    if total > limit:
        errors.append(f"{total} reader-visible words; the limit is {limit}. Cut routine steps, background the reader "
                      "already has, and anything the margins already say. Explain less, better.")
    elif total > WORDS_WARN:
        warnings.append(f"{total} words; aim for under {WORDS_WARN}.")
    if review is not None:
        for f in ("main_idea_after_page_1", "undefined_terms", "restates_paper", "unclear", "verdict"):
            if f not in review:
                errors.append(f"review: missing '{f}' (see references/walkthroughs.md, 'Reader review')")
        if review.get("undefined_terms"):
            errors.append(f"review: undefined terms {review['undefined_terms']}; define each at first use or in the "
                          "notation table.")
        if review.get("verdict") != "pass":
            errors.append(f"review: verdict is {review.get('verdict')!r}; fix what it lists, then review again.")
        elif review.get("spec_sha") != A.spec_sha(spec):
            errors.append(f"review: it reviewed spec {review.get('spec_sha')!r}, but the spec is now "
                          f"{A.spec_sha(spec)!r}. The spec changed after the review; review the new build again.")
    spec["_words"] = total
    return errors, warnings


# ---------------------------------------------------------------- rendering
def font_paths():
    for path in A.FONT_CANDIDATES:
        p = Path(path)
        if p.exists():
            bold = p.parent / "DejaVuSansCondensed-Bold.ttf"
            return str(p), str(bold) if bold.exists() else str(p)
    return None, None


def draw_diagram(page, rect, d):
    """Boxes left to right with arrows; optional loop-back arrow and highlighted boxes."""
    reg, bold = font_paths()
    nodes = d["nodes"]
    n = len(nodes)
    gap = 18
    top = rect.y0 + (14 if d.get("highlight_label") else 2)
    bh = 66
    bw = (rect.width - gap * (n - 1)) / n
    hl = set(d.get("highlight", []))
    sh = page.new_shape()
    centers = []
    for i, node in enumerate(nodes):
        x0 = rect.x0 + i * (bw + gap)
        r = pymupdf.Rect(x0, top, x0 + bw, top + bh)
        fill, stroke = ((0.99, 0.94, 0.88), (0.75, 0.48, 0.17)) if i in hl else ((0.93, 0.95, 0.98), (0.23, 0.44, 0.71))
        sh.draw_rect(r)
        sh.finish(color=stroke, fill=fill, width=0.8)
        centers.append(r)
        if i:
            y = top + bh / 2
            sh.draw_line((x0 - gap + 1, y), (x0 - 3, y))
            sh.draw_polyline([(x0 - 7, y - 3), (x0 - 2, y), (x0 - 7, y + 3)])
            sh.finish(color=(0.33, 0.33, 0.33), width=0.8)
    if d.get("loop") and n > 1:
        y = top + bh + 12
        lp = d["loop"]
        a, b = (centers[lp[0]], centers[lp[1]]) if isinstance(lp, list) else (centers[-1], centers[0])
        sh.draw_polyline([(a.x0 + a.width / 2, top + bh), (a.x0 + a.width / 2, y), (b.x0 + b.width / 2, y),
                          (b.x0 + b.width / 2, top + bh + 3)])
        sh.finish(color=(0.33, 0.33, 0.33), width=0.8)
        cx = b.x0 + b.width / 2
        sh.draw_polyline([(cx - 3, top + bh + 7), (cx, top + bh + 2), (cx + 3, top + bh + 7)])
        sh.finish(color=(0.33, 0.33, 0.33), width=0.8)
    sh.commit()
    kw = {"fontname": "dvr", "fontfile": reg} if reg else {"fontname": "helv"}
    kb = {"fontname": "dvb", "fontfile": bold} if bold else {"fontname": "hebo"}
    for i, (node, r) in enumerate(zip(nodes, centers)):
        inner = pymupdf.Rect(r.x0 + 3, r.y0 + 5, r.x1 - 3, r.y1 - 2)
        for fs in (8.5, 8, 7.5, 7, 6.5):
            tmp = pymupdf.open()
            tp = tmp.new_page(width=page.rect.width, height=page.rect.height)
            ok1 = tp.insert_textbox(pymupdf.Rect(inner.x0, inner.y0, inner.x1, inner.y0 + 24), plain(node["title"]),
                                    fontsize=fs, align=1, **kb)
            ok2 = tp.insert_textbox(pymupdf.Rect(inner.x0, inner.y0 + 22, inner.x1, inner.y1), plain(node.get("text", "")),
                                    fontsize=fs - 0.5, align=1, **kw)
            tmp.close()
            if ok1 >= 0 and ok2 >= 0:
                break
        page.insert_textbox(pymupdf.Rect(inner.x0, inner.y0, inner.x1, inner.y0 + 24), plain(node["title"]),
                            fontsize=fs, align=1, **kb)
        page.insert_textbox(pymupdf.Rect(inner.x0, inner.y0 + 22, inner.x1, inner.y1), plain(node.get("text", "")),
                            fontsize=fs - 0.5, align=1, color=(0.15, 0.15, 0.15), **kw)
    if d.get("highlight_label") and hl:
        xs = [centers[i] for i in sorted(hl)]
        page.insert_textbox(pymupdf.Rect(xs[0].x0 - 30, rect.y0, xs[-1].x1 + 30, rect.y0 + 12),
                            plain(d["highlight_label"]), fontsize=7.5, align=1, color=(0.75, 0.48, 0.17), **kw)
    if d.get("caption"):
        y = top + bh + (16 if d.get("loop") else 6)
        page.insert_textbox(pymupdf.Rect(rect.x0, y, rect.x1, y + 12), plain(d["caption"]), fontsize=7.8, align=1,
                            color=(0.25, 0.25, 0.25), **kw)


def diagram_height(d):
    return (14 if d.get("highlight_label") else 2) + 66 + (16 if d.get("loop") else 6) + (14 if d.get("caption") else 0) + 6


def build(spec, out, fs=10.3, lh=1.42):
    archive, base = A.font_setup()
    doc = pymupdf.open()
    W, H, M = 612, 792, 56
    css = base + """
* {font-size: %spt; line-height: %s; color: #1d1d1d;}
h1 {font-size: 18pt; margin: 0 0 2pt 0;}
h2 {font-size: 12.5pt; margin: 14pt 0 3pt 0; color: #111;}
h3 {font-size: 11pt; margin: 10pt 0 2pt 0;}
p {margin: 0 0 5pt 0;} li {margin-bottom: 3pt;} ol.mv {margin-top: 0; margin-bottom: 0;}
.sub {color: #5a5a5a; font-size: 9pt;}
.idea {background: #f1f5fb; border-left: 3pt solid #3b6fb6; padding: 6pt 8pt;}
.ex {border-left: 2.5pt solid #b9b9b9; padding: 1pt 8pt;} .ex p {margin: 0;}
.q {background: #f5f5ee; border-left: 3pt solid #8a8a4a; padding: 5pt 8pt;}
.ans {color: #555; font-size: 9pt;}
.ref {color: #666; font-size: 9pt; font-weight: normal;}
.nt {font-size: 9.6pt; margin: 0 0 3pt 0; padding-top: 2pt; border-top: 0.5pt solid #dcdcdc;}
a {color: #1a4f8a;}
""" % (fs, lh)
    B = []   # blocks: ("html", s) | ("rot", s) | ("diagram", d) | ("head", s)
    words = spec.get("_words", 0)
    pg = spec["pages"]
    loc = f"Paper pp. {pg[0]}&ndash;{pg[1]}" if pg[0] != pg[1] else f"Paper p. {pg[0]}"
    if spec.get("proof_pages"):
        pp = spec["proof_pages"]
        loc += f"; proof pp. {pp[0]}&ndash;{pp[1]}"
    B.append(("html", f"<h1>{html.escape(spec['title'])}</h1><p class='sub'>{loc}. About "
                      f"{max(3, round(words / READ_WPM))} minutes. Read it beside the annotated copy of the "
                      "paper that comes with it; mistakes in the paper itself are marked in that copy's margins.</p>"))
    B.append(("html", f"<div class='idea'><b>The idea.</b> {spec['idea']}</div>"))
    rows = "".join(f"<p class='nt'><b>{r['sym']}</b>: {r['means']}</p>" for r in spec["notation"])
    B.append(("head", "<h2>Notation</h2>"))
    B.append(("html", f"<div>{rows}</div>"))
    m = spec["mechanism"]
    B.append(("head", f"<h2>{m['title']}</h2>"))
    B.append(("diagram", m["diagram"]))
    B.append(("html", "<ol>" + "".join(f"<li>{s}</li>" for s in m["steps"]) + "</ol>"))
    B.append(("html", f"<p><b>Why it works.</b> {m['why']}</p>"))
    ex = spec["example"]
    B.append(("head", f"<h2>{ex['title']}</h2>"))
    # one block per top-level paragraph, so a long example can break across pages instead of leaving a gap
    paras = [p for p in re.findall(r"(?s)<p\b.*?</p>|<(?:ul|ol|table)\b.*?</(?:ul|ol|table)>", ex["body"])]
    if not paras or len("".join(paras)) < 0.9 * len(ex["body"].strip()):
        paras = [ex["body"]]
    for p in paras:
        B.append(("html", f"<div class='ex'>{p}</div>"))
    qn = 0

    def add_check(ck):
        nonlocal qn
        qn += 1
        B.append(("html", f"<div class='q'><b>Check {qn}.</b> {ck['q']}</div>"))
        B.append(("rot", f"<p class='ans'><b>Answer {qn}.</b> {ck['answer']}</p>"))

    for ck in spec.get("checks", []):
        add_check(ck)
    pr = spec.get("proof")
    if pr:
        ref = f" <span class='ref'>(pp. {spec['proof_pages'][0]}&ndash;{spec['proof_pages'][1]})</span>" \
            if spec.get("proof_pages") else ""
        B.append(("head", f"<h2>How the proof works{ref}</h2>"))
        B.append(("html", f"<p>{pr['goal']}</p>"))
        B.append(("diagram", pr["diagram"]))
        B.append(("html", "<ol>" + "".join(f"<li><b>{s['name']}</b> <span class='ref'>({s['where']})</span>. "
                                          f"{s['claim']}</li>" for s in pr["steps"]) + "</ol>"))
        B.append(("html", f"<p><b>Result.</b> {pr['result']}</p>"))
    for i, h in enumerate(spec.get("hard_steps", []), 1):
        B.append(("head", f"<h3>Hard step {i}: {h['title']} <span class='ref'>({h['where']})</span></h3>"))
        intro = f"<p>{h['intro']}</p>" if h.get("intro") else ""
        # one block per move (the first stays with the intro), so a long hard step can break across pages
        for k, mv in enumerate(h["moves"], 1):
            B.append(("html", (intro if k == 1 else "") + f"<ol class='mv' start='{k}'><li>{mv}</li></ol>"))
        add_check(h["check"])
    B.append(("head", "<h2>Safe to skim</h2>"))
    B.append(("html", f"<p>{spec['skim']}</p>"))
    if spec.get("links"):
        items = []
        for lk in spec["links"]:
            tag = LINK_TAG.get(lk.get("type"), "Link") + (" · not opened" if lk.get("status") in ("search", "expected") else "")
            items.append(f"<li><b>{tag}.</b> <a href='{html.escape(lk['url'])}'>{html.escape(lk.get('title', lk['url']))}</a>"
                         + (f": {lk['fit']}" if lk.get("fit") else "") + "</li>")
        B.append(("head", "<h2>To go further</h2>"))
        B.append(("html", "<ul>" + "".join(items) + "</ul>"))

    area = pymupdf.Rect(M, M, W - M, H - M)

    def height(b):
        kind, x = b
        return diagram_height(x) if kind == "diagram" else A.measure(x, area.width, css, archive) + 3

    page = doc.new_page(width=W, height=H)
    y = area.y0
    for i, b in enumerate(B):
        kind, x = b
        h = height(b)
        need = h
        nxt = B[i + 1] if i + 1 < len(B) else None
        if kind == "head" and nxt:
            # keep a heading with what follows: the next block; if a diagram comes within the next two blocks,
            # everything through the diagram plus the first block that explains it
            grp = [nxt]
            j = next((k for k in (i + 1, i + 2) if k < len(B) and B[k][0] == "diagram"), None)
            if j is not None:
                grp = B[i + 1:j + 2]
            need += min(sum(height(g) for g in grp), area.height * 0.8)
        elif kind == "diagram" and nxt and nxt[0] in ("html", "head"):
            need += min(height(nxt), area.height * 0.4)   # a diagram never ends a page alone
        if nxt and nxt[0] == "rot":           # keep a question with its upside-down answer
            need += height(nxt)
        if kind != "rot" and y + need > area.y1 and y > area.y0:
            page = doc.new_page(width=W, height=H)
            y = area.y0
        if kind == "diagram":
            draw_diagram(page, pymupdf.Rect(area.x0, y, area.x1, y + h), x)
        elif kind == "rot":
            page.insert_htmlbox(pymupdf.Rect(area.x0 + 8, y, area.x1, y + h), x, css=css, archive=archive,
                                scale_low=1, rotate=180)
        else:
            page.insert_htmlbox(pymupdf.Rect(area.x0, y, area.x1, min(y + h + 40, area.y1)), x, css=css,
                                archive=archive, scale_low=1)
        y += h
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    return len(doc), (y - area.y0) / area.height


def build_fitted(spec, out):
    """Build; if the last page would hold under a fifth of a page, retry with slightly tighter type so the
    tail joins the page before (a near-empty last page reads as a mistake). Falls back to the normal size."""
    n0, fill = build(spec, out)
    if n0 == 1 or fill >= 0.2:
        return n0
    for fs, lh in ((10.1, 1.38), (9.9, 1.34)):
        n, fill = build(spec, out, fs, lh)
        if n < n0 or fill >= 0.2:
            return n
    return build(spec, out)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--render")
    ap.add_argument("--verify-runs", action="store_true", help="re-run example.verify.cmd and compare to its out")
    ap.add_argument("--review", help="the fresh-reader review JSON (references/walkthroughs.md, 'Reader review')")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    n_pages = len(pymupdf.open(args.pdf))
    review = json.loads(Path(args.review).read_text()) if args.review else None
    errors, warnings = validate(spec, n_pages, spec_dir=str(Path(args.spec).parent.resolve()),
                                verify_runs=args.verify_runs, review=review)
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)
    if errors:
        print("\nERRORS (no PDF written):\n- " + "\n- ".join(errors), file=sys.stderr)
        sys.exit(1)
    sha = A.spec_sha(spec)
    n = build_fitted(spec, args.out)
    print(f"wrote {args.out}: {spec['_words']} words, {n} pages, {len(spec.get('hard_steps', []))} hard steps, "
          f"spec {sha}" + ("" if review else f" (no --review yet: required before delivery; the reviewer records "
                                              f"\"spec_sha\": \"{sha}\")"))
    if args.render:
        rd = Path(args.render)
        rd.mkdir(parents=True, exist_ok=True)
        for old in rd.glob("wt_*.png"):
            old.unlink()
        for i, pg in enumerate(pymupdf.open(args.out), 1):
            pg.get_pixmap(dpi=96).save(rd / f"wt_{i}.png")


if __name__ == "__main__":
    main()
