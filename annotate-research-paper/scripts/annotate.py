#!/usr/bin/env python3
"""Apply an annotation spec to a PDF: highlights, margin bars, numbered notes
in an added right-hand gutter (check questions carry their answer upside down
right under the question), a reading-guide page, a critique page, and a
key-references page that says exactly how each link was checked.

Usage:
    python annotate.py input.pdf spec.json output.pdf [--render DIR] [--gutter 240]
                       [--link-report link_report.json]

references/spec-format.md documents every spec field. Pages are 1-based physical pages of the
INPUT pdf; line IDs come from extract.py. Exit code 1 on any rule violation
(unresolved anchor, uncovered figure/table/equation, restated note, figure note
not in Metric/Baseline/Takeaway form, verdict that contradicts its note, link
without an honest check status, existence claim without a recorded check, ...).
"""
import argparse
import html
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from extract import page_lines, text_column  # noqa: E402

KINDS = {
    # kind: (label, highlight rgb, accent rgb, box tint rgb)
    "key":       ("Key",       (1.00, 0.90, 0.35), (0.85, 0.60, 0.00), (1.00, 0.97, 0.86)),
    "intuition": ("Intuition", (0.62, 0.84, 1.00), (0.15, 0.45, 0.80), (0.92, 0.96, 1.00)),
    "subtle":    ("Watch out", (1.00, 0.70, 0.70), (0.80, 0.20, 0.20), (1.00, 0.93, 0.93)),
    "check":     ("Check",     (0.72, 0.92, 0.72), (0.20, 0.55, 0.25), (0.93, 0.98, 0.93)),
}
CRITIQUE_DOMAINS = {"security", "ml", "rl", "ir", "systems", "general", "article"}
CRITIQUE_VERDICTS = {"ok", "concern", "flaw", "cant-tell", "not-applicable"}
LINK_TYPES = {"paper", "article", "video", "rfc", "standard", "docs", "book", "repo", "course", "blog", "cwe"}
# How a link was checked. The model may only record what a tool call actually
# returned this session; "script" is set by annotate.py from check_links.py.
LINK_STATUS = {
    "opened":   "Opened during research; page title seen: ",
    "search":   "Seen only in search results (not opened); listed as: ",
    "expected": "Expected open access, not opened: ",
}
SCRIPT_STATUS = "Opened by the build script (HTTP {http}); page title: "
NEEDS_CHOICE = {"video", "course"}      # quality varies most: record fit + alternatives
# A quoted phrase attributed to the paper must be found in the paper; outside
# facts must be labeled as not-from-the-paper; an inference must be labeled.
QUOTE_RE = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{12,240})[\"\u201d]"
                      r"|\u2018([^\u2018\u2019]{8,240})\u2019"
                      r"|(?:(?<=^)|(?<=[\s(\[]))'([^'\n]{8,240}?)'(?![A-Za-z])")
PAPER_ATTR = re.compile(r"(?i)\b(the paper|the authors?|they (?:say|state|claim|report|write|note)|"
                        r"states?|claims?|reports?|writes?|p\.\s*\d|section\s+\d|abstract|"
                        r"\bfig(?:ure)?\.?\s*\d|\btable\s*\d)\b")
OUTSIDE_LAB = re.compile(r"(?i)\b(background \(not in the paper\)|not in the paper|outside the paper|"
                         r"not stated in the paper|the paper does not (?:say|mention|define)|"
                         r"not in this paper)\b")
INFER_LAB = re.compile(r"(?i)\b(inference|inferred|reading between|my read|not spelled out|"
                       r"the paper does not say this directly|implied|not stated outright)\b")
# Phrases that assert the paper's internal method/structure as fact (candidates
# for being the model's inference rather than the paper's words).
METHOD_ASSERT = re.compile(r"(?i)\b(holding .* fixed|assumes? that|the method (?:is|was)|"
                           r"computed by|calculated by|they (?:hold|held|fix|fixed|replay|replayed))\b")
# Note parts. Every non-check note needs an implication part; descriptive
# parts may not dominate the note or merely echo the annotated text.
DESCRIPTIVE = {"means", "does", "reads as", "shows", "is", "definition", "what it is", "layout"}
IMPLICATION = {"why it matters", "matters because", "so what", "takeaway", "point", "supports?",
               "implication", "trade-off", "cost", "limit", "if it fails", "load-bearing?",
               "actually", "why this form", "why it works", "why here", "claim", "scope",
               "assumes", "evidence", "looks like", "hint"}
FIG_PARTS = {"data": ["metric", "baseline", "takeaway"], "diagram": ["point", "why it matters"]}
FIGTAB_RE = re.compile(r"\b(Fig(?:ure)?\.?|Table)\s*\d", re.I)
RESTATE_MAX = 0.70    # share of a descriptive part's content words copied from the anchor text
DESCRIPTIVE_MAX = 0.55  # share of a note's words spent in descriptive parts
STOP = set("""a an the and or of to in on for with by is are was were be been it its this that these
those as at from into than then so such can may might will would should could not no but if
which who whom what when where how why their there they them our we you your he she his her
has have had do does did also only just more most very much many one two three each any all""".split())
# "Never guess": statements that something is fake / nonexistent / unverifiable
# need a recorded check ('checked': what was searched and what came back).
EXISTENCE_RE = re.compile(r"(?i)\b(fictional|hypothetical|made[- ]up|does not exist|doesn't exist|"
                          r"no public record|not a real|never happened|probably (?:fake|fictional|not real))\b")
# Critique verdict/language consistency.
ABSENCE_RE = re.compile(r"(?i)\b(not report|doesn'?t report|does not report|no (?:variance|trials?|baselines?|"
                        r"statistical|seeds?|error bars|data|ablation|numbers?)|not (?:given|provided|stated|"
                        r"measured|reported)|without (?:evidence|data|numbers)|single[- ]run|none of the)\b")
NEGATIVE_RE = re.compile(r"(?i)\b(fails?|failed|lacks?|missing|no baseline|overclaims?|unsupported|"
                         r"not supported|contradicts?|cherry[- ]pick|hearsay|unverified)\b")

# Link lint (from the sourcing research). REJECT = fatal; CHECK = warn (likely paywall).
BAD_LINK = {
    "search_page":      r"^https?://(www\.)?(google\.[a-z.]+/search|bing\.com/search|duckduckgo\.com/\?|search\.yahoo\.com/)",
    "google_scholar":   r"^https?://scholar\.google\.[a-z.]+/",
    "site_search":      r"^https?://(arxiv\.org/(search|find)|dblp\.org/search|(www\.)?semanticscholar\.org/search|(www\.)?youtube\.com/results)",
    "shortener":        r"^https?://(bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|is\.gd|rb\.gy|shorturl\.at|lnkd\.in|tiny\.cc|cutt\.ly)/",
    "login_mirror":     r"^https?://(www\.)?(researchgate\.net|academia\.edu|scribd\.com)/",
    "arxiv_mirror":     r"^https?://(export\.arxiv\.org|(www\.)?alphaxiv\.org|huggingface\.co/papers|(www\.)?arxiv-vanity\.com)/",
    "proxy_session":    r"(ezproxy|\.idm\.oclc\.org|[?&]casa_token=|[?&](sessionid|jsessionid|sid)=)",
    "pirate_copy":      r"(sci-hub\.|libgen\.|z-lib\.)",
    "acm_abs":          r"^https?://dl\.acm\.org/doi/abs/",
    "sciencedirect_abs": r"^https?://(www\.)?sciencedirect\.com/science/article/abs/",
    "openreview_rev":   r"^https?://openreview\.net/references/pdf\?",
}
PAYWALL_LINK = {
    "ieeexplore":    r"^https?://(www\.)?ieeexplore\.ieee\.org/(abstract/)?document/\d+",
    "springer":      r"^https?://link\.springer\.com/(article|chapter|book)/",
    "sciencedirect": r"^https?://(www\.)?sciencedirect\.com/science/article/pii/",
    "wiley":         r"^https?://(www\.)?onlinelibrary\.wiley\.com/doi/",
    "tandf":         r"^https?://(www\.)?tandfonline\.com/doi/",
    "doi_resolver":  r"^https?://(dx\.)?doi\.org/",
    "acm_landing":   r"^https?://dl\.acm\.org/doi/(?!pdf/)",
}


def spec_sha(spec):
    """Short content hash of an explainer spec (keys starting with '_' ignored). A reader review records the
    hash it reviewed, so a spec edited after its review fails the build until it is reviewed again."""
    import hashlib
    clean = {k: v for k, v in spec.items() if not str(k).startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def lint_link(url):
    """Return (fatal_reason_or_None, warn_reason_or_None)."""
    if not url or not isinstance(url, str):
        return "empty url", None
    if not url.startswith("https://"):
        return (None, "http (not https); use https") if url.startswith("http://") else ("url must be a full https link", None)
    for name, pat in BAD_LINK.items():
        if re.search(pat, url, re.I):
            return f"bad-link pattern '{name}'", None
    for name, pat in PAYWALL_LINK.items():
        if re.search(pat, url, re.I):
            return None, f"likely paywalled/abstract ('{name}'); prefer an open full-text link"
    return None, None


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/TTF/DejaVuSansCondensed.ttf",
    "/Library/Fonts/DejaVuSansCondensed.ttf",
]
FONT_SIZES = [8.5]  # gutter font; readable at 100% zoom
MAX_FILL = 1.0  # use the whole gutter before adding a notes page
SHORT_LIMIT = 45  # notes longer than this (words) need a 'short' fallback, used only if the page overflows
BOX_GAP = 6
PAD = 4
TOP_MARGIN = 36
BOTTOM_MARGIN = 36


# ---------------------------------------------------------------- fonts / html
def font_setup():
    for path in FONT_CANDIDATES:
        p = Path(path)
        if p.exists():
            d = p.parent
            faces = [("DejaVuSansCondensed.ttf", "normal", "normal"),
                     ("DejaVuSansCondensed-Bold.ttf", "bold", "normal"),
                     ("DejaVuSansCondensed-Oblique.ttf", "normal", "italic"),
                     ("DejaVuSansCondensed-BoldOblique.ttf", "bold", "italic")]
            rules = [f"@font-face {{font-family: nf; src: url({f}); font-weight: {w}; font-style: {s};}}"
                     for f, w, s in faces if (d / f).exists()]
            return pymupdf.Archive(str(d)), "\n".join(rules) + "\n* {font-family: nf;}"
    print("WARNING: DejaVu Sans Condensed not found; math symbols in notes may render as boxes. "
          "Install fonts-dejavu-core.", file=sys.stderr)
    return None, "* {font-family: sans-serif;}"


def css_for(size, base_css):
    return base_css + f"""
* {{font-size: {size}pt; line-height: 1.25; color: #222;}}
p {{margin: 0 0 2pt 0;}}
.lab {{font-weight: bold;}}
"""


def note_html(num, kind, title, body, check_index=None):
    label = KINDS[kind][0]
    accent = "#%02x%02x%02x" % tuple(int(c * 255) for c in KINDS[kind][2])
    head = f"<span class='lab' style='color:{accent}'>{num} · {label}"
    if kind == "check" and check_index is not None:
        head += f" Q{check_index}"
    if title:
        head += f": {title}"
    head += "</span>"
    return f"<p>{head}</p><p>{split_parts(body)}</p>"


ANSWER_LABEL = ("<p style='color:#2f6b34; font-size:7pt'><i>Answer below, upside down: answer first, "
                "then turn the page.</i></p>")


def answer_html(ans):
    """Muted answer text; drawn rotated 180 degrees directly under its question."""
    return f"<p style='color:#6f6f6f; font-size:7.6pt'>{split_parts(ans)}</p>"


PART_RE = re.compile(r"\s*(<b>[^<]{1,28}[.:]</b>)")


def split_parts(body):
    """Start each bold 'Label.' part on its own line so the gutter can be skimmed."""
    out = PART_RE.sub(lambda m: "</p><p class='part'>" + m.group(1), body.strip())
    return out[len("</p><p class='part'>"):] if out.startswith("</p><p class='part'>") else out


def stub_html(n):
    label = KINDS[n["kind"]][0]
    accent = "#%02x%02x%02x" % tuple(int(c * 255) for c in KINDS[n["kind"]][2])
    head = f"<span class='lab' style='color:{accent}'>{n['num']} · {label}"
    if n["kind"] == "check" and n.get("check"):
        head += f" Q{n['check']}"
    if n["title"]:
        head += f": {n['title']}"
    short = n.get("short") or first_sentence(n["body"])
    tail = ("full question and its upside-down answer after this page" if n["kind"] == "check"
            else "full note after this page")
    return f"<p>{head}</span></p><p>{short} <i>({tail})</i></p>"


def note_parts(body):
    """Split a note into [(label_lower, text)], using its bold 'Label.' parts."""
    out, last, cur = [], "", 0
    for m in re.finditer(r"<b>([^<]{1,28}?)[.:]?</b>", body or ""):
        lab = m.group(1).strip().rstrip(".:").lower()
        if len(lab.split()) <= 5:
            txt = plain(body[cur:m.start()])
            if txt or last:
                out.append((last, txt))
            last, cur = lab, m.end()
    tail = plain(body[cur:])
    if tail or last:
        out.append((last, tail))
    return out


def is_impl(lab):
    lab = (lab or "").rstrip("?").strip()
    return lab.startswith("why") or any(lab.startswith(x.rstrip("?")) for x in IMPLICATION)


def is_desc(lab):
    lab = (lab or "").strip()
    return lab == "" or any(lab == d or lab.startswith(d + " ") for d in DESCRIPTIVE)


def has_part(labels, req):
    return [i for i, l in enumerate(labels) if l.rstrip("?").startswith(req)]


def content_words(text):
    return [w for w in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(w) > 2 and w not in STOP]


def first_sentence(body):
    text = re.sub(r"^\s*<b>[^<]{1,28}[.:]</b>\s*", "", body)  # drop a leading part label
    m = re.match(r"(.+?[.?!])(\s|<|$)", text, re.S)
    return m.group(1) if m else text[:140]


def plain(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def measure(html_str, width, css, archive):
    tmp = pymupdf.open()
    pg = tmp.new_page(width=width + 20, height=3000)
    spare, _ = pg.insert_htmlbox(pymupdf.Rect(0, 0, width, 3000), html_str, css=css,
                                 archive=archive, scale_low=1)
    tmp.close()
    return 3000 - spare


# ---------------------------------------------------------------- anchors
def parse_line_range(s, n_lines):
    m = re.fullmatch(r"\s*L?(\d+)\s*(?:-\s*L?(\d+))?\s*", s)
    if not m:
        raise ValueError(f"bad line range {s!r} (use 'L7' or 'L7-L12')")
    a = int(m.group(1))
    b = int(m.group(2) or a)
    if not (1 <= a <= b <= n_lines):
        raise ValueError(f"line range {s!r} outside 1..{n_lines}")
    return a, b


def resolve(ann, page, lines):
    """Return a list of rects on the page covering the anchor."""
    if "rect" in ann:
        return [pymupdf.Rect(ann["rect"])]
    if "lines" in ann:
        a, b = parse_line_range(ann["lines"], len(lines))
        return [lines[i - 1][0] for i in range(a, b + 1)]
    if "quote" in ann:
        clip = None
        if "within" in ann:
            a, b = parse_line_range(ann["within"], len(lines))
            clip = pymupdf.Rect(lines[a - 1][0])
            for i in range(a, b + 1):
                clip |= lines[i - 1][0]
            clip = clip + (-2, -2, 2, 2)
        q = " ".join(ann["quote"].split())
        hits = search_quote(page, q, clip=clip)
        if not hits:
            first = q.split()[0].lower() if q.split() else ""
            near = [f"L{i}" for i, (_, t) in enumerate(lines, 1) if first and first in t.lower()][:8]
            raise ValueError(f"quote not found: {q!r}. Lines containing {first!r}: {near}. "
                             "Use a shorter quote, add 'within', or switch to 'lines'.")
        # Several hits: keep the first contiguous run (search_for splits a
        # multi-line match into one rect per line).
        run = [hits[0]]
        for r in hits[1:]:
            if r.y0 - run[-1].y1 < 6 and r.y0 >= run[-1].y0:
                run.append(r)
            else:
                break
        return run
    raise ValueError("annotation needs one of 'lines', 'quote', or 'rect'")



# ---------------------------------------------------------------- text helpers
# search_for's default flags keep ligatures (a PDF's "ﬁnd" does not match a typed
# "find"), so quotes are searched with ligatures expanded and hyphenated line
# breaks joined; typographic quotes/dashes in the PDF are tried as well.
SEARCH_FLAGS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP


def search_quote(page, q, clip=None):
    variants = [q]
    if "'" in q:
        variants.append(q.replace("'", "’"))
    if '"' in q:
        variants.append(re.sub(r'"(\S)', "“\\1", q).replace('"', "”"))
    if " - " in q or "--" in q:
        variants += [q.replace(" - ", " – ").replace("--", "–"), q.replace(" - ", "—").replace("--", "—")]
    for v in variants:
        hits = page.search_for(v, clip=clip, flags=SEARCH_FLAGS)
        if hits:
            return hits
    return find_words(page, q, clip)


EDGE_PUNCT = ".,;:!?\"'()[]{}\u201c\u201d\u2018\u2019"


def _tok(t):
    return norm_text(t).strip(EDGE_PUNCT)


def find_words(page, q, clip=None):
    """Word-sequence match, for quotes search_for misses: a word hyphenated across
    a line break ("pro-" / "gram"), or a real hyphen at a break ("state-of-" /
    "the-art"). Returns one rect per word fragment, in reading order."""
    toks = []   # (forms, rects, line id)
    for w in page.get_text("words", flags=SEARCH_FLAGS, clip=clip):
        t, r, lid = w[4], pymupdf.Rect(w[:4]), (w[5], w[6])
        if toks and toks[-1][2] != lid and toks[-1][3].endswith("-") and t[:1].islower():
            forms, rects, _, raw = toks[-1]
            toks[-1] = ({_tok(raw[:-1] + t), _tok(raw + t)}, rects + [r], lid, raw + t)
            continue
        toks.append(({_tok(t)}, [r], lid, t))
    want = [_tok(x) for x in q.split()]
    want = [x for x in want if x]
    n = len(want)
    for i in range(len(toks) - n + 1):
        if all(want[j] in toks[i + j][0] for j in range(n)):
            rects = [r for k in range(i, i + n) for r in toks[k][1]]
            merged = []   # one rect per line
            for r in rects:
                if merged and abs(merged[-1].y0 - r.y0) < 2 and r.x0 >= merged[-1].x0:
                    merged[-1] = merged[-1] | r
                else:
                    merged.append(pymupdf.Rect(r))
            return merged
    return []


def norm_text(t):
    """Lowercased, whitespace-collapsed text with ligatures expanded, hyphenated
    line breaks joined, and typographic quotes/dashes made ASCII."""
    t = unicodedata.normalize("NFKC", t or "")
    t = re.sub(r"(\w)-\n(?=[a-z])", r"\1", t)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"') \
        .replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).lower().strip()


REF_HEAD = re.compile(r"^\s*(\d{1,2}\.?\s+)?(references|bibliography|works cited)\s*$", re.I)


def ref_cut(doc):
    """(page, block_no) of the reference-list heading, or None."""
    for pno in range(len(doc), 0, -1):
        for b in doc[pno - 1].get_text("blocks"):
            if REF_HEAD.match(b[4].strip()) and pno > 1:
                return pno, b[5]
    return None


def scope_words(page, pno, cut, keep_after=False):
    """Body words on a page (not the reference list): (x0,y0,x1,y1,text,block,line,wno)."""
    ws = page.get_text("words", flags=SEARCH_FLAGS)
    if cut and not keep_after:
        cp, cb = cut
        if pno > cp:
            return []
        if pno == cp:
            ws = [w for w in ws if w[5] < cb]
    return ws


ABBREV = re.compile(r"(?i)^(e\.g|i\.e|et al|fig|figs|eq|eqs|sec|vs|cf|etc|no|resp|approx|ch)\.$")


def sentences(words):
    """Group words into sentences; each is (text, [rects])."""
    toks = []
    for i, w in enumerate(words):
        t, r = w[4], pymupdf.Rect(w[:4])
        if toks and toks[-1][0].endswith("-") and toks[-1][2] != (w[5], w[6]) and t[:1].islower():
            pt, pr, _ = toks[-1]
            toks[-1] = (pt[:-1] + t, pr + [r], (w[5], w[6]))
            continue
        toks.append((t, [r], (w[5], w[6])))
    out, cur, rects = [], [], []
    for i, (t, rs, _) in enumerate(toks):
        cur.append(t)
        rects += rs
        nxt = toks[i + 1][0] if i + 1 < len(toks) else ""
        if re.search(r"[.!?][)\]\"”']?$", t) and not ABBREV.match(t) \
                and (not nxt or re.match(r"[A-Z(\[“\"]", nxt)):
            out.append((" ".join(cur), rects))
            cur, rects = [], []
    if cur:
        out.append((" ".join(cur), rects))
    return out


# Sentences that usually carry a key point: numbers with units of comparison,
# the authors' own findings and design decisions, definitions and coinages,
# limitations, and conclusions. A net, not a judge: each hit needs a decision.
SALIENT_RE = re.compile(
    r"(?i)(\b\d[\d,.]*\s?(?:%|percent\b|x\b|×|times\b|fold\b)"
    r"|\bwe (?:show|find|found|demonstrate|observe|observed|propose|present|introduce|develop|developed|design|"
    r"designed|built|implement|implemented|conclude|argue|believe|identify|identified|evaluate|chose|decided|"
    r"learned|noticed|realized)\b"
    r"|\b(?:we call|called|refer(?:red)? to as|known as|is defined as|defined as|dubbed|termed|we define)\b"
    r"|\b(?:however|unfortunately|limitation|cannot|unable|fails? to|failed to|insufficient|drawback|"
    r"trade-?off|at the cost of|in contrast|unlike)\b"
    r"|\b(?:in summary|in conclusion|our results|lessons? learned|key insight|the main|the key|"
    r"works by|relies on|is responsible for|in order to|allows? (?:us|it|the)|which means|because|instead of|"
    r"in other words|that is,|combines?|the (?:first|second|third) step|approach|strategy|"
    r"we (?:used|use|chose|choose|targeted|came up with|added|applied|inserted|limited|rely))\b)")

POINT_WHAT = {"claim", "result", "number", "mechanism", "definition", "limitation", "concept", "assumption",
              "design"}
CONCEPT_WHAT = {"mechanism", "concept", "definition"}
HL_WARN, HL_MAX = 0.15, 0.25
# Floors. Every check above is a ceiling; without floors the cheapest passing spec is a sparse one, and
# a long paper's back half drifts down to it (a 44-page survey's 23-page appendix got 2 highlights a page
# and 4 gutter notes in total). Floors scale with the page's body words.
FLOOR_WORDS = 300              # highlight-share floor applies to pages with at least this many body words
HL_MIN, HL_MIN_WARN = 0.04, 0.06
KP_WORDS, KP_MIN, KP_CAP = 150, 2, 6   # key points needed: ceil(words / 150), at least 2, never more than 6
NOTE_FLOOR = ((500, 2), (250, 1))      # (body words, gutter notes anchored on the page)
CHECK_EVERY = 6                # at least one check question per this many reviewed pages
THIN_SHARE = 0.10              # at most this share of reviewed pages may claim 'thin' to go under a floor
NOTE_MIN_WORDS = 15            # a gutter note (not a check question) says why its target matters in at least this
WHY_MIN = 6                    # words in a highlight's hover reason
REASON_MIN = 5                 # words in a not_key / thin / nothing_key reason
NOTKEY_MIN_QUOTE = 4           # words in a not_key quote (a 2-word quote like "in the" matches half the page)
COVER_MAX_LINES = 12           # an anchor longer than this does not count toward the salience net
GIVEAWAY_MAX = 0.45            # share of a check answer's content words already in an upright note on the page   # share of a page's body words under highlights
REVIEW_MIN_WORDS = 150         # pages with fewer body words need no review entry
WHY_MAX = 25                   # words in a highlight's hover note
WHY_DESC = re.compile(r"(?i)^\s*(names|describes|defines|lists|states|introduces|mentions|sizes|shows|notes|says|"
                      r"gives|explains what|summari[sz]es|restates)\b")
# a not_key reason that says a note or walkthrough explains the sentence means it IS a key point
NOTKEY_DODGE = re.compile(r"(?i)\b(covers?|covered|walkthrough|unpacks?|builds? (?:this|it)|folded into|"
                          r"explained (?:in|by)|handled (?:in|by))\b")
NOTKEY_OK = re.compile(r"(?i)\b(restates?|repeats?|duplicates?|anecdote|aside|logistics|acknowledg|"
                       r"pointer to (?:ref|the reference)|citation only|page furniture|caption text)\b")


# ---------------------------------------------------------------- coverage
# Research-paper objects that must each get a decision (annotate or skip) and an
# audit entry: figures, tables, algorithms, and numbered display equations.
FIG_RE = re.compile(r"^(Figure|Fig\.?)\s*(\d{1,3})\b", re.I)
TAB_RE = re.compile(r"^Table\s*(\d{1,3})\b", re.I)
ALG_RE = re.compile(r"^(Algorithm|Listing)\s*(\d{1,3})\b", re.I)
EQ_RE = re.compile(r"\((\d{1,3})\)\s*$")
EQ_REF_RE = re.compile(r"(Equation|Eqn?\.?|Eq\.?)\s*\(\d{1,3}\)\s*$", re.I)
SEC_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][A-Za-z].{2,60})$")


def find_objects(doc):
    """Figures, tables, algorithms and numbered equations, each with the vertical
    span it occupies. Returns [(name, page, y0, y1)]. Captions are detected from
    their first line; equations from a trailing '(N)' in the text column."""
    out = []
    for pno, page in enumerate(doc, start=1):
        lines = page_lines(page)
        col = text_column(lines, page.rect)
        for rect, text in lines:
            t = text.strip()
            mf = FIG_RE.match(t)
            mt = TAB_RE.match(t)
            ma = ALG_RE.match(t)
            if mf:
                out.append((f"Figure {mf.group(2)}", pno, rect.y0 - 2, rect.y1 + 2))
                continue
            if mt:
                out.append((f"Table {mt.group(1)}", pno, rect.y0 - 2, rect.y1 + 2))
                continue
            if ma:
                out.append((f"{ma.group(1).capitalize()} {ma.group(2)}", pno, rect.y0 - 2, rect.y1 + 2))
                continue
            eq = EQ_RE.search(t)
            if eq and not EQ_REF_RE.search(t) and rect.x0 > col[0] + 20 and len(t) < 160:
                out.append((f"Eq. ({eq.group(1)})", pno, rect.y0 - 14, rect.y1 + 2))
    # merge duplicate names on a page (a figure caption that wraps, a table cited twice)
    merged = {}
    for name, pno, y0, y1 in out:
        key = (name, pno)
        if key in merged:
            m = merged[key]
            merged[key] = (name, pno, min(m[2], y0), max(m[3], y1))
        else:
            merged[key] = (name, pno, y0, y1)
    return sorted(merged.values(), key=lambda t: (t[1], t[2]))


SECTION_HEAD = re.compile(r"^((?:\d{1,2})(?:\.\d{1,2}){0,2}\.?\s+)?"
                          r"(Abstract|Introduction|Background|Related Work|Related work|Motivation|"
                          r"Threat Model|Overview|Design|Approach|Method(?:s|ology)?|Model|System|"
                          r"Implementation|Setup|Experiment(?:s|al Setup)?|Evaluation|Results|Analysis|"
                          r"Discussion|Ablation|Limitations|Conclusion(?:s)?|Future Work|"
                          r"Acknowledg(?:e)?ments|References|Appendix)\b", re.I)


def find_sections(doc):
    """Best-effort list of section headings (name, page) for the reading-route check."""
    secs = []
    for pno, page in enumerate(doc, start=1):
        for rect, text in page_lines(page):
            t = text.strip()
            if len(t) > 70:
                continue
            m = SECTION_HEAD.match(t) or SEC_RE.match(t)
            if m:
                secs.append((t, pno))
    # dedupe consecutive repeats
    seen, out = set(), []
    for t, p in secs:
        k = re.sub(r"\W+", "", t.lower())
        if k and k not in seen:
            seen.add(k)
            out.append((t, p))
    return out


def find_citations(doc):
    """Reference-list entries, as a convenience list for choosing which cited works
    to link. Not enforced for coverage. Returns [(id, text)]."""
    refs, started, cur = [], False, []
    for pno, page in enumerate(doc, start=1):
        for rect, text in page_lines(page):
            t = text.strip()
            if not started and re.match(r"^(References|Bibliography)\s*$", t, re.I):
                started = True
                continue
            if not started:
                continue
            m = re.match(r"^\[(\d{1,3})\]\s+(.*)", t)
            if m:
                if cur:
                    refs.append(" ".join(cur))
                cur = [m.group(2)]
            elif cur:
                cur.append(t)
    if cur:
        refs.append(" ".join(cur))
    return [(f"C{i+1}", r[:200]) for i, r in enumerate(refs)]

# ---------------------------------------------------------------- layout
def layout(items, top, bottom):
    """items: list of dicts with 'anchor_y' and 'h'. Sets 'y'. Returns overflow (pt)."""
    y = top
    for it in items:
        it["y"] = max(it["anchor_y"] - 4, y)
        y = it["y"] + it["h"] + BOX_GAP
    # push up from the bottom if needed
    limit = bottom
    for it in reversed(items):
        if it["y"] + it["h"] > limit:
            it["y"] = limit - it["h"]
        limit = it["y"] - BOX_GAP
    return max(0.0, top - items[0]["y"]) if items else 0.0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--render", help="directory for PNG renders of annotated pages")
    ap.add_argument("--gutter", type=float, default=240)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--link-report", help="JSON written by check_links.py; links it opened are labeled so")
    ap.add_argument("--part", action="store_true",
                    help="validate one page chunk (a worker's spec_part): needs 'pages'; skips the paper-wide "
                         "pieces (guide, critique, cited works) and accepts walkthrough decisions for walkthroughs "
                         "that 'sections' plans but that are not written yet")
    args = ap.parse_args()
    report = json.loads(Path(args.link_report).read_text()) if args.link_report else {}

    spec = json.loads(Path(args.spec).read_text())
    doc = pymupdf.open(args.pdf)
    n_pages = len(doc)
    archive, base_css = font_setup()
    errors, warnings = [], []
    src_doc = pymupdf.open(args.pdf)

    norm = norm_text
    raw_pages = [pg.get_text() for pg in src_doc]
    paper_text = norm("\n".join(raw_pages))
    # second copy keeping the hyphen at line breaks ("state-of-\nthe-art")
    paper_text_h = norm(re.sub(r"-\n", "-", "\n".join(raw_pages)))

    def in_paper(phrase):
        q = norm(phrase)
        return q in paper_text or q in paper_text_h

    def check_quotes(text, where, attributed, other_source=False):
        """Quotation marks are for the paper's own words. In a note, any quoted phrase of 3+ words must be in
        the paper, unless the note cites another source (outside/links). Elsewhere, phrases of 4+ words that
        are attributed to the paper must be in it."""
        for m in QUOTE_RE.finditer(text or ""):
            q = next(g for g in m.groups() if g)
            nw = len(q.split())
            if attributed == "note":
                if nw >= 3 and not other_source and not in_paper(q):
                    errors.append(f"{where}: {q[:60]!r} is in quotation marks but is not the paper's text. "
                                  "Quotation marks are only for the paper's own words (or, with a link, a cited "
                                  "source's). Quote verbatim or drop the marks; never put your own phrasing in quotes.")
                continue
            if nw < 4:
                continue
            if attributed or PAPER_ATTR.search(text):
                if not in_paper(q):
                    errors.append(f"{where}: the quoted phrase {q[:60]!r} is attributed to the paper but is "
                                  "not in it (whitespace-insensitive). Quote the paper verbatim or drop the "
                                  "quotation marks. Never invent a quote.")

    if args.part and not spec.get("pages"):
        sys.exit("--part needs 'pages': [first, last] in the spec (the worker's chunk).")
    # group annotations by page, numbered in reading order
    anns = spec.get("annotations", [])
    for k, a in enumerate(anns):
        if a.get("kind") not in KINDS:
            errors.append(f"annotation #{k}: kind must be one of {list(KINDS)}")
        if not (1 <= int(a.get("page", 0)) <= n_pages):
            errors.append(f"annotation #{k}: page {a.get('page')} outside 1..{n_pages}")
        elif args.part and not (spec["pages"][0] <= int(a["page"]) <= spec["pages"][1]):
            errors.append(f"annotation #{k}: page {a['page']} is outside this chunk {spec['pages']}")
        if a.get("kind") != "check" and a.get("note") and len(plain(a["note"]).split()) < NOTE_MIN_WORDS:
            errors.append(f"annotation #{k} (p{a.get('page')}, {plain(a.get('title', ''))[:40]!r}): the note is "
                          f"{len(plain(a['note']).split())} words. A gutter note says why its target matters and "
                          f"what follows from it; that takes at least {NOTE_MIN_WORDS} words.")
        if a.get("kind") == "check":
            ans = plain(a.get("answer", ""))
            if not ans:
                errors.append(f"annotation #{k}: check questions need an 'answer'")
            elif len(ans.split()) < 15:
                errors.append(f"annotation #{k}: the answer is {len(ans.split())} words; explain WHY, not just "
                              "what (elaborated feedback beats bare answers). Aim for 20-60 words.")
    if errors:
        sys.exit("\n".join(errors))

    # ---- explainers ('walkthroughs'): short separate PDFs for the paper's central ideas (walkthrough.py).
    # Each gets a gutter note on its first page pointing to the file.
    spec_dir = Path(args.spec).parent
    wts = {}
    for j, w in enumerate(spec.get("walkthroughs") or []):
        wtag = f"walkthrough #{j} ({w.get('id', '?')})"
        miss = [f for f in ("id", "title", "pages", "spec", "file", "why", "anchor") if not w.get(f)]
        if miss:
            errors.append(f"{wtag}: missing {miss}. 'anchor' is a verbatim phrase on the first page (e.g., the "
                          "section heading) where the pointer note goes.")
            continue
        try:
            a0, a1 = int(w["pages"][0]), int(w["pages"][1])
        except (TypeError, ValueError, IndexError):
            errors.append(f"{wtag}: 'pages' must be [first, last]")
            continue
        wsp = spec_dir / w["spec"]
        if not wsp.exists():
            errors.append(f"{wtag}: walkthrough spec {wsp} not found. Write it (see walkthrough.py) first.")
            continue
        wspec = json.loads(wsp.read_text())
        if "passages" in wspec or "concepts" in wspec:
            errors.append(f"{wtag}: {wsp.name} uses the retired concepts/passages format. Explainers follow "
                          "references/walkthroughs.md and the docstring of walkthrough.py.")
            continue
        rngs = [(a0, a1)]
        if w.get("proof_pages"):
            rngs.append((int(w["proof_pages"][0]), int(w["proof_pages"][1])))
        w = dict(w, _pages=(a0, a1), _ranges=rngs, _passages=[])
        wts[w["id"]] = w
        if not args.part:
            rv = spec_dir / (w.get("review") or f"review_{w['id']}.json")
            if not rv.exists():
                errors.append(f"{wtag}: no reader review at {rv}. A fresh reader reviews every explainer before "
                              "delivery (references/walkthroughs.md, 'Reader review').")
            else:
                r_ = json.loads(rv.read_text())
                if r_.get("verdict") != "pass" or r_.get("undefined_terms"):
                    errors.append(f"{wtag}: the reader review in {rv.name} does not pass "
                                  f"(verdict {r_.get('verdict')!r}, undefined terms {r_.get('undefined_terms')}).")
                elif r_.get("spec_sha") != spec_sha(wspec):
                    errors.append(f"{wtag}: {wsp.name} changed after its reader review ({rv.name} reviewed "
                                  f"{r_.get('spec_sha')!r}, the spec is now {spec_sha(wspec)!r}). Review it again.")
        if not (Path(args.out).parent / w["file"]).exists():
            warnings.append(f"{wtag}: {w['file']} is not beside the output yet; build it with walkthrough.py "
                            "and save it next to the annotated PDF.")
        parts_ = ["the idea in one paragraph", "notation", "the mechanism as a diagram", "a worked example"]
        if wspec.get("proof"):
            parts_.append(f"a proof outline with its {len(wspec.get('hard_steps', []))} hardest steps broken down")
        anns.append({"page": a0, "quote": w["anchor"], "kind": "key", "mark": "none", "_synthetic": True,
                     "title": f"Explainer: {w['title']}",
                     "short": f"Short explainer for pp. {a0}-{a1}: <i>{html.escape(w['file'])}</i>.",
                     "note": (f"<b>Why it matters.</b> {w['why']} <b>Where.</b> a separate short PDF, "
                              f"<i>{html.escape(w['file'])}</i>: " + ", ".join(parts_) + ".")})
    if errors:
        sys.exit("\n".join(errors))

    # ---- sections: the paper map's section plan. Walkthroughs are decided here, per section, before any page
    # is annotated, and page_review 'section' must name one of these ids (free-text section names let a dense
    # stretch be split into many small "sections" that never reach the density trigger).
    sections = spec.get("sections") or []
    sec_by_id, page_secs = {}, {}
    sel = spec.get("pages") or [1, n_pages]
    if args.part:   # planned walkthroughs not written yet: span every section that names them
        for s in sections:
            wid = s.get("walkthrough")
            try:
                s0, s1 = int(s["pages"][0]), int(s["pages"][1])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if wid and (wid not in wts or wts[wid].get("_passages") is None):
                p0, p1 = wts[wid]["_pages"] if wid in wts else (s0, s1)
                wts[wid] = {"id": wid, "_pages": (min(p0, s0), max(p1, s1)), "_passages": None,
                            "file": f"walkthrough {wid}"}
    for j, s in enumerate(sections):
        stag = f"section #{j} ({s.get('id', '?')})"
        try:
            s0, s1 = int(s["pages"][0]), int(s["pages"][1])
            assert s.get("id") and s.get("title") and s0 <= s1
        except (KeyError, TypeError, ValueError, IndexError, AssertionError):
            errors.append(f"{stag}: needs 'id', 'title', and 'pages': [first, last]")
            continue
        s = dict(s, _pages=(s0, s1))
        sec_by_id[s["id"]] = s
        for p_ in range(s0, s1 + 1):
            page_secs.setdefault(p_, []).append(s["id"])
        wid, why_not = s.get("walkthrough"), plain(s.get("no_walkthrough", ""))
        if s.get("references"):
            continue   # a reference list: no walkthrough decision, and its pages are not body text
        if wid and wid not in wts:
            errors.append(f"{stag}: plans walkthrough {wid!r}, but 'walkthroughs' has no entry with that id.")
        elif wid and not any(r0 <= s0 and s1 <= r1 for r0, r1 in wts[wid].get("_ranges", [wts[wid]["_pages"]])):
            errors.append(f"{stag}: walkthrough {wid!r} covers pp. {wts[wid]['_pages'][0]}-{wts[wid]['_pages'][1]}, "
                          f"not the whole section (pp. {s0}-{s1}).")
    if sections:
        orphan = [p_ for p_ in range(sel[0], sel[1] + 1)
                  if p_ not in page_secs and len(scope_words(doc[p_ - 1], p_, None)) >= REVIEW_MIN_WORDS]
        if orphan:
            errors.append(f"pages in no declared section: {orphan}. Every page belongs to a section; mark a "
                          "reference list with \"references\": true.")
    elif args.part or sel[1] - sel[0] + 1 > 5:
        errors.append("missing 'sections': list the paper's sections as {id, title, pages: [first, last]} "
                      "(the paper map; a reference list gets \"references\": true).")

    def body_after_refs(p_):
        """A page after the reference heading is still body text if a declared non-reference section holds it
        (appendices often follow the references)."""
        return any(not sec_by_id[i].get("references") for i in page_secs.get(p_, []) if i in sec_by_id)
    if errors:
        sys.exit("\n".join(errors))

    per_page = {}
    covered_rects = []
    page_lines_cache = {}
    for k, a in enumerate(anns):
        pno = int(a["page"])
        if pno not in page_lines_cache:
            page_lines_cache[pno] = page_lines(doc[pno - 1])
        try:
            rects = resolve(a, doc[pno - 1], page_lines_cache[pno])
        except ValueError as e:
            errors.append(f"p{pno} annotation #{k}: {e}")
            continue
        per_page.setdefault(pno, []).append((min(r.y0 for r in rects), k, a, rects))
        covered_rects.append((pno, rects, " ".join([plain(a.get("title", ""))] + list(a.get("covers", []))),
                              a["kind"], a.get("fig_kind")))

    objects = find_objects(pymupdf.open(args.pdf))
    pages_sel = spec.get("pages")  # optional [first, last] for partial requests
    skips = [(sk["item"].strip().lower(), sk.get("page")) for sk in spec.get("skip", [])]
    coverage = []
    for name, opno, y0, y1 in objects:
        if pages_sel and not (pages_sel[0] <= opno <= pages_sel[1]):
            continue
        # An annotation covers an object if its title/covers names it (same or
        # adjacent page), or its anchor touches the object's first lines. Mere
        # overlap with a long object's body doesn't count.
        name_re = re.compile(r"(?<![\w.])" + re.escape(name) + r"(?![\d.]*\d)", re.I)
        head_y1 = y1 if name.startswith("Eq.") else min(y1, y0 + 30)
        twin_pages = {op for nm, op, _a, _b in objects if nm == name and op != opno}
        hitters = [fk for p, rs, names, _, fk in covered_rects
                   if (abs(p - opno) <= 1 and (p == opno or p not in twin_pages) and name_re.search(names))
                   or (p == opno and any(r.y1 > y0 - 2 and r.y0 < head_y1 for r in rs))]
        hit = bool(hitters)
        if hit and re.match(r"(?i)(fig|table)", name) and not any(hitters):
            errors.append(f"{name} (p{opno}): no annotation covering it has 'fig_kind'. Give its main note "
                          "fig_kind 'data' (Metric/Baseline/Takeaway) or 'diagram' (Point/Why it matters).")
        skipped = any(item == name.lower() and (sp is None or int(sp) == opno) for item, sp in skips)
        status = "annotated" if hit else ("skipped" if skipped else "MISSING")
        if hit and skipped:
            warnings.append(f"{name} (p{opno}) is in 'skip' but an annotation also touches it; "
                            "check that this is intended.")
        coverage.append((name, opno, status))
    missing = [f"{n} (p{p})" for n, p, st in coverage if st == "MISSING"]

    # ---- error audit: every formal object in scope, including skipped ones,
    # must be checked for errors and the check recorded in 'audit'.
    audit = {}
    for en in spec.get("audit", []):
        audit[(en.get("item", "").strip().lower(), en.get("page"))] = en
    def audit_entry(name, opno):
        return audit.get((name.lower(), opno)) or audit.get((name.lower(), None))
    unaudited, flagged, bad_flags = [], 0, []
    for name, opno, y0, y1 in objects:
        if pages_sel and not (pages_sel[0] <= opno <= pages_sel[1]):
            continue
        en = audit_entry(name, opno)
        if not en or en.get("result") not in ("ok", "flagged") or not en.get("how"):
            unaudited.append(f"{name} (p{opno})")
            continue
        if en["result"] == "flagged":
            flagged += 1
            name_re = re.compile(r"(?<![\w.])" + re.escape(name) + r"(?![\d.]*\d)", re.I)
            head_y1 = y1 if name.startswith("Eq.") else min(y1, y0 + 30)
            has_note = any(kd == "subtle" and ((abs(p - opno) <= 1 and name_re.search(names))
                           or (p == opno and any(r.y1 > y0 - 2 and r.y0 < y1 for r in rs)))
                           for p, rs, names, kd, _fk in covered_rects)
            if not has_note:
                bad_flags.append(f"{name} (p{opno})")
    if unaudited:
        errors.append("objects with no error-audit entry: " + ", ".join(unaudited) + ". Add "
                      "{\"item\": name, \"page\": p, \"result\": \"ok\"|\"flagged\", \"how\": \"what you checked\"} "
                      "to 'audit' for each, including skipped objects.")
    if bad_flags:
        errors.append("audit says 'flagged' but there is no 'subtle' note on: " + ", ".join(bad_flags))
    if missing:
        errors.append("figures/tables/equations/algorithms with no annotation and no 'skip' entry: "
                      + ", ".join(missing) + ". Annotate each (anchor on its caption line, or name it in "
                      "'title'/'covers'), or add {\"item\": name, \"page\": p, \"why\": reason} to 'skip'.")

    # ---- page-by-page review: every key point on every page gets a decision and a highlight
    def grams(t, n=4):
        w = t.split()
        return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

    def first_run(hits):
        run = [hits[0]]
        for r_ in hits[1:]:
            if r_.y0 - run[-1].y1 < 6 and r_.y0 >= run[-1].y0:
                run.append(r_)
            else:
                break
        return run

    def default_mark(a_, rs_):
        return ("bar" if "lines" in a_ and len(rs_) > 3 else "highlight") if a_["kind"] in ("key", "subtle") \
            else "none"

    cut = ref_cut(src_doc)
    review = {}
    for r_ in spec.get("page_review") or []:
        try:
            review[int(r_["page"])] = r_
        except (KeyError, TypeError, ValueError):
            errors.append(f"page_review entry {str(r_)[:60]}: needs an integer 'page'")
    first_p, last_p = (pages_sel or [1, n_pages])
    titles_by_page, ann_hl = {}, {}
    for pno_, items_ in per_page.items():
        for _y, _k, a_, rs_ in items_:
            titles_by_page.setdefault(pno_, []).append(plain(a_.get("title", "")).lower())
            if (a_.get("mark") or default_mark(a_, rs_)) == "highlight":
                ann_hl.setdefault(pno_, []).extend(rs_)
    review_marks, sec_concepts, density = [], {}, {}
    n_points = n_marked = 0
    thin_pages, cov_rows = [], []
    for pno in sorted(review):
        if not (first_p <= pno <= last_p):
            warnings.append(f"page_review for p{pno} is outside the requested pages; ignored")
    for pno in range(first_p, last_p + 1):
        page = doc[pno - 1]
        words = scope_words(page, pno, cut, keep_after=(pno in review) or bool(cut and pno > cut[0]
                                                                             and body_after_refs(pno)))
        r_ = review.get(pno)
        if len(words) < REVIEW_MIN_WORDS and not r_:
            continue
        if not r_:
            errors.append(f"p{pno}: no page_review entry. Read the page, then list its key points (claims, "
                          "results, numbers, mechanisms, definitions, limitations, design choices), each with a "
                          "verbatim quote and a decision.")
            continue
        pts = r_.get("points") or []
        if not pts and not r_.get("nothing_key"):
            errors.append(f"p{pno}: page_review lists no key points. List them, or give 'nothing_key' with the "
                          "reason (e.g., 'acknowledgments and funding only, no claims').")
        # floors: key points, gutter notes, and highlighted share scale with the page's body text. A page under
        # any floor needs 'thin' (or 'nothing_key'), and every such page counts against one shared budget.
        nk_reason = r_.get("nothing_key")
        thin = plain(r_.get("thin", ""))
        for fld, val in (("thin", thin), ("nothing_key", plain(nk_reason) if isinstance(nk_reason, str) else "")):
            if (fld == "thin" and val or fld == "nothing_key" and nk_reason) and len(val.split()) < REASON_MIN:
                errors.append(f"p{pno}: '{fld}' must say what is on the page and why it carries fewer ideas than "
                              f"its length suggests ({REASON_MIN}+ words); the verifier checks it.")
        exempt = bool(thin or nk_reason)
        under = []
        need_pts = min(KP_CAP, max(KP_MIN, math.ceil(len(words) / KP_WORDS)))
        n_notes_here = sum(1 for _y, _k, a_, _r in per_page.get(pno, [])
                           if a_.get("note") and not a_.get("_synthetic") and a_.get("kind") != "check")
        need_notes = next((n for w_, n in NOTE_FLOOR if len(words) >= w_), 0)
        if len(words) >= REVIEW_MIN_WORDS:
            if len(pts) < need_pts:
                under.append(f"{len(pts)} key points (needs {need_pts} for {len(words)} words of body text)")
            if n_notes_here < need_notes:
                under.append(f"{n_notes_here} gutter note(s), not counting check questions (needs {need_notes})")
        sec_default = r_.get("section")
        if sections:
            if sec_default and sec_default not in sec_by_id:
                errors.append(f"p{pno}: section {sec_default!r} is not a declared section id "
                              f"({', '.join(sec_by_id)}).")
            elif not sec_default:
                sec_default = (page_secs.get(pno) or [None])[0]
        covered = [rr for _y, _k, _a, rs_ in per_page.get(pno, []) if len(rs_) <= COVER_MAX_LINES for rr in rs_]
        hl_here = list(ann_hl.get(pno, []))
        page_concepts = 0
        for i, pt in enumerate(pts):
            ptag = f"p{pno} key point #{i}"
            q = " ".join((pt.get("quote") or "").split())
            nq = len(q.split())
            if nq < 2:
                errors.append(f"{ptag}: 'quote' must be at least 2 verbatim words from p{pno}")
                continue
            if nq > 45:
                errors.append(f"{ptag}: quote is {nq} words; highlight the load-bearing words, not the paragraph "
                              "(3-30 words).")
            elif nq > 30:
                warnings.append(f"{ptag}: quote is {nq} words; tighten it to the words that carry the point.")
            hits = search_quote(page, q)
            if not hits:
                elsewhere = [p2 for p2 in range(1, n_pages + 1) if p2 != pno and search_quote(src_doc[p2 - 1], q)]
                errors.append(f"{ptag}: quote {q[:70]!r} is not on p{pno}"
                              + (f" (it is on p{elsewhere[0]})" if elsewhere else
                                 " or anywhere in the paper; copy it from lines.txt") + ".")
                continue
            rects = first_run(hits)
            if nq == 2 and len(hits) > len(rects):
                errors.append(f"{ptag}: the 2-word quote {q!r} occurs more than once on p{pno}; add a word so the "
                              "highlight lands on the right occurrence.")
            what = pt.get("what")
            if what not in POINT_WHAT:
                errors.append(f"{ptag}: 'what' must be one of {sorted(POINT_WHAT)}")
            why = plain(pt.get("why", ""))
            if why:
                check_quotes(why, ptag, attributed="note")
                if WHY_DESC.match(why):
                    errors.append(f"{ptag}: 'why' opens with {WHY_DESC.match(why).group(0)!r}, which says what the "
                                  "text does. Say why it matters: what it explains, enables, costs, or breaks.")
                qw = set(content_words(q))
                ww = content_words(why)
                if len(ww) >= 5 and sum(1 for x in ww if x in qw) / len(ww) >= RESTATE_MAX:
                    errors.append(f"{ptag}: 'why' repeats the quote; say why it matters instead.")
                if len(why.split()) > WHY_MAX + 10:
                    errors.append(f"{ptag}: 'why' is {len(why.split())} words; a hover note is one line "
                                  f"(<= {WHY_MAX}). Longer explanations belong in a gutter note.")
                elif len(why.split()) > WHY_MAX:
                    warnings.append(f"{ptag}: 'why' is {len(why.split())} words; aim for <= {WHY_MAX}.")
            dec = (pt.get("decision") or "").strip()
            hover = None
            psec = pt.get("section") or sec_default
            if dec == "mark":
                if not why:
                    errors.append(f"{ptag}: decision 'mark' needs 'why': one line on why this matters "
                                  "(shown when hovering over the highlight).")
                elif len(why.split()) < WHY_MIN:
                    errors.append(f"{ptag}: 'why' is {len(why.split())} words; say what this explains, enables, "
                                  f"costs, or breaks ({WHY_MIN}+ words).")
                hover = why
            elif dec.lower().startswith("note:"):
                want = dec[5:].strip().lower()
                ok = want and any(t and (t.startswith(want) or want.startswith(t))
                                  for p2 in (pno - 1, pno, pno + 1) for t in titles_by_page.get(p2, []))
                if not ok:
                    errors.append(f"{ptag}: decision {dec!r} names no annotation title on p{pno - 1}-p{pno + 1}.")
                hover = f"Key point: see the note '{dec[5:].strip()}'." + (f" {why}" if why else "")
            elif dec.lower().startswith("walkthrough:"):
                wid = dec.split(":", 1)[1].strip()
                w = wts.get(wid)
                if not w:
                    errors.append(f"{ptag}: decision {dec!r} names no entry in 'walkthroughs'.")
                else:
                    if not any(r0 <= pno <= r1 for r0, r1 in w.get("_ranges", [w["_pages"]])):
                        errors.append(f"{ptag}: walkthrough {wid!r} covers pp. {w['_pages'][0]}-{w['_pages'][1]}, "
                                      f"not p{pno}.")
                    if not why:
                        errors.append(f"{ptag}: decision {dec!r} also needs 'why' (the hover line): the explainer "
                                      "covers the idea, not every sentence, so say why this point matters.")
                    hover = f"The explainer ({w['file']}) covers the idea." + (f" {why}" if why else "")
            else:
                errors.append(f"{ptag}: 'decision' must be 'mark', 'note:<annotation title>', or "
                              "'walkthrough:<id>'.")
            covered += rects
            n_points += 1
            if what in CONCEPT_WHAT:
                page_concepts += 1
                sec = pt.get("section") or sec_default or f"p{pno}"
                if sections and sec not in sec_by_id:
                    errors.append(f"{ptag}: section {sec!r} is not a declared section id.")
                sec_concepts.setdefault(sec, []).append(pno)
            if not all(any(abs(rr & h) > 0.6 * abs(rr) for h in hl_here) for rr in rects):
                review_marks.append((pno, rects, what or "key point", hover or ""))
                hl_here += rects
                n_marked += 1
        for j, nk in enumerate(r_.get("not_key") or []):
            q = " ".join((nk.get("quote") or "").split())
            if not nk.get("why") or len(plain(nk["why"]).split()) < REASON_MIN:
                errors.append(f"p{pno} not_key #{j}: give 'why' it is not a key point in {REASON_MIN}+ words (e.g., "
                              "'restates the abstract's claim on p. 1'); the verifier checks it.")
            elif NOTKEY_DODGE.search(nk["why"]) and not NOTKEY_OK.search(nk["why"]):
                errors.append(f"p{pno} not_key #{j} ({q[:40]!r}): the reason says a note or the walkthrough "
                              "explains it, which makes it a key point. List it under 'points' with decision "
                              "'note:<title>' or 'walkthrough:<id>' so it is highlighted. 'not_key' is for "
                              "restatements, asides, and logistics.")
            hits = search_quote(page, q) if len(q.split()) >= NOTKEY_MIN_QUOTE else []
            if not hits:
                errors.append(f"p{pno} not_key #{j}: quote {q[:60]!r} not found on p{pno} "
                              f"({NOTKEY_MIN_QUOTE}+ verbatim words).")
            else:
                covered += first_run(hits)
        uncovered = []
        for text, rs_ in sentences(words):
            if len(text.split()) < 6 or not SALIENT_RE.search(text):
                continue
            hit = sum(1 for rr in rs_ if any(abs(rr & c) > 0.5 * abs(rr) for c in covered))
            if hit >= 3 or hit >= 0.3 * len(rs_):
                continue
            uncovered.append(text)
        if uncovered:
            errors.append(f"p{pno}: {len(uncovered)} sentence(s) carry a signal of a key point (a number, a "
                          "finding, a definition, a limitation, a design decision) and have no decision. Add each "
                          "as a key point, or list it in this page's 'not_key' with a reason:\n      - "
                          + "\n      - ".join(u[:170] for u in uncovered))
        if words:
            hw = sum(1 for wd in words if any(h.contains(pymupdf.Point((wd[0] + wd[2]) / 2, (wd[1] + wd[3]) / 2))
                                              for h in hl_here))
            density[pno] = hw / len(words)
            if density[pno] > HL_MAX:
                errors.append(f"p{pno}: {density[pno]:.0%} of the body text is highlighted (max {HL_MAX:.0%}). "
                              "Shorten quotes to the words that carry each point.")
            elif density[pno] > HL_WARN:
                warnings.append(f"p{pno}: {density[pno]:.0%} of the body text is highlighted; aim for under "
                                f"{HL_WARN:.0%}.")
            elif len(words) >= FLOOR_WORDS and density[pno] < HL_MIN:
                under.append(f"only {density[pno]:.2%} of the body text highlighted (floor {HL_MIN:.0%}, aim for "
                             f"{HL_MIN_WARN:.0%}-{HL_WARN:.0%}; key points are missing or their quotes stop short of "
                             "the words that carry them)")
            elif len(words) >= FLOOR_WORDS and density[pno] < HL_MIN_WARN and not exempt:
                warnings.append(f"p{pno}: {density[pno]:.1%} of the body text is highlighted; aim for "
                                f"{HL_MIN_WARN:.0%}-{HL_WARN:.0%}.")
        if exempt and (under or (nk_reason and len(words) >= REVIEW_MIN_WORDS)):
            thin_pages.append(pno)
        elif under:
            errors.append(f"p{pno}: " + "; ".join(under) + ". Reread the page: list every claim, number, "
                          "mechanism, definition, assumption, limitation, and design choice, and give the "
                          "load-bearing ones a gutter note. Only if the page truly carries fewer ideas than its length "
                          "(a reference list, a table of contents, boilerplate), add 'thin': '<why>' to its review entry.")
        cov_rows.append((pno, len(words), len(pts), n_notes_here, density.get(pno, 0.0), pno in thin_pages))
    reviewed = len(cov_rows)
    if args.part and thin_pages:
        warnings.append(f"thin pages in this chunk: {thin_pages}. The full build allows {THIN_SHARE:.0%} of all "
                        "pages paper-wide, so a chunk's thin pages can still fail there.")
    if len(thin_pages) > max(1, int(THIN_SHARE * reviewed)):
        errors.append(f"{len(thin_pages)} of {reviewed} pages claim 'thin' to go under a floor ({thin_pages}); at "
                      f"most {max(1, int(THIN_SHARE * reviewed))} may. 'thin' is for pages that carry fewer ideas "
                      "than their length, not for pages that were skimmed.")
    n_checks = sum(1 for a_ in anns if a_.get("kind") == "check" and sel[0] <= int(a_["page"]) <= sel[1])
    if reviewed and n_checks < math.ceil(reviewed / CHECK_EVERY):
        errors.append(f"{n_checks} check question(s) for {reviewed} reviewed pages; add at least "
                      f"{math.ceil(reviewed / CHECK_EVERY)} (one per {CHECK_EVERY} pages), spread across sections.")
    for pno, rects, what, hover in review_marks:
        rpage = doc[pno - 1]   # keep a reference: an annotation is unbound once its page object is freed
        an = rpage.add_highlight_annot(rects)
        an.set_colors(stroke=KINDS["key"][1])
        an.set_opacity(0.45)
        an.set_info(title=f"Key point ({what})", content=hover)
        an.update()

    # ---- links everywhere: lint + an honest record of how each was checked
    def title_overlap(a, b):
        wa, wb = set(content_words(a)), set(content_words(b))
        return len(wa & wb) / max(1, min(len(wa), len(wb)))

    def check_links(links, where, expect_title=None, in_note=False):
        for lk in links or []:
            url = lk.get("url", "")
            if lk.get("type") not in LINK_TYPES:
                errors.append(f"{where}: link type {lk.get('type')!r} must be one of {sorted(LINK_TYPES)}")
            fatal, warn = lint_link(url)
            if fatal:
                errors.append(f"{where}: link {url!r} rejected ({fatal})")
            if warn:
                warnings.append(f"{where}: link {url} {warn}")
            if "verified" in lk:
                errors.append(f"{where}: 'verified' is retired. Record what actually happened: "
                              "status 'opened' (a tool returned the page this session; give 'seen' = the title it "
                              "showed), 'search' (only a search listing; 'seen' = listed title), or 'expected' "
                              "(built from an arXiv ID/DOI, not opened; give 'basis').")
            st = lk.get("status")
            if st not in LINK_STATUS:
                errors.append(f"{where}: link {url} needs status one of {sorted(LINK_STATUS)}")
            elif st in ("opened", "search") and not lk.get("seen"):
                errors.append(f"{where}: link {url} has status {st!r} but no 'seen' (the exact title the tool "
                              "showed). Without it, record status 'expected'.")
            elif st == "expected" and not lk.get("basis"):
                errors.append(f"{where}: link {url} is 'expected' but has no 'basis' (e.g., 'arXiv ID from the "
                              "paper's reference list').")
            if st == "expected" and lk.get("type") == "video":
                errors.append(f"{where}: a video link can't be 'expected'; video IDs can't be constructed. Open it "
                              "(oEmbed or the page) or drop it.")
            if in_note and lk.get("type") in NEEDS_CHOICE:
                if not lk.get("fit"):
                    errors.append(f"{where}: {lk.get('type')} link {url} needs 'fit': one line on why it teaches "
                                  "THIS concept at THIS reader's level, standalone (not a mid-series episode).")
                if len(lk.get("chosen_over") or []) < 2:
                    errors.append(f"{where}: {lk.get('type')} link {url} needs 'chosen_over': at least two "
                                  "alternatives you compared and why this one won.")
            want = expect_title or lk.get("title")
            if lk.get("seen") and want and title_overlap(lk["seen"], want) < 0.35:
                warnings.append(f"{where}: the page title you saw ({lk['seen'][:60]!r}) shares few words with "
                                f"{want[:60]!r}; make sure the link goes to the right work.")
            rep = report.get(url)
            if rep and rep.get("ok") and want and title_overlap(rep.get("title") or "", want) < 0.35:
                warnings.append(f"{where}: check_links.py opened {url} and got the title "
                                f"{(rep.get('title') or '')[:70]!r}, which doesn't match {want[:60]!r}.")
    for k, a in enumerate(anns):
        check_links(a.get("links"), f"annotation #{k}", in_note=True)
    for cw in spec.get("cited_works", []):
        for f in ("cite", "url", "why", "status"):
            if not cw.get(f):
                errors.append(f"cited_work ({plain(cw.get('cite',''))[:40]}): missing '{f}'")
        check_links([dict(cw, type=cw.get("type", "paper"))], f"cited_work ({plain(cw.get('cite',''))[:40]})",
                    expect_title=plain(cw.get("cite", "")))

    # ---- critique block: the domain-specific soundness review
    cr = spec.get("critique")
    if args.part and not cr:
        pass   # the orchestrator writes the critique after merging the chunks
    elif not cr or not cr.get("items"):
        errors.append("missing 'critique': add {\"domain\": one of %s, \"items\": [ >=4 "
                      "{check, verdict, where, note} ]}. This is the paper's soundness review."
                      % sorted(CRITIQUE_DOMAINS))
    else:
        if cr.get("domain") not in CRITIQUE_DOMAINS:
            errors.append(f"critique.domain must be one of {sorted(CRITIQUE_DOMAINS)}")
        items = cr.get("items", [])
        if len(items) < 4:
            errors.append(f"critique needs at least 4 items (has {len(items)}); apply the domain checklist")
        need = {"ok": "evidence", "concern": "claim", "flaw": "claim", "cant-tell": "missing",
                "not-applicable": "why_na"}
        for j, it in enumerate(items):
            for f in ("check", "verdict", "where", "note"):
                if not it.get(f):
                    errors.append(f"critique item #{j}: missing '{f}'")
            v = it.get("verdict")
            if v not in CRITIQUE_VERDICTS:
                errors.append(f"critique item #{j}: verdict must be one of {sorted(CRITIQUE_VERDICTS)}")
                continue
            if not it.get(need[v]):
                what = {"evidence": "what in the paper satisfies the check (with page)",
                        "claim": "the paper's claim whose support this weakens (with page)",
                        "missing": "the specific material that exists but you could not see, and where it is",
                        "why_na": "why this check does not apply to this kind of paper"}[need[v]]
                errors.append(f"critique item #{j} ({v}): add '{need[v]}': {what}.")
            cwhere = f"critique item #{j}"
            for fld in ("note", "evidence", "claim", "missing"):
                check_quotes(plain(it.get(fld, "")), f"{cwhere} ('{fld}')", attributed=(fld in ("evidence", "claim")))
            blob = plain(it.get("note", "")) + " " + plain(it.get("claim", "")) + " " + plain(it.get("evidence", ""))
            mm = METHOD_ASSERT.search(blob)
            if mm and not INFER_LAB.search(blob):
                m2 = QUOTE_RE.search(plain(it.get("claim", "")) + plain(it.get("evidence", "")))
                if not (m2 and in_paper(m2.group(1))):
                    warnings.append(f"{cwhere}: states the paper's method as fact ({mm.group(0)!r}) without a "
                                    "verbatim quote and without labeling it an inference. If the paper does not "
                                    "say this in these terms, mark it inferred or quote the text.")
            txt = plain(it.get("note", "")) + " " + plain(it.get("missing", ""))
            if v == "cant-tell" and ABSENCE_RE.search(txt):
                errors.append(f"critique item #{j}: verdict 'cant-tell' but the note says the paper omits "
                              f"something ({ABSENCE_RE.search(txt).group(0)!r}). If a claim in the paper depends "
                              "on that missing evidence, the verdict is 'concern' (or 'flaw' if the claim is "
                              "central). 'cant-tell' is only for evidence that may exist but you could not see.")
            if v == "ok" and NEGATIVE_RE.search(plain(it.get("note", ""))):
                errors.append(f"critique item #{j}: verdict 'ok' but the note reports a problem "
                              f"({NEGATIVE_RE.search(plain(it.get('note', ''))).group(0)!r}). Match the verdict "
                              "to the text: 'concern' or 'flaw'.")
            if v in ("concern", "flaw") and re.search(r"(?i)\b(no (?:problem|issue)s?|is fine|well done|"
                                                     r"does this well)\b", plain(it.get("note", ""))):
                warnings.append(f"critique item #{j}: verdict {v!r} but the note sounds positive; recheck.")

    for j, it in enumerate((spec.get("critique") or {}).get("items", [])):
        m = EXISTENCE_RE.search(plain(it.get("note", "")))
        if m and not it.get("checked"):
            errors.append(f"critique item #{j}: says {m.group(0)!r} without 'checked' (what you searched, what came "
                          "back). Never guess that something does not exist.")
    m = EXISTENCE_RE.search(plain((spec.get("guide") or {}).get("summary", "")))
    if m:
        errors.append(f"guide summary says {m.group(0)!r}; existence claims belong in a note with 'checked'.")

    # ---- reading guide route should mention the main sections (warn only; detection is noisy)
    guide = spec.get("guide") or {}
    route_txt = " ".join(plain(x.get("what", "")) + " " + plain(x.get("why", ""))
                         for x in (guide.get("route") or [])).lower()
    route_txt += " " + plain(guide.get("summary", "")).lower()
    if guide and not guide.get("route"):
        warnings.append("guide has no 'route' (what to read first, what to skim); add one")
    for secname, spno in ([] if args.part else find_sections(pymupdf.open(args.pdf))):
        core = re.sub(r"^\d[\d.]*\.?\s+", "", secname).strip().lower().strip(" :.-")
        if pages_sel and not (pages_sel[0] <= spno <= pages_sel[1]):
            continue
        first = core.split()[0].strip(" :.-") if core.split() else ""
        if core and len(core) < 40 and core not in route_txt and first not in route_txt \
                and core not in ("references", "acknowledgments", "acknowledgements", "appendix"):
            warnings.append(f"section '{secname}' (p{spno}) is not mentioned in the reading guide route")

    MONO_FONT = re.compile(r"(?i)(mono|courier|cmtt|consol|menlo|inconsolata|code)")

    def anchor_words(page, rects, ann):
        if "lines" in ann:
            clip = pymupdf.Rect(rects[0])
            for r in rects[1:]:
                clip |= r
        else:
            clip = pymupdf.Rect(rects[0])
            for r in rects[1:]:
                clip |= r
            clip = pymupdf.Rect(clip.x0 - 200, clip.y0 - 14, clip.x1 + 200, clip.y1 + 14) & page.rect
        return set(content_words(" ".join(w[4] for w in page.get_text("words", clip=clip))))

    def is_code(page, rects):
        clip = pymupdf.Rect(rects[0])
        for r in rects[1:]:
            clip |= r
        for b in page.get_text("dict", clip=clip).get("blocks", []):
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    if sp.get("text", "").strip() and MONO_FONT.search(sp.get("font", "")):
                        return True
        return False

    def quality_checks(k, pno, page, a, rects):
        tag = f"p{pno} annotation #{k} ({plain(a.get('title', '')) or a['kind']})"
        body = a.get("note", "")
        for fld in ("note", "answer", "short"):
            m = EXISTENCE_RE.search(plain(a.get(fld, "")))
            if m and not a.get("checked"):
                errors.append(f"{tag}: says {m.group(0)!r} but has no 'checked' (what you searched and what came "
                              "back). Never infer that something is fake or nonexistent from not recognizing it; "
                              "search, then either cite the result or say 'I could not verify this'.")
        other = bool(a.get("outside") or a.get("links"))
        for fld in ("note", "short", "title", "answer"):
            check_quotes(plain(a.get(fld, "")), f"{tag} ('{fld}')", attributed="note", other_source=other)
        if a.get("outside") and not (a.get("links") or a.get("checked")):
            errors.append(f"{tag}: marked 'outside' (not from the paper) but has no link or 'checked'. Label "
                          "outside knowledge and give it a source.")
        if a["kind"] == "check":
            return
        parts = note_parts(body)
        labels = [lab for lab, _ in parts if lab]
        if not any(is_impl(lab) for lab in labels):
            errors.append(f"{tag}: no implication part. Add one of: Why it matters, Takeaway, Supports?, "
                          "Trade-off, Limit, If it fails, Actually, Point ... A note must say why the thing "
                          "matters, not only what it is.")
        total = sum(len(t.split()) for _, t in parts) or 1
        desc = sum(len(t.split()) for lab, t in parts if is_desc(lab))
        if desc / total > DESCRIPTIVE_MAX and total > 25:
            errors.append(f"{tag}: {desc / total:.0%} of the note is description ('Means'/'Does'/'Shows' or "
                          f"unlabeled). Cut the restatement; spend the words on why it matters.")
        src = anchor_words(page, rects, a)
        for lab, t in parts:
            if is_desc(lab):
                cw = content_words(t)
                if len(cw) >= 8:
                    share = sum(1 for w in cw if w in src) / len(cw)
                    if share >= RESTATE_MAX:
                        errors.append(f"{tag}: the '{lab or 'opening'}' part restates the annotated text "
                                      f"({share:.0%} of its words are copied from it). Say what it implies instead.")
        names = plain(a.get("title", "")) + " " + " ".join(a.get("covers", []))
        lead = labels[0] if labels else ""
        if (is_code(page, rects) or re.search(r"(?i)\b(eq\.?|equation|listing|code)\b", names)) \
                and not is_impl(lead):
            errors.append(f"{tag}: code/equation notes must LEAD with the implication (e.g., 'Why it matters.' or "
                          f"'Point.'), not with '{lead or 'an unlabeled restatement'}'. Say what the example "
                          "demonstrates before tracing it.")
        fk = a.get("fig_kind")
        if fk:
            if fk not in FIG_PARTS:
                errors.append(f"{tag}: fig_kind must be 'data' or 'diagram'")
                return
            req = FIG_PARTS[fk]
            pos = [(has_part(labels, r) or [-1])[0] for r in req]
            if -1 in pos or pos != sorted(pos):
                errors.append(f"{tag}: a {fk} figure/table note needs the parts "
                              + " -> ".join(r.title() for r in req) + " in that order (labels found: "
                              + ", ".join(labels) + ").")
            if fk == "data":
                bad = [l for l in labels if l.split(" ")[0] in ("shows", "layout", "columns", "rows", "axes")]
                if bad:
                    errors.append(f"{tag}: data figure notes may not describe the layout ({bad}); give the metric, "
                                  "the baseline it is compared against, and the takeaway.")
                for lab, t in parts:
                    if not lab.startswith("takeaway") and t.count(",") >= 5:
                        warnings.append(f"{tag}: the '{lab}' part reads like a list of rows/columns; "
                                        "name only what the reader needs for the comparison.")
        elif FIGTAB_RE.search(plain(a.get("title", ""))) and a["kind"] == "key":
            warnings.append(f"{tag}: names a figure/table but has no fig_kind; if this is its main note, add one.")

    def eff_status(lk):
        rep = report.get(lk.get("url", ""))
        return "script" if rep and rep.get("ok") else lk.get("status")

    note_num = 0
    check_num = 0
    continuations = {}
    fills = {}
    target_w = max(pg.rect.width for pg in src_doc) + args.gutter
    for pno in sorted(per_page):
        page = doc[pno - 1]
        W, H = page.rect.width, page.rect.height
        col = text_column(page_lines_cache[pno], page.rect)
        items = sorted(per_page[pno], key=lambda t: (t[0], t[1]))

        # ---- marks (highlight / bar), as real annotations so they can be toggled
        notes = []
        for y0, k, a, rects in items:
            kind = a["kind"]
            label, hl, accent, _ = KINDS[kind]
            # Only key and subtle items are highlighted by default; explanatory
            # notes get just a numbered badge so highlights stay sparse.
            default = ("bar" if "lines" in a and len(rects) > 3 else "highlight") \
                if kind in ("key", "subtle") else "none"
            mark = a.get("mark") or default
            popup = plain(a.get("title", "")) + (": " if a.get("title") else "") + plain(a.get("note", ""))
            if mark == "highlight":
                annot = page.add_highlight_annot([r for r in rects])
                annot.set_colors(stroke=hl)
                annot.set_opacity(0.55)
                annot.set_info(title=label, content=popup)
                annot.update()
            elif mark == "bar":
                x = max(col[0] - 9, 4)
                top, bot = min(r.y0 for r in rects), max(r.y1 for r in rects)
                annot = page.add_line_annot((x, top), (x, bot))
                annot.set_colors(stroke=accent)
                annot.set_border(width=2.5)
                annot.set_info(title=label, content=popup)
                annot.update()
            elif mark != "none":
                errors.append(f"p{pno} annotation #{k}: mark must be highlight, bar, or none")
            if a.get("note"):
                quality_checks(k, pno, page, a, rects)
                note_body = a["note"]
                if a.get("links"):
                    bits = []
                    for lk in a["links"]:
                        tag = lk.get("type", "link")
                        if eff_status(lk) in ("search", "expected"):
                            tag += " · not opened"
                        bits.append(f"<a href='{html.escape(lk['url'])}'>[{tag}]</a>")
                    note_body = note_body + " <b>See.</b> " + " ".join(bits)
                a = dict(a, note=note_body)
                if re.search(r"\bL\d+(\s*-\s*L?\d+)?\b", plain(a["note"])):
                    warnings.append(f"p{pno} annotation #{k}: note text contains a line ID like 'L12'; "
                                    "readers can't see those. Refer to equations, lines of text, or pages.")
                long_note = len(plain(a["note"]).split()) > SHORT_LIMIT
                short = a.get("short") or a.get("lead")
                if long_note and not short:
                    errors.append(f"p{pno} annotation #{k} ({plain(a.get('title', '')) or a['kind']}): "
                                  f"note is over {SHORT_LIMIT} words, so it needs a 'short' version "
                                  "(1-2 sentences, the single most useful point) for the gutter.")
                note_num += 1
                ci = None
                if kind == "check":
                    check_num += 1
                    ci = check_num
                notes.append({"num": note_num, "kind": kind, "title": a.get("title", ""),
                              "body": a["note"], "check": ci, "short": short,
                              "answer": a.get("answer") if kind == "check" else None,
                              "spilled": False,
                              "anchor_y": y0, "anchor_y1": max(r.y1 for r in rects)})

        checks_here = [n for n in notes if n["kind"] == "check" and n.get("answer")]
        for cn in checks_here:
            ans_cw = set(content_words(plain(cn["answer"])))
            if len(ans_cw) < 6:
                continue
            for n in notes:
                if n is cn or n["kind"] == "check":
                    continue
                nb = set(content_words(plain(n["body"])))
                if ans_cw and len(ans_cw & nb) / len(ans_cw) > GIVEAWAY_MAX:
                    warnings.append(f"p{pno}: note {n['num']} ({plain(n['title'])[:30]}) shares most of its "
                                    f"content words with the answer to Q{cn.get('check')}; it may give the answer "
                                    "away. Move the question earlier, or make it ask for more than that note says.")
        if not notes:
            continue

        # ---- widen the page to make a gutter (to one uniform width: pages a few points narrower than the
        # widest used to get a second gutter in the "widen un-annotated pages" pass below)
        page.set_mediabox(pymupdf.Rect(0, 0, target_w, H))
        bx0, bx1 = W + 10, W + args.gutter - 12
        inner_w = bx1 - bx0 - 2 * PAD - 3

        def measure_all(css):
            for n in notes:
                n["qh"] = n["lh"] = n["ah"] = 0
                if n.get("spilled"):
                    n["html"] = stub_html(n)
                    n["qh"] = measure(n["html"], inner_w, css, archive)
                else:
                    n["html"] = note_html(n["num"], n["kind"], n["title"], n["body"], n["check"])
                    n["qh"] = measure(n["html"], inner_w, css, archive)
                    if n.get("answer"):
                        n["lh"] = measure(ANSWER_LABEL, inner_w, css, archive)
                        n["ah"] = measure(answer_html(n["answer"]), inner_w, css, archive) + 1
                n["h"] = n["qh"] + n["lh"] + n["ah"] + 2 * PAD
            over = layout(notes, TOP_MARGIN, H - BOTTOM_MARGIN)
            total = sum(n["h"] for n in notes) + BOX_GAP * (len(notes) - 1)
            return max(over, total - MAX_FILL * (H - TOP_MARGIN - BOTTOM_MARGIN))

        size = FONT_SIZES[0]
        css = css_for(size, base_css)
        over = measure_all(css)
        # Full notes stay in the gutter whenever they fit. Only when the page
        # overflows do notes move to a notes page, fewest first: each step moves
        # the note whose short version saves the most height.
        while over > 0.5:
            cands = [n for n in notes if not n["spilled"] and n["short"]]
            if not cands:
                break
            gain = {}
            for n in cands:
                n["spilled"] = True
                gain[n["num"]] = n["h"] - (measure(stub_html(n), inner_w, css, archive) + 2 * PAD)
                n["spilled"] = False
            best = max(cands, key=lambda n: gain[n["num"]])
            best["spilled"] = True
            over = measure_all(css)
        for n in notes:
            if n["spilled"]:
                continuations.setdefault(pno, []).append(n)
        if over > 0.5:
            errors.append(f"p{pno}: gutter overflows by {over:.0f}pt even after moving every note that has "
                          "a 'short' version to a notes page. Add 'short' to more notes on this page, or "
                          "shorten the notes that stay in the gutter.")
        used = sum(n["h"] for n in notes) + BOX_GAP * (len(notes) - 1)
        fills[pno] = used / (H - TOP_MARGIN - BOTTOM_MARGIN)

        # badge positions: next to the anchor, nudged down if two collide
        last_badge = -1e9
        for n in notes:
            ay = (n["anchor_y"] + min(n["anchor_y1"], n["anchor_y"] + 12)) / 2
            ay = max(ay, last_badge + 11.5)
            n["badge_y"] = last_badge = ay

        shape = page.new_shape()
        # gutter separator
        shape.draw_line((W + 2, TOP_MARGIN - 10), (W + 2, H - BOTTOM_MARGIN + 10))
        shape.finish(color=(0.85, 0.85, 0.85), width=0.5)
        for n in notes:
            label, hl, accent, tint = KINDS[n["kind"]]
            box = pymupdf.Rect(bx0, n["y"], bx1, n["y"] + n["h"])
            shape.draw_rect(box)
            shape.finish(fill=tint, color=None)
            shape.draw_rect(pymupdf.Rect(bx0, box.y0, bx0 + 2.5, box.y1))
            shape.finish(fill=accent, color=None)
            # badge beside the text column + leader into the gutter
            badge = pymupdf.Point(col[1] + 9, n["badge_y"])
            shape.draw_circle(badge, 5.2)
            shape.finish(fill=accent, color=None)
            shape.draw_line((badge.x + 5.2, badge.y), (bx0, box.y0 + 7))
            shape.finish(color=accent, width=0.5, dashes="[1.5 1.5] 0")
        shape.commit()
        for n in notes:
            ay = n["badge_y"]
            page.insert_text((col[1] + 9 - (2.3 if n["num"] < 10 else 4.3), ay + 2.4),
                             str(n["num"]), fontsize=6.5, fontname="hebo", color=(1, 1, 1))
            y = n["y"] + PAD
            page.insert_htmlbox(pymupdf.Rect(bx0 + 3 + PAD, y, bx1 - PAD, y + n["qh"] + 20), n["html"],
                                css=css, archive=archive, scale_low=1)
            if n["ah"]:
                y += n["qh"]
                page.insert_htmlbox(pymupdf.Rect(bx0 + 3 + PAD, y, bx1 - PAD, y + n["lh"] + 4), ANSWER_LABEL,
                                    css=css, archive=archive, scale_low=1)
                y += n["lh"]
                page.insert_htmlbox(pymupdf.Rect(bx0 + 3 + PAD, y, bx1 - PAD, y + n["ah"]),
                                    answer_html(n["answer"]), css=css, archive=archive, scale_low=1, rotate=180)

    # ---- notes pages: full text of long notes, right after their page,
    # in note order, as many pages as needed (descending so indices stay valid)
    cont_pages = {}
    ccss = css_for(9.5, base_css) + "h1 {font-size: 13pt; margin: 0 0 8pt 0;}"
    for pno in sorted(continuations, reverse=True):
        base = doc[pno - 1]
        pw, ph = base.rect.width, base.rect.height
        area = pymupdf.Rect(72, 60, min(pw - 72, 72 + 440), ph - 50)
        items = sorted(continuations[pno], key=lambda n: n["num"])
        blocks = []
        for n in items:
            q = note_html(n["num"], n["kind"], n["title"], n["body"], n["check"])
            qh = measure(q, area.width, ccss, archive)
            lh = measure(ANSWER_LABEL, area.width, ccss, archive) if n.get("answer") else 0
            ah = measure(answer_html(n["answer"]), area.width, ccss, archive) + 1 if n.get("answer") else 0
            blocks.append((q, qh, n.get("answer"), lh, ah))
        pages, cur, used = [], [], 30.0  # 30pt for the heading
        for b in blocks:
            h = b[1] + b[3] + b[4] + 12
            if cur and used + h > area.height:
                pages.append(cur)
                cur, used = [], 30.0
            if h > area.height - 30:
                errors.append(f"notes page for p{pno}: a single note is too long for a page; shorten it")
            cur.append(b)
            used += h
        if cur:
            pages.append(cur)
        for i, blist in enumerate(pages):
            cp = doc.new_page(pno + i, width=pw, height=ph)
            head = f"Notes on p. {pno}" + (" (continued)" if i else "")
            cp.insert_htmlbox(pymupdf.Rect(area.x0, area.y0, area.x1, area.y0 + 30), f"<h1>{head}</h1>",
                              css=ccss, archive=archive, scale_low=1)
            y = area.y0 + 30
            for q, qh, ans, lh, ah in blist:
                cp.insert_htmlbox(pymupdf.Rect(area.x0, y, area.x1, y + qh + 20), q, css=ccss,
                                  archive=archive, scale_low=1)
                y += qh
                if ans:
                    cp.insert_htmlbox(pymupdf.Rect(area.x0, y, area.x1, y + lh + 4), ANSWER_LABEL, css=ccss,
                                      archive=archive, scale_low=1)
                    y += lh
                    cp.insert_htmlbox(pymupdf.Rect(area.x0, y, area.x1, y + ah), answer_html(ans), css=ccss,
                                      archive=archive, scale_low=1, rotate=180)
                    y += ah
                y += 12
        cont_pages[pno] = len(pages)
    inserted_before = lambda p: sum(n for q, n in cont_pages.items() if q < p)

    # ---- widen un-annotated pages too so the document has a uniform width
    for page in doc:
        if abs(page.rect.width - target_w) > 0.5:
            page.set_mediabox(pymupdf.Rect(0, 0, target_w, page.rect.height))

    full_w = max(p.rect.width for p in doc)
    H0 = doc[0].rect.height
    css = css_for(10, base_css) + """
h1 {font-size: 17pt; margin: 0 0 6pt 0;} h2 {font-size: 12pt; margin: 12pt 0 4pt 0;}
li {margin-bottom: 4pt;} .muted {color: #666;}
"""

    back_toc = []  # (title, pre-guide 1-based page) to add to bookmarks

    # ---- critique page (domain soundness review)
    cr = spec.get("critique")
    if cr and cr.get("items"):
        vcol = {"ok": "#2f7d32", "concern": "#c77700", "flaw": "#b00020",
                "cant-tell": "#555555", "not-applicable": "#888888"}
        def basis(it):
            for f, lab in (("claim", "Claim affected"), ("evidence", "Evidence in the paper"),
                           ("missing", "Could not see"), ("why_na", "Why not applicable")):
                if it.get(f):
                    return f"<br><span class='muted'><b>{lab}:</b> {it[f]}</span>"
            return ""
        cp = doc.new_page(-1, width=full_w, height=H0)
        back_toc.append(("Critique", len(doc)))
        dom = html.escape(cr.get("domain", "general"))
        body = (f"<h1>Critique</h1><p class='muted'>Soundness review using the {dom} reading checklist. "
                "Verdicts: ok (the paper does this; evidence cited), concern (a weakness that limits a claim), flaw "
                "(the evidence cannot support a central claim), cant-tell (evidence may exist but could not be seen), "
                "not-applicable.</p><table "
                "style='width:100%; border-collapse:collapse;'>"
                "<tr><th align='left'>Check</th><th align='left'>Verdict</th>"
                "<th align='left'>Where</th><th align='left'>What the paper does</th></tr>")
        for it in cr["items"]:
            c = vcol.get(it.get("verdict"), "#333")
            body += (f"<tr style='border-top:0.5pt solid #ccc;'><td>{html.escape(plain(it['check']))}</td>"
                     f"<td><b style='color:{c}'>{html.escape(it['verdict'])}</b></td>"
                     f"<td class='muted'>{html.escape(plain(it['where']))}</td>"
                     f"<td>{it['note']}{basis(it)}</td></tr>")
        body += "</table>"
        spare, _ = cp.insert_htmlbox(pymupdf.Rect(50, 50, full_w - 50, H0 - 40), body,
                                     css=css, archive=archive, scale_low=0.55)
        if spare < 0:
            errors.append("critique page overflows; shorten the notes")

    # ---- key references page (verified links to cited works)
    cws = spec.get("cited_works")
    if cws:
        kp = doc.new_page(-1, width=full_w, height=H0)
        back_toc.append(("Key references", len(doc)))
        counts = {}
        for cw in cws:
            counts[eff_status(cw)] = counts.get(eff_status(cw), 0) + 1
        names = {"script": "opened by the build script", "opened": "opened during research",
                 "search": "seen only in search results", "expected": "expected but not opened"}
        summary = "; ".join(f"{counts[k]} {names[k]}" for k in names if counts.get(k))
        body = ("<h1>Key references</h1><p class='muted'>Cited works worth reading. Each line under a link says "
                f"exactly how that link was checked; nothing here claims a check that did not happen. "
                f"Link checks: {summary}.</p>")
        for i, cw in enumerate(cws, 1):
            u = html.escape(cw["url"])
            where = f" <span class='muted'>({html.escape(cw['where'])})</span>" if cw.get("where") else ""
            st = eff_status(cw)
            if st == "script":
                rep = report[cw["url"]]
                chk = SCRIPT_STATUS.format(http=rep.get("http")) + "'" + html.escape(rep.get("title") or "") + "'"
            elif st == "expected":
                chk = LINK_STATUS["expected"] + html.escape(cw.get("basis", ""))
            else:
                chk = LINK_STATUS.get(st, "") + "'" + html.escape(cw.get("seen", "")) + "'"
                if cw.get("via"):
                    chk += f" (via {html.escape(cw['via'])})"
            body += (f"<p style='margin-bottom:8pt'><b>{i}.</b> {html.escape(cw['cite'])}{where}<br>"
                     f"<a href='{u}' style='color:#1a55aa'>{u}</a><br>"
                     f"<b>Why.</b> {cw['why']}<br>"
                     f"<span class='muted'><b>Link check.</b> {chk}</span></p>")
        spare, _ = kp.insert_htmlbox(pymupdf.Rect(60, 50, full_w - 60, H0 - 40), body,
                                     css=css, archive=archive, scale_low=0.6)
        if spare < 0:
            errors.append("key references page overflows; link fewer works")

    # ---- reading guide page (front)
    guide = spec.get("guide")
    if guide:
        gp = doc.new_page(0, width=full_w, height=H0)
        title = spec.get("title", "")
        body = "<h1>Reading guide</h1>"
        if title:
            body += f"<p class='muted' style='font-size:12pt'>{html.escape(title)}</p>"
        body += f"<p>{guide.get('summary', '')}</p>"
        if guide.get("route"):
            body += "<h2>Reading route</h2><ol>"
            for r in guide["route"]:
                pg = f" <span class='muted'>(p. {r['page']})</span>" if r.get("page") else ""
                depth = f" <span class='muted'>[{r['read']}]</span>" if r.get("read") else ""
                body += f"<li><b>{r['what']}</b>{pg}{depth}<br>{r.get('why', '')}</li>"
            body += "</ol>"
        if guide.get("priorities"):
            body += "<h2>Where to spend your attention</h2><ol>"
            for pr in guide["priorities"]:
                body += f"<li><b>{pr['what']}</b> <span class='muted'>(p. {pr['page']})</span><br>{pr.get('why', '')}</li>"
            body += "</ol>"
        if guide.get("skim"):
            body += f"<h2>Fine to skim</h2><p>{guide['skim']}</p>"
        body += "<h2>Key</h2><p>"
        for kind, (label, hl, accent, tint) in KINDS.items():
            hexc = "#%02x%02x%02x" % tuple(int(c * 255) for c in accent)
            body += f"<b style='color:{hexc}'>■ {label}</b>&nbsp;&nbsp; "
        body += ("</p><p class='muted'>Highlights and margin bars are PDF annotations (hover for the note; "
                 "delete or hide them in your reader). A highlight with no numbered badge marks a key point; "
                 "hover over it for one line on why it matters. Numbered notes sit in the right gutter. Each Check "
                 "question has its answer printed upside down right under it: answer first, then turn the page. "
                 "A link tag marked 'not opened' was not opened during preparation (see Key references). "
                 "Page numbers refer to the original paper; this guide page is extra.</p>")
        spare, _ = gp.insert_htmlbox(pymupdf.Rect(60, 50, full_w - 60, H0 - 40), body,
                                     css=css, archive=archive, scale_low=0.7)
        if spare < 0:
            errors.append("guide page overflows; shorten it")
        offset = 1
    else:
        offset = 0

    # ---- bookmarks: guide + original TOC (shifted) + answers
    toc = [[lvl, t, p + offset + inserted_before(p)] for lvl, t, p in pymupdf.open(args.pdf).get_toc(simple=True)]
    if guide:
        toc.insert(0, [1, "Reading guide", 1])
    for bt_title, bt_pre in back_toc:
        toc.append([1, bt_title, bt_pre + offset])
    try:
        doc.set_toc(toc)
    except Exception as e:  # malformed original TOC shouldn't block output
        warnings.append(f"could not set bookmarks: {e}")

    # insert_htmlbox embeds the note font once per call; subsetting collapses
    # the copies (a 20+ MB file becomes roughly the input size).
    doc.subset_fonts()
    doc.save(args.out, garbage=4, deflate=True, use_objstms=1)

    if args.render:
        rd = Path(args.render)
        rd.mkdir(parents=True, exist_ok=True)
        out_doc = pymupdf.open(args.out)
        # Name each render by the paper's own page number (out_p27.png is the paper's p. 27, whatever was inserted
        # before it); notes pages are out_p27_notes1.png, ...; added pages are out_guide/out_critique/out_refs.png.
        names = (["out_guide.png"] if offset else [])
        for p_ in range(1, n_pages + 1):
            names.append(f"out_p{p_}.png")
            names += [f"out_p{p_}_notes{k}.png" for k in range(1, cont_pages.get(p_, 0) + 1)]
        names += ["out_critique.png" if t == "Critique" else "out_refs.png" for t, _ in back_toc]
        if len(names) != len(out_doc):
            warnings.append("render names could not be matched to pages; renders are numbered by position")
            names = [f"out_{i}.png" for i in range(1, len(out_doc) + 1)]
        for name, pg in zip(names, out_doc):
            m = re.match(r"out_p(\d+)", name)
            if args.part and m and not (sel[0] <= int(m.group(1)) <= sel[1]):
                continue   # --part: only the chunk's pages and their notes pages
            pg.get_pixmap(dpi=args.dpi).save(rd / name)

    print(f"wrote {args.out}: {note_num} notes ({check_num} check questions, answers inline) on {len(fills)} pages")
    all_links = [lk for a in anns for lk in (a.get("links") or [])] + list(spec.get("cited_works") or [])
    st_counts = {}
    for lk in all_links:
        st_counts[eff_status(lk)] = st_counts.get(eff_status(lk), 0) + 1
    if density:
        dmax = max(density, key=density.get)
        print(f"page review: {len(density)} pages, {n_points} key points, {n_marked} new highlights; "
              f"densest p{dmax} at {density[dmax]:.0%} of body text highlighted")
        # per-page coverage, so a sparse stretch is visible at a glance (and comparable across chunks)
        print("  page  words  points  notes  highlighted")
        for pno_, nw_, np_, nn_, d_, th_ in cov_rows:
            print(f"  p{pno_:<4} {nw_:>5} {np_:>7} {nn_:>6} {d_:>11.1%}" + ("  thin" if th_ else ""))
    shown = [w for w in wts.values() if not args.part or (w["_pages"][0] <= sel[1] and sel[0] <= w["_pages"][1])]
    if shown:
        print(("walkthroughs touching this chunk: " if args.part else "walkthroughs: ") + ", ".join(
            f"{w['id']} (pp. {w['_pages'][0]}-{w['_pages'][1]})" + ("" if w["_passages"] is None else f" -> {w['file']}")
            for w in shown))
    print("link checks: " + ", ".join(f"{k} {v}" for k, v in sorted(st_counts.items(), key=lambda x: str(x[0]))))
    moved = {p: len(v) for p, v in continuations.items()}
    if moved:
        print("  notes pages added after: " + ", ".join(
            f"p{p} ({n} note(s) moved, gutter {fills[p]:.0%} full)" for p, n in sorted(moved.items())))
    else:
        print(f"  all notes fit beside the text; fullest gutter {max(fills.values(), default=0):.0%}")
    if coverage:
        n_ann = sum(1 for c in coverage if c[2] == "annotated")
        n_skip = sum(1 for c in coverage if c[2] == "skipped")
        print(f"coverage: {len(coverage)} formal objects, {n_ann} annotated, {n_skip} skipped with reason, "
              f"{len(missing)} missing")
        print(f"error audit: {len(coverage) - len(unaudited)} of {len(coverage)} objects checked, {flagged} flagged")
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)
    if errors:
        print("\nERRORS:\n" + "\n".join(errors), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
