"""BookIndex part 1: the document tree T.

BookRAG builds T from layout blocks (MinerU) + an LLM that assigns heading
levels. The Ewha regulations have an explicit legal hierarchy, so heading levels
are assigned by rules instead (no LLM, deterministic):

    root
    ├── 본칙 (main text)
    │   └── 제N장 (chapter)            level 1
    │       └── 제N조 (article)         level 2   <- retrieval unit
    ├── 부칙 (addenda)
    │   └── 부칙(YYYY. M. D. 개정)       level 1   <- retrieval unit
    └── 별표 (appendix tables)
        └── [별표 N] (one per version)  level 1   <- retrieval unit

Layout parsing (MinerU replacement):
  * body/addenda text  : PDF text layer via pypdfium2 (exact characters)
  * appendix tables    : Upstage Document Parse (tables -> markdown), cached;
                         falls back to the raw text layer if the API fails.
"""
import json
import os
import re

import pypdfium2 as pdfium

import config

_HEADER = [re.compile(r"^\s*이화여자대학교\s*학칙\s*$"),
           re.compile(r"^\s*\d+\s*-\s*\d+\s*-\s*\d+\s*$")]
# "제N장 title" at a line start, or mid-line right after a sentence ("...한다. 제16장")
CHAPTER_RE = re.compile(r"(?:^|(?<=다\.)\s)\s*(제(\d+)장)\s*([^\n]*)", re.M)
# An article heading "제N조(제목)" or "제N조의M(제목)". In the text layer a new
# article may start mid-line right after the previous sentence ends.
ARTICLE_RE = re.compile(r"(?:(?<=^)|(?<=[.)\]]\s)|(?<=다\.)\s?)(제(\d+)조(?:의(\d+))?)\(([^)]{1,40})\)", re.M)
# [ \t]* (not \s*) at line starts so a match never swallows the previous line
# break, which would attribute the heading to the previous page.
ADDENDUM_RE = re.compile(r"^[ \t]*부\s*칙[ \t]*(\([^)]*\))?[ \t]*$|^[ \t]*부\s*칙\s*\(", re.M)
# Document Parse may wrap the heading in markdown: "## **[별표 1] ...**"
TABLE_RE = re.compile(r"^[ \t#*]*(\[\s*별\s*표\s*(\d+)\s*\])", re.M)
DATE_RE = re.compile(r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})")
AMEND_RE = re.compile(r"\((?:개정|신설|전문개정|본조신설|제목개정)[^)]*\)|\[(?:전문개정|본조신설|제목개정)[^\]]*\]")


def norm_date(y, m, d):
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def dates_in(text):
    return sorted({norm_date(*m) for m in DATE_RE.findall(text)})


# -----------------------------------------------------------------------------
# layout parsing
# -----------------------------------------------------------------------------
def pdf_pages(pdf_path=config.PDF_PATH):
    """List of (page_no, text) with running headers/page numbers removed."""
    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    for i, page in enumerate(pdf, 1):
        raw = page.get_textpage().get_text_range()
        lines = [ln for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                 if not any(p.match(ln) for p in _HEADER)]
        pages.append((i, "\n".join(lines)))
    pdf.close()
    return pages


def table_pages_markdown(pdf_path=config.PDF_PATH, use_api=True):
    """{page_no: markdown} for the appendix-table pages via Document Parse.

    Only the table pages are sent (a sub-PDF), and the result is cached in
    index/docparse_tables.json so the API is called once.
    """
    cache = os.path.join(config.INDEX_DIR, "docparse_tables.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    if not use_api:
        return {}
    import llm  # local import: tree building without the API stays possible
    src = pdfium.PdfDocument(pdf_path)
    sub = pdfium.PdfDocument.new()
    pages = list(config.DOCPARSE_PAGES)
    sub.import_pages(src, [p - 1 for p in pages])
    import io
    buf = io.BytesIO()
    sub.save(buf)
    result = llm.document_parse(buf.getvalue(), "ewha_tables.pdf")
    out = {}
    for el in result.get("elements", []):
        if el.get("category") in ("header", "footer"):
            continue
        page_no = pages[int(el.get("page", 1)) - 1]
        md = (el.get("content") or {}).get("markdown") or (el.get("content") or {}).get("text", "")
        out.setdefault(page_no, []).append(md)
    out = {p: "\n\n".join(v) for p, v in out.items()}
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# -----------------------------------------------------------------------------
# tree construction
# -----------------------------------------------------------------------------
def _node(nid, kind, title, text="", page=None, parent=None, **meta):
    return {"id": nid, "kind": kind, "title": title, "text": text.strip(),
            "page": page, "parent": parent, "children": [], **meta}


def _with_pages(pages):
    """Concatenate pages, returning text and a char-offset -> page lookup."""
    text, starts = "", []
    for no, t in pages:
        starts.append((len(text), no))
        text += t + "\n"
    def page_at(pos):
        cur = starts[0][1]
        for off, no in starts:
            if off > pos:
                break
            cur = no
        return cur
    return text, page_at


def build_tree(pdf_path=config.PDF_PATH, use_api=True):
    pages = pdf_pages(pdf_path)
    first_table_page = min(config.TABLE_PAGES)
    body_pages = [(n, t) for n, t in pages if n < first_table_page]
    text, page_at = _with_pages(body_pages)

    # split 본칙 / 부칙 at the first standalone "부칙" heading
    m = ADDENDUM_RE.search(text)
    main_text, add_text = text[:m.start()], text[m.start():]
    add_offset = m.start()

    nodes = {"root": _node("root", "root", "이화여자대학교 학칙 (2017. 8. 16. 개정)")}
    def add(node):
        nodes[node["id"]] = node
        nodes[node["parent"]]["children"].append(node["id"])

    add(_node("main", "part", "본칙", parent="root"))
    add(_node("addenda", "part", "부칙", parent="root"))
    add(_node("tables", "part", "별표", parent="root"))

    # --- chapters and articles ------------------------------------------------
    chapter_id = None
    heads = []  # (pos, kind, match)
    last_ch = 0
    for cm in CHAPTER_RE.finditer(main_text):   # monotonic: skips references
        if int(cm.group(2)) == last_ch + 1:
            last_ch += 1
            heads.append((cm.start(1), "chapter", cm))
    last_no = (0, 0)
    for am in ARTICLE_RE.finditer(main_text):
        no = (int(am.group(2)), int(am.group(3) or 0))
        if no <= last_no:      # cross-references like "제31조(재입학)" in text
            continue
        last_no = no
        heads.append((am.start(1), "article", am))
    heads.sort(key=lambda h: h[0])

    preamble_end = heads[0][0] if heads else len(main_text)
    nodes["main"]["text"] = main_text[:preamble_end].strip()   # 제정/개정일
    for i, (pos, kind, hm) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(main_text)
        body = main_text[pos:end]
        if kind == "chapter":
            chapter_id = f"ch{hm.group(2)}"
            title = f"{hm.group(1)} {AMEND_RE.sub('', hm.group(3)).strip()}"
            add(_node(chapter_id, "chapter", title, page=page_at(pos), parent="main",
                      amend_dates=dates_in(hm.group(3))))
        else:
            label = hm.group(1)
            nid = "art" + hm.group(2) + (f"-{hm.group(3)}" if hm.group(3) else "")
            add(_node(nid, "article", f"{label}({hm.group(4)})", text=body,
                      page=page_at(pos), parent=chapter_id or "main",
                      article=label, amend_dates=dates_in(" ".join(AMEND_RE.findall(body)))))

    # --- addenda ---------------------------------------------------------------
    starts = [a.start() for a in ADDENDUM_RE.finditer(add_text)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(add_text)
        block = add_text[s:e].strip()
        head = block.split("\n", 1)[0]
        hd = dates_in(head)
        nid = f"add{i:03d}"
        add(_node(nid, "addendum", re.sub(r"\s+", " ", head).strip(), text=block,
                  page=page_at(add_offset + s), parent="addenda",
                  date=hd[0] if hd else None,
                  effective_dates=dates_in(block)))

    # --- appendix tables ---------------------------------------------------------
    md = table_pages_markdown(pdf_path, use_api=use_api)
    md = {p: v for p, v in md.items() if p in config.DOCPARSE_PAGES}
    table_pages = [(n, md.get(n) or t) for n, t in pages if n >= first_table_page]
    ttext, tpage_at = _with_pages(table_pages)
    tstarts = [(t.start(1), t.group(2)) for t in TABLE_RE.finditer(ttext)]
    if not tstarts:   # Document Parse may drop the bracket; fall back to "별표 N"
        tstarts = [(t.start(), t.group(1))
                   for t in re.finditer(r"^\W*별\s*표\s*(\d+)", ttext, re.M)]
    for i, (s, num) in enumerate(tstarts):
        e = tstarts[i + 1][0] if i + 1 < len(tstarts) else len(ttext)
        block = ttext[s:e].strip()
        head = block.split("\n", 1)[0]
        nid = f"tab{num}-{i:02d}"
        years = sorted(set(re.findall(r"(\d{4})\s*학년도", block)))
        head = re.sub(r"[#*]", "", head)
        add(_node(nid, "table", re.sub(r"\s+", " ", head).strip()[:80],
                  text=block, page=tpage_at(s), parent="tables", table_no=int(num),
                  amend_dates=dates_in(head), years=years,
                  source="document-parse" if tpage_at(s) in md else "pdf-text"))
    return nodes


def leaves(nodes):
    """Retrieval units: articles, addenda blocks, tables."""
    return [n for n in nodes.values() if n["kind"] in ("article", "addendum", "table")]


def path_of(nodes, nid):
    """Human-readable breadcrumb, e.g. '본칙 > 제8장 휴학… > 제28조(제적)'."""
    parts = []
    while nid and nid != "root":
        parts.append(nodes[nid]["title"])
        nid = nodes[nid]["parent"]
    return " > ".join(reversed(parts))


# -----------------------------------------------------------------------------
# structural edges (rule-based, no LLM)
# -----------------------------------------------------------------------------
REF_RE = re.compile(r"제(\d+)조(?:의(\d+))?")


def structural_edges(nodes):
    """Tree-node links used to expand retrieval:
    * cross-reference : node text mentions 제N조 -> that article
    * amendment       : article amended/inserted on date D <-> 부칙 dated D
    """
    arts = {n["id"]: n for n in nodes.values() if n["kind"] == "article"}
    adds_by_date = {}
    for n in nodes.values():
        if n["kind"] == "addendum" and n.get("date"):
            adds_by_date.setdefault(n["date"], []).append(n["id"])
    edges = set()
    for n in leaves(nodes):
        body = n["text"]
        if n["kind"] == "article":   # skip the article's own heading
            body = body[len(n["article"]):]
        for a, b in REF_RE.findall(body):
            target = "art" + a + (f"-{b}" if b else "")
            if target in arts and target != n["id"]:
                edges.add((n["id"], target, "refers"))
    for a in arts.values():
        for d in a.get("amend_dates", []):
            for add_id in adds_by_date.get(d, []):
                edges.add((a["id"], add_id, "amended_by"))
    return sorted(edges)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    t = build_tree(use_api="--api" in sys.argv)
    kinds = {}
    for n in t.values():
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    print(kinds)
    print("structural edges:", len(structural_edges(t)))
    for nid in ["art28", "art31", "add000", "tab1-00"]:
        if nid in t:
            n = t[nid]
            print(f"\n[{nid}] {path_of(t, nid)} p{n['page']} {n.get('amend_dates') or n.get('date')}\n{n['text'][:200]}")
