"""BookRAG-style query-time retrieval over the BookIndex B = (T, G, M).

Operators (paper -> here):
  Formulator  Extract/Decompose   1 Solar call: query type + entities + sub-questions
  Selector    Filter_Range/Modal  rule-based: 제N조 refs, dates -> 부칙, 별표/학년도 -> tables
              Select_by_Entity    KG entity match -> tree nodes through GT-Link
  Reasoner    Graph_Reasoning     personalized PageRank on G (+ structural tree edges)
              Text_Reasoning      dense (Solar embedding, option-aware) + char-bigram BM25
              Skyline_Ranker      Pareto frontier over (graph score, text score)
  Synthesizer Map / Reduce        answer call (global questions: map-reduce)

Per question: 2 Solar calls (single/multi) or up to 4 (global), always <= 5.
"""
import json
import math
import os
import re
from collections import Counter

import numpy as np

import config
import llm
from graph import EntityGraph, _unit
from tree import REF_RE, dates_in, path_of

FORMULATOR_SYSTEM = """너는 이화여자대학교 학칙 질의의 검색 계획을 세운다. 답을 풀지 말고 계획만 JSON으로 출력하라.
- type: "single"(한 조항으로 답함) | "multi"(여러 조항/부칙/별표를 결합하거나 계산 필요) | "global"(문서 전체에서 여러 항목을 찾아 세거나 비교)
- entities: 질문과 보기에서 학칙 검색에 쓸 핵심 개념 (예: "재입학", "통산 휴학기간", "학사경고", "계절학기 학점")
- sub_questions: multi/global이면 학칙에서 각각 찾아야 할 하위 질문 (최대 4개), single이면 []
- focus: 근거가 있을 곳 목록, 값은 "본칙" | "부칙" | "별표"
형식: {"type": "...", "entities": [...], "sub_questions": [...], "focus": [...]}"""

ANSWER_SYSTEM = """You answer multiple-choice questions about the Ewha Womans University regulations (이화여자대학교 학칙, 2017. 8. 16. 개정).
Use the retrieved passages as the primary evidence; each is labeled with its location (본칙 조문 / 부칙 / 별표).
- 본칙 = current rules. 부칙(날짜) = transitional rules and effective dates (시행일) of each amendment.
- For calculations (grades, credits, years, counts), write out each step explicitly.
- For "옳지 않은 것 / 아닌 것" questions, check every option against the passages.
Reason briefly in Korean, then finish with exactly one final line: [ANSWER]: (X)"""

MAP_SYSTEM = """You extract evidence. From the passages, list every fact relevant to the question (quote numbers, dates, article labels exactly). Do not answer the question."""


# -----------------------------------------------------------------------------
class BM25:
    """Character-bigram BM25 (robust for Korean without a tokenizer)."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.toks = [self.tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for ts in self.toks for t in set(ts))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(ts) for ts in self.toks]

    @staticmethod
    def tok(text):
        s = re.sub(r"\s+", "", text)
        return [s[i:i + 2] for i in range(len(s) - 1)]

    def scores(self, query):
        q = set(self.tok(query))
        out = np.zeros(len(self.toks), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            L = len(self.toks[i])
            for t in q:
                if t in tf:
                    f = tf[t]
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * L / self.avg))
        return out


def split_question(prompt):
    """(stem, [option texts]) from 'QUESTIONn) ...\\n(A) ...'."""
    body = re.sub(r"^\s*QUESTION\s*\d+\)\s*", "", prompt)
    parts = re.split(r"(?:^|[\n\u2028])[ \t]*\(([A-J])\)\s*", body)
    stem, opts = parts[0].strip(), []
    for i in range(1, len(parts) - 1, 2):
        opts.append(parts[i + 1].strip())
    return stem, opts


def _norm(scores):
    if not scores:
        return {}
    hi = max(scores.values())
    return {k: (v / hi if hi > 0 else 0.0) for k, v in scores.items()}


def skyline(cands):
    """Pareto frontier over (graph, text): keep nodes no other node dominates."""
    keep = []
    for a, (ga, ta) in cands.items():
        dominated = any((gb >= ga and tb >= ta) and (gb > ga or tb > ta)
                        for b, (gb, tb) in cands.items() if b != a)
        if not dominated:
            keep.append(a)
    return keep


# -----------------------------------------------------------------------------
class BookRetriever:
    def __init__(self, index_dir=config.INDEX_DIR, kg_mode="names"):
        P = lambda n: os.path.join(index_dir, n)  # noqa: E731
        data = json.load(open(P("tree.json"), encoding="utf-8"))
        self.tree, self.edges = data["nodes"], data["edges"]
        nv = np.load(P("node_vecs.npz"))
        self.ids = [str(i) for i in nv["ids"]]
        self.vecs = nv["vecs"]
        self.pos = {nid: i for i, nid in enumerate(self.ids)}
        self.bm25 = BM25([f"{self.tree[i]['title']} {self.tree[i]['text']}" for i in self.ids])
        self.nbrs = {}
        for a, b, rel in self.edges:
            self.nbrs.setdefault(a, []).append((b, rel))
            self.nbrs.setdefault(b, []).append((a, rel))
        self.kg = None                     # kg_mode "none": tree-only ablation
        if kg_mode != "none":
            g = json.load(open(P(f"graph_{kg_mode}.json"), encoding="utf-8"))
            self.kg = EntityGraph(g, np.load(P(f"entity_vecs_{kg_mode}.npz"))["vecs"])
        self.adds_by_date, self.tables = {}, []
        for nid in self.ids:
            n = self.tree[nid]
            if n["kind"] == "addendum" and n.get("date"):
                self.adds_by_date.setdefault(n["date"], []).append(nid)
            if n["kind"] == "table":
                self.tables.append(nid)

    # ---- Formulator ----------------------------------------------------------
    def formulate(self, prompt, budget):
        raw = llm.chat([{"role": "system", "content": FORMULATOR_SYSTEM},
                        {"role": "user", "content": prompt}],
                       budget=budget, max_tokens=1024, json_mode=True)
        plan = llm.parse_json(raw) or {}
        qtype = plan.get("type") if plan.get("type") in ("single", "multi", "global") else "single"
        ents = [str(e) for e in plan.get("entities", []) if str(e).strip()][:8]
        subs = [str(s) for s in plan.get("sub_questions", []) if str(s).strip()][:4]
        focus = [f for f in plan.get("focus", []) if f in ("본칙", "부칙", "별표")]
        return {"type": qtype, "entities": ents, "sub_questions": subs, "focus": focus,
                "raw": raw}

    # ---- Selector (rule-based filters) --------------------------------------
    def rule_filters(self, stem, options):
        forced, reasons = {}, []
        text_all = stem + "\n" + "\n".join(options)
        for a, b in REF_RE.findall(text_all):
            nid = "art" + a + (f"-{b}" if b else "")
            if nid in self.pos:
                forced[nid] = 1.0
                reasons.append(f"ref:{nid}")
        # dates written in the stem -> 부칙 of that date (+ articles amended then)
        stem_dates = dates_in(stem) + [
            f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            for y, m, d in re.findall(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", stem)]
        for d in stem_dates:
            for nid in self.adds_by_date.get(d, []):
                forced[nid] = 1.0
                reasons.append(f"date:{d}->{nid}")
        # 별표 / 입학정원 / 학년도 -> tables (filtered by year when given)
        if re.search(r"별\s*표|입학정원|정원", text_all):
            years = set(re.findall(r"(\d{4})\s*학년도", text_all))
            for nid in self.tables:
                tn = self.tree[nid]
                m = re.search(r"별\s*표\s*(\d)", text_all)
                if m and int(m.group(1)) != tn.get("table_no"):
                    continue
                if years and not (years & set(tn.get("years", []))) and tn.get("table_no") == 1:
                    continue
                forced[nid] = 0.8
                reasons.append(f"table:{nid}")
        return forced, reasons

    # ---- full pipeline -------------------------------------------------------
    def retrieve(self, prompt, budget, use_formulator=True):
        stem, options = split_question(prompt)
        plan = (self.formulate(prompt, budget) if use_formulator
                else {"type": "single", "entities": [], "sub_questions": [], "focus": []})
        forced, reasons = self.rule_filters(stem, options)

        # Text_Reasoning: option-aware dense + lexical
        queries = [stem] + [f"{stem}\n{o}" for o in options] + plan["sub_questions"]
        qv = _unit(llm.embed(queries + plan["entities"], config.EMBED_QUERY_MODEL))
        dense = (self.vecs @ qv[:len(queries)].T).max(1)
        lex = self.bm25.scores(stem + " " + " ".join(options + plan["entities"]))
        lex = lex / (lex.max() or 1.0)
        text = 0.7 * dense + 0.3 * lex
        focus_kinds = {"본칙": "article", "부칙": "addendum", "별표": "table"}
        if plan["focus"]:
            boost = {focus_kinds[f] for f in plan["focus"]}
            text = text + np.array([0.03 if self.tree[i]["kind"] in boost else 0.0
                                    for i in self.ids], dtype=np.float32)

        # Graph_Reasoning: Select_by_Entity + PPR over the KG, mapped by GT-Link
        graph = {}
        if self.kg is not None and plan["entities"]:
            ev = qv[len(queries):]
            seeds = self.kg.match(plan["entities"], ev)
            graph = self.kg.node_scores(self.kg.ppr(seeds))
        graph = _norm(graph)
        # structural expansion (cross-references, article <-> its 부칙)
        for nid in list(forced):
            for nb, _ in self.nbrs.get(nid, []):
                graph[nb] = max(graph.get(nb, 0.0), 0.5)
        for nid, w in forced.items():
            graph[nid] = max(graph.get(nid, 0.0), w)

        # candidate pool
        pool = set(forced) | {n for n in graph if n in self.pos}
        pool |= {self.ids[i] for i in np.argsort(-text)[:config.DENSE_POOL]}
        cands = {n: (graph.get(n, 0.0), float(text[self.pos[n]])) for n in pool}

        # Skyline_Ranker (+ forced nodes, + top text nodes up to MIN_CONTEXT_NODES)
        sky = skyline(cands)
        rank = lambda n: cands[n][0] * 0.5 + cands[n][1]  # noqa: E731
        chosen = list(dict.fromkeys(
            sorted(forced, key=rank, reverse=True) + sorted(sky, key=rank, reverse=True)))
        for n in sorted(cands, key=lambda n: cands[n][1], reverse=True):
            if len(chosen) >= config.MIN_CONTEXT_NODES:
                break
            if n not in chosen:
                chosen.append(n)
        # second pass: fill with structural neighbours of top text hits
        for n in sorted(cands, key=rank, reverse=True):
            if len(chosen) >= config.MAX_CONTEXT_NODES:
                break
            if n not in chosen and cands[n][0] > 0:
                chosen.append(n)

        # Effective-date questions: follow article -> 부칙 links (amended_by) of
        # the top articles, so "언제 시행" finds the addendum that introduced it.
        if re.search(r"시행|부칙|개정|신설", stem):
            top_arts = [n for n in chosen if self.tree[n]["kind"] == "article"][:2]
            linked = {nb for a in top_arts for nb, rel in self.nbrs.get(a, [])
                      if rel == "amended_by"}
            linked = sorted(linked, key=lambda n: -float(text[self.pos[n]]))[:3]
            for n in linked:   # register in the score table used by the trace
                cands.setdefault(n, (max(graph.get(n, 0.0), 0.5), float(text[self.pos[n]])))
            chosen = list(dict.fromkeys(chosen[:config.MAX_CONTEXT_NODES - len(linked)]
                                        + linked + chosen))
            reasons += [f"amended_by:{n}" for n in linked]

        out, size = [], 0
        limit = config.MAX_CONTEXT_CHARS * (2 if plan["type"] == "global" else 1)
        for n in chosen[:config.MAX_CONTEXT_NODES * (2 if plan["type"] == "global" else 1)]:
            t = self.tree[n]["text"]
            if out and size + len(t) > limit:
                continue
            out.append(n)
            size += len(t)
        # present in document order (tree order is the order of self.ids)
        out.sort(key=lambda n: self.pos[n])
        trace = {"plan": {k: v for k, v in plan.items() if k != "raw"},
                 "rules": reasons, "skyline": len(sky),
                 "nodes": [{"id": n, "graph": round(cands[n][0], 3),
                            "text": round(cands[n][1], 3)} for n in out]}
        return out, plan, trace

    def render(self, nids):
        blocks = []
        for i, n in enumerate(nids, 1):
            node = self.tree[n]
            blocks.append(f"[{i}] {path_of(self.tree, n)} (p.{node['page']})\n{node['text']}")
        return "\n\n".join(blocks)


def answer(retriever, prompt, letters, budget):
    nids, plan, trace = retriever.retrieve(prompt, budget)
    context = retriever.render(nids)
    valid = ", ".join(letters)
    if plan["type"] == "global" and budget.left >= 3 and len(nids) > 4:
        half = len(nids) // 2
        notes = []
        for part in (nids[:half], nids[half:]):          # Map
            notes.append(llm.chat(
                [{"role": "system", "content": MAP_SYSTEM},
                 {"role": "user", "content": f"Passages:\n{retriever.render(part)}\n\nQuestion:\n{prompt}"}],
                budget=budget, max_tokens=1500))
        context = "\n\n".join(f"[Evidence {i}]\n{t}" for i, t in enumerate(notes, 1))
        trace["map_reduce"] = True
    raw = llm.chat([{"role": "system", "content": ANSWER_SYSTEM},          # Reduce
                    {"role": "user", "content":
                     f"Retrieved passages:\n{context}\n\nQuestion:\n{prompt}\n\nValid options: {valid}"}],
                   budget=budget)
    return raw, trace
