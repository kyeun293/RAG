"""BookIndex parts 2-3: knowledge graph G and GT-Link M (entity -> tree nodes).

* Entity/relation extraction : Solar Mini 4, several tree nodes per call (offline).
* Entity resolution          : BookRAG's gradient-based ER, using Solar
                               embedding cosine instead of a reranker model, and
                               taking the top candidate instead of an LLM tie-break.
* GT-Link                     : every entity keeps the set of tree nodes it came
                               from; merged entities union their node sets.
* Graph reasoning (query time): personalized PageRank over G (numpy).
"""
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
import llm
from tree import leaves, path_of

EXTRACT_SYSTEM = """너는 대학 학칙 문서에서 지식 그래프를 추출하는 도우미다.
각 노드(조문/부칙/별표)에서 질의응답에 유용한 핵심 개체와 관계를 뽑아라.

개체(entity) 유형: 제도, 신분, 기간, 학점, 성적, 기관, 조직, 문서, 절차, 수치, 날짜, 기타
- 이름은 문서 표현 그대로 짧게 (예: "재입학", "통산 휴학기간", "계절학기", "학사경고").
- 수치/기간/날짜는 의미를 붙여서 (예: "재입학 횟수 1회", "시행일 2017-09-01").
관계(relation): 두 개체 사이의 사실을 짧은 서술로 (예: "제한한다", "적용된다", "시행일").

반드시 아래 JSON 형식으로만 답하라:
{"nodes": [{"node_id": "...", "entities": [{"name": "...", "type": "...", "desc": "한 문장 설명"}],
            "relations": [{"src": "...", "tgt": "...", "rel": "..."}]}]}"""


def _batches(nodes, max_chars=config.KG_BATCH_CHARS):
    batch, size = [], 0
    for n in nodes:
        cost = len(n["text"]) + 80
        if batch and size + cost > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(n)
        size += cost
    if batch:
        yield batch


def _extract(batch, tree):
    body = "\n\n".join(
        f"### node_id: {n['id']}\n경로: {path_of(tree, n['id'])}\n{n['text'][:4000]}"
        for n in batch)
    msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": body}]
    for attempt in range(2):
        out = llm.parse_json(llm.chat(msgs, budget=None, json_mode=True))
        if out and isinstance(out.get("nodes"), list):
            return out["nodes"]
    return []


def extract_raw(tree, workers=8, log=print, existing=None):
    """[{node_id, entities, relations}] for every leaf node.

    Batched extraction, then a second pass node-by-node for anything the model
    dropped (long JSON outputs can get truncated). `existing` records are kept.
    """
    valid = set(tree)
    records = {r["node_id"]: r for r in (existing or []) if r.get("node_id") in valid}
    for max_chars in (config.KG_BATCH_CHARS, 0):          # 0 -> one node per call
        todo = [n for n in leaves(tree) if n["id"] not in records]
        if not todo:
            break
        batches = list(_batches(todo, max_chars))
        log(f"KG extraction: {len(todo)} nodes in {len(batches)} Solar calls")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rs in pool.map(lambda b: _extract(b, tree), batches):
                for r in rs:
                    if isinstance(r, dict) and r.get("node_id") in valid:
                        records.setdefault(r["node_id"], r)
    missing = [n["id"] for n in leaves(tree) if n["id"] not in records]
    if missing:
        log(f"KG extraction: still missing {len(missing)} nodes: {missing[:10]}")
    return list(records.values())


# -----------------------------------------------------------------------------
# variant "names": Solar returns only entity names per node (short output)
# -----------------------------------------------------------------------------
NAMES_SYSTEM = """너는 대학 학칙 문서의 색인(찾아보기)을 만든다.
각 노드에서 질의응답 검색에 유용한 핵심 개념 이름을 3~6개 뽑아라.
- 문서 표현 그대로 짧게 (예: "재입학", "통산 휴학기간", "계절학기", "학사경고", "입학정원").
- 설명이나 관계는 쓰지 마라.
반드시 JSON으로만: {"<node_id>": ["개념1", "개념2", ...], ...}"""


def _extract_names(batch, tree):
    body = "\n\n".join(
        f"### node_id: {n['id']}\n경로: {path_of(tree, n['id'])}\n{n['text'][:3000]}"
        for n in batch)
    msgs = [{"role": "system", "content": NAMES_SYSTEM},
            {"role": "user", "content": body}]
    for _ in range(2):
        out = llm.parse_json(llm.chat(msgs, budget=None, json_mode=True))
        if isinstance(out, dict) and out:
            return [{"node_id": k, "entities": [str(x) for x in v if str(x).strip()]}
                    for k, v in out.items() if isinstance(v, list)]
    return []


def extract_names(tree, workers=8, log=print, existing=None):
    valid = set(tree)
    records = {r["node_id"]: r for r in (existing or []) if r.get("node_id") in valid}
    for max_chars in (config.KG_BATCH_CHARS_NAMES, config.KG_BATCH_CHARS):
        todo = [n for n in leaves(tree) if n["id"] not in records]
        if not todo:
            break
        batches = list(_batches(todo, max_chars))
        log(f"KG names extraction: {len(todo)} nodes in {len(batches)} Solar calls")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rs in pool.map(lambda b: _extract_names(b, tree), batches):
                for r in rs:
                    if r["node_id"] in valid:
                        records.setdefault(r["node_id"], r)
    return list(records.values())


# -----------------------------------------------------------------------------
# variant "rule": no LLM at all
# -----------------------------------------------------------------------------
_JOSA = sorted("""은 는 이 가 을 를 에 의 로 으로 와 과 도 만 에서 에게 까지 부터 보다 이나 나
으로서 으로써 로서 로써 에는 에도 에서는 와의 과의 이라 라 이며 하며 한다 된다 할 될 하는
되는 한 된 하여 되어 하고 되고 함 됨 이다 인 적""".split(), key=len, reverse=True)
_STOP = set("""경우 사항 다만 각호 각 호 해당 따로 이하 이상 이내 이후 이전 기타 관련 규정 본교 총장
학칙 시행 개정 신설 삭제 부칙 제정 날 때 자 것 수 등 및 또는 그 이 위 아래 같다 정한다""".split())


_STOP |= set("""의하 의하여 하여 위하 위하여 대하 대하여 따라 따른 있는 있을 있다 없는 없다 둔다
받은 받아 받을 그러나 다음 해당하 이를 이에 같은 모든 어느""".split())
_VERB_END = tuple("다 여 는 을 며 면 게 서도 려".split())


def _terms(text):
    out = []
    for tok in re.findall(r"[가-힣A-Za-z·]{2,}", text):
        for j in _JOSA:
            if tok.endswith(j):
                if len(tok) - len(j) < 2:   # "회에", "조에": 1-char stem -> drop
                    tok = ""
                else:
                    tok = tok[:-len(j)]
                break
        if (2 <= len(tok) <= 10 and tok not in _STOP
                and not tok.endswith(_VERB_END)):
            out.append(tok)
    return out


def extract_rule(tree, per_node=6):
    """Entities without an LLM:
    * article titles  : 제31조(재입학) -> "재입학" (split on ·/,)
    * definitions     : (이하 "X"라 한다), "X"이라 한다
    * tf-idf terms    : document-frequency filtered content words per node
    """
    units = leaves(tree)
    docs = {n["id"]: _terms(n["text"]) for n in units}
    df = Counter(t for ts in docs.values() for t in set(ts))
    N = len(units)
    recs = []
    for n in units:
        ents = []
        if n["kind"] == "article":
            m = re.search(r"\(([^)]+)\)", n["title"])
            if m:
                ents += [p.strip() for p in re.split(r"[·,및]", m.group(1)) if len(p.strip()) >= 2]
        ents += re.findall(r"[\"“'‘]([^\"”'’]{2,15})[\"”'’]\s*(?:이)?라\s*한다", n["text"])
        tf = Counter(docs[n["id"]])
        scored = sorted(((c * math.log(N / df[t]), t) for t, c in tf.items()
                         if 2 <= df[t] <= max(3, N // 8)), reverse=True)
        ents += [t for _, t in scored[:per_node]]
        recs.append({"node_id": n["id"], "entities": list(dict.fromkeys(ents))})
    return recs


def _nums(name):
    return [int(x) for x in re.findall(r"\d+", str(name))]


def _key(name):
    return "".join(str(name).split()).lower()


def resolve(raw, log=print):
    """Gradient-based entity resolution + GT-Link. Returns graph dict."""
    # 1) exact-key merge
    ents = {}          # key -> {name, type, desc, nodes}
    order = []
    for rec in raw:
        for e in rec.get("entities", []):
            if isinstance(e, str):
                e = {"name": e}
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            if not name:
                continue
            k = _key(name)
            if k not in ents:
                ents[k] = {"name": name, "type": e.get("type", "기타"),
                           "desc": e.get("desc", ""), "nodes": set()}
                order.append(k)
            ents[k]["nodes"].add(rec["node_id"])

    # 2) gradient-based ER on embeddings: walk the sorted similarity list of
    #    already-accepted entities; keep candidates until the first sharp drop
    #    (s_i < g * s_{i-1}); merge if exactly one strong candidate remains.
    texts = [f"{ents[k]['name']}: {ents[k]['desc']}" for k in order]
    vecs = _unit(llm.embed(texts, config.EMBED_PASSAGE_MODEL)) if texts else np.zeros((0, 1))
    canon = {}                       # key -> canonical key
    accepted, acc_vecs = [], []
    for i, k in enumerate(order):
        target = None
        if accepted:
            sims = np.asarray(acc_vecs) @ vecs[i]
            top = np.argsort(-sims)[:config.ER_TOP_K]
            cands = []
            for j, idx in enumerate(top):
                s = float(sims[idx])
                if s < config.ER_MIN_SIM:
                    break
                if j > 0 and s < config.ER_GRADIENT * float(sims[top[j - 1]]):
                    break
                # embeddings barely separate numbers: "시행일 1982-03-01" vs
                # "시행일 1966-02-01" must never merge
                if _nums(ents[accepted[idx]]["name"]) == _nums(ents[k]["name"]):
                    cands.append(accepted[idx])
            if cands:
                target = cands[0]
        if target is None:
            canon[k] = k
            accepted.append(k)
            acc_vecs.append(vecs[i])
        else:
            canon[k] = target
            ents[target]["nodes"] |= ents[k]["nodes"]
    log(f"entities: {len(order)} raw -> {len(accepted)} after ER")

    # 3) relations between canonical entities
    edges = {}
    for rec in raw:
        for r in rec.get("relations", []):
            if not isinstance(r, dict):
                continue
            a, b =canon.get(_key(r.get("src", ""))), canon.get(_key(r.get("tgt", "")))
            if a and b and a != b:
                edges.setdefault((a, b), set()).add(str(r.get("rel", "")))
    # co-occurrence in the same node (weak link)
    for rec in raw:
        names = [e if isinstance(e, str) else e.get("name", "")
                 for e in rec.get("entities", []) if isinstance(e, (str, dict))]
        ks = sorted({canon[_key(nm)] for nm in names if _key(nm) in canon})
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                edges.setdefault((ks[x], ks[y]), set()).add("co-occur")

    ent_list = [{"key": k, "name": ents[k]["name"], "type": ents[k]["type"],
                 "desc": ents[k]["desc"], "nodes": sorted(ents[k]["nodes"])}
                for k in accepted]
    idx = {k: i for i, k in enumerate(accepted)}
    vec_by_key = {k: acc_vecs[i] for i, k in enumerate(accepted)}
    rel_list = [{"src": idx[a], "tgt": idx[b], "rels": sorted(r)}
                for (a, b), r in edges.items()]
    return {"entities": ent_list, "relations": rel_list}, np.asarray(
        [vec_by_key[k] for k in accepted], dtype=np.float32)


def _unit(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


class EntityGraph:
    def __init__(self, graph, entity_vecs):
        self.entities = graph["entities"]
        self.vecs = entity_vecs
        n = len(self.entities)
        # IDF-like specificity: generic hubs ("시행", "부칙", "학칙") linked to
        # many tree nodes get little weight in propagation and node scoring.
        n_nodes = max(1, len({x for e in self.entities for x in e["nodes"]}))
        self.spec = np.array([math.log(1 + n_nodes / max(1, len(e["nodes"])))
                              for e in self.entities], dtype=np.float32)
        self.spec /= self.spec.max() if n else 1.0
        self.A = np.zeros((n, n), dtype=np.float32)
        for r in graph["relations"]:
            w = 0.3 if r["rels"] == ["co-occur"] else 1.0
            w *= float(self.spec[r["src"]] * self.spec[r["tgt"]])
            self.A[r["src"], r["tgt"]] += w
            self.A[r["tgt"], r["src"]] += w
        self.name_index = {_key(e["name"]): i for i, e in enumerate(self.entities)}

    def match(self, names, name_vecs, min_sim=config.ENTITY_MATCH_SIM):
        """Query entity strings -> KG entity ids (exact/substring + embedding)."""
        hits = {}
        for name, v in zip(names, name_vecs):
            k = _key(name)
            if k in self.name_index:
                hits[self.name_index[k]] = 1.0
            for ek, i in self.name_index.items():
                if len(k) >= 2 and (k in ek or ek in k):
                    hits[i] = max(hits.get(i, 0), 0.9)
            if len(self.vecs):
                sims = self.vecs @ v
                for i in np.argsort(-sims)[:5]:
                    if sims[i] >= min_sim:
                        hits[int(i)] = max(hits.get(int(i), 0), float(sims[i]))
        return hits

    def ppr(self, seeds, alpha=config.PPR_ALPHA, iters=30):
        """Personalized PageRank from {entity_id: weight}."""
        n = len(self.entities)
        if not seeds or n == 0:
            return np.zeros(n, dtype=np.float32)
        p = np.zeros(n, dtype=np.float32)
        for i, w in seeds.items():
            p[i] = w * self.spec[i]
        if p.sum() <= 0:
            return np.zeros(n, dtype=np.float32)
        p /= p.sum()
        deg = self.A.sum(1, keepdims=True)
        P = np.divide(self.A, deg, out=np.zeros_like(self.A), where=deg > 0)
        r = p.copy()
        for _ in range(iters):
            r = alpha * (P.T @ r) + (1 - alpha) * p
        return r

    def node_scores(self, ent_scores):
        """Map entity scores onto tree nodes through GT-Link."""
        out = {}
        for i in np.nonzero(ent_scores > 0)[0]:
            for nid in self.entities[i]["nodes"]:
                out[nid] = out.get(nid, 0.0) + float(ent_scores[i] * self.spec[i])
        return out


def save(graph, vecs, path_json, path_npz):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=1)
    np.savez(path_npz, vecs=vecs)
