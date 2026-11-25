"""Offline BookIndex construction for ewha.pdf.

    python index.py            # build everything (cached steps are reused)
    python index.py --no-kg    # tree + embeddings only (no LLM calls)
    python index.py --rebuild  # ignore caches

Outputs (index/):
    docparse_tables.json  Document Parse markdown for the appendix-table pages
    tree.json             document tree T (+ structural edges)
    kg_raw_{mode}.json    raw entity extraction per node (mode = full|names|rule)
    graph_{mode}.json / entity_vecs_{mode}.npz   resolved KG G with GT-Link M
    node_vecs.npz         Solar embeddings of retrieval nodes
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

import config
import graph as kg
import llm
from tree import build_tree, leaves, path_of, structural_edges

P = lambda name: os.path.join(config.INDEX_DIR, name)  # noqa: E731


def node_passage(tree, n):
    return f"{path_of(tree, n['id'])}\n{n['text']}"


def build(args):
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    if args.rebuild:
        for f in os.listdir(config.INDEX_DIR):
            os.remove(P(f))

    tree = build_tree(use_api=not args.no_docparse)
    edges = structural_edges(tree)
    with open(P("tree.json"), "w", encoding="utf-8") as f:
        json.dump({"nodes": tree, "edges": edges}, f, ensure_ascii=False, indent=1)
    units = leaves(tree)
    print(f"tree: {len(units)} retrieval nodes, {len(edges)} structural edges")

    # node embeddings (cache keyed by content)
    passages = [node_passage(tree, n) for n in units]
    digest = hashlib.md5("\n\0".join(passages).encode()).hexdigest()[:10]
    if os.path.exists(P("node_vecs.npz")) and str(np.load(P("node_vecs.npz"))["digest"]) == digest:
        print("node embeddings: cached")
    else:
        vecs = kg._unit(llm.embed(passages, config.EMBED_PASSAGE_MODEL))
        np.savez(P("node_vecs.npz"), vecs=vecs, ids=np.array([n["id"] for n in units]),
                 digest=digest)
        print(f"node embeddings: {vecs.shape}")

    if args.no_kg:
        return
    modes = config.KG_MODES if args.kg_mode == "all" else [args.kg_mode]
    for mode in modes:
        raw_path = P(f"kg_raw_{mode}.json")
        existing = (json.load(open(raw_path, encoding="utf-8"))
                    if os.path.exists(raw_path) else [])
        if mode == "full":     # fills gaps only; cached records are kept
            raw = kg.extract_raw(tree, workers=args.workers, existing=existing)
        elif mode == "names":
            raw = kg.extract_names(tree, workers=args.workers, existing=existing)
        else:
            raw = kg.extract_rule(tree)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=1)
        print(f"[{mode}] raw extraction: {len(raw)}/{len(units)} node records")
        g, evecs = kg.resolve(raw)
        kg.save(g, evecs, P(f"graph_{mode}.json"), P(f"entity_vecs_{mode}.npz"))
        print(f"[{mode}] graph: {len(g['entities'])} entities, {len(g['relations'])} relations")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-kg", action="store_true", help="skip LLM graph extraction")
    ap.add_argument("--no-docparse", action="store_true",
                    help="use the PDF text layer for tables instead of Document Parse")
    ap.add_argument("--kg-mode", default="all", choices=list(config.KG_MODES) + ["all"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    build(ap.parse_args())
