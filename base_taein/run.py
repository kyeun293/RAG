"""Baseline QA runner.

Usage (from inside base_taein/):
    python run.py --input data/test_samples.csv --output responses.csv --mode zeroshot
    python run.py --input data/test_samples.csv --output responses.csv --mode rag --k 5
    python ../eval.py responses.csv data/test_samples.csv

    # MMLU-Pro domain subset (law/psychology/business/philosophy/history):
    python make_mmlu_pro_subset.py --per-domain 400 --seed 42
    python run.py --input data/mmlu_pro_subset.csv --output resp_mmlu_b0.csv \
                  --mode zeroshot --workers 8

Modes
    zeroshot : B0. Prompt only, no retrieval. 1 LLM call per question.
    rag      : B1. Naive RAG: top-k fixed-size chunks of ewha.pdf by cosine
               similarity, for every question. 1 LLM call per question.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from llm import CHAT_MODEL, chat
from utils import (DEFAULT_LETTER, extract_letter, format_response,
                   has_hangul, option_letters)

SYSTEM_ZEROSHOT = (
    "You are an expert at answering multiple-choice questions. "
    "Questions may be written in Korean or English. "
    "Reason briefly, then finish with exactly one final line in the form "
    "[ANSWER]: (X), where X is one of the valid option letters."
)

SYSTEM_RAG = (
    "You are an expert at answering multiple-choice questions. "
    "Questions may be written in Korean or English. "
    "You are given context passages retrieved from the Ewha Womans University "
    "regulations (이화여자대학교 학칙). Use them when they are relevant; if they "
    "are not relevant, rely on your own knowledge. "
    "Reason briefly, then finish with exactly one final line in the form "
    "[ANSWER]: (X), where X is one of the valid option letters."
)


def build_messages(question, letters, context_chunks=None):
    valid = ", ".join(letters)
    if context_chunks is None:
        system = SYSTEM_ZEROSHOT
        user = f"Question:\n{question}\n\nValid options: {valid}"
    else:
        system = SYSTEM_RAG
        context = "\n\n".join(f"[{i}] {text}"
                              for i, (_, _, text) in enumerate(context_chunks, 1))
        user = (f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\nValid options: {valid}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def answer_one(question, mode, retriever=None, k=5):
    """Returns (letter, log_record). Never raises: falls back to DEFAULT_LETTER."""
    letters = option_letters(question)
    record = {"mode": mode, "valid_letters": letters, "llm_calls": 0}
    try:
        chunks = retriever.search(question, k=k) if mode == "rag" else None
        if chunks is not None:
            record["retrieved"] = [{"chunk_id": cid, "score": round(score, 4),
                                    "text": text} for score, cid, text in chunks]
        messages = build_messages(question, letters, chunks)
        record["llm_calls"] += 1
        raw = chat(messages)
        record["raw"] = raw
        letter = extract_letter(raw, letters)
        record["parse_ok"] = letter is not None
    except Exception as exc:  # API error, timeout, etc.
        record["error"] = f"{type(exc).__name__}: {exc}"
        letter = None
    if letter is None:
        letter = DEFAULT_LETTER
        record["fallback"] = True
    record["letter"] = letter
    return letter, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test_samples.csv")
    parser.add_argument("--output", default="responses.csv")
    parser.add_argument("--mode", choices=["zeroshot", "rag"], default="zeroshot")
    parser.add_argument("--k", type=int, default=5, help="top-k chunks (rag mode)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N questions (quick test)")
    parser.add_argument("--log", default=None, help="jsonl log path")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel questions (API concurrency)")
    parser.add_argument("--resume", action="store_true",
                        help="skip questions already finished in the log")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    if args.limit:
        df = df.head(args.limit)
    if "question_id" in df.columns:
        ids = df["question_id"].astype(str).tolist()
    else:
        ids = [f"row-{i:04d}" for i in range(1, len(df) + 1)]

    retriever = None
    if args.mode == "rag":
        from retriever import NaiveRetriever
        retriever = NaiveRetriever()
        print(f"Indexed {len(retriever.chunks)} chunks from ewha.pdf")

    log_path = args.log or os.path.join(
        "logs", f"{os.path.splitext(os.path.basename(args.output))[0]}_{args.mode}.jsonl")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # Checkpointing: every finished question is appended to the log at once.
    # With --resume, questions already in the log (without an API error) are
    # skipped, so an interrupted run can be continued.
    done = {}
    if args.resume and os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially written last line
                if "error" not in rec:
                    done[rec["question_id"]] = rec
        print(f"Resuming: {len(done)} questions already done")
    lock = threading.Lock()
    ckpt = open(log_path, "a" if args.resume else "w", encoding="utf-8")

    def work(item):
        qid, question = item
        if qid in done:
            return done[qid]["letter"], done[qid]
        start = time.time()
        letter, record = answer_one(question, args.mode, retriever, args.k)
        record.update(question_id=qid, model=CHAT_MODEL,
                      seconds=round(time.time() - start, 2))
        flag = " (fallback)" if record.get("fallback") else ""
        with lock:
            ckpt.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"{qid}: ({letter}){flag}", flush=True)
        return letter, record

    items = list(zip(ids, df["prompts"].astype(str)))
    # map() returns results in input order, so output order == input order.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(work, items))
    ckpt.close()

    # Rewrite the log cleanly in input order (drops stale/errored duplicates).
    responses = []
    with open(log_path, "w", encoding="utf-8") as log:
        for letter, record in results:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            responses.append(format_response(letter))

    assert len(responses) == len(df), "one response per input row"
    pd.DataFrame({"question_id": ids, "responses": responses}).to_csv(
        args.output, index=False, encoding="utf-8")
    print(f"Saved {len(responses)} responses to {args.output}; log: {log_path}")

    # Quick breakdown when labels are available (practice data only).
    if "answers" in df.columns:
        gold = df["answers"].astype(str).str.strip().tolist()
        pred = [f"({r[-2]})" for r in responses]
        korean = [has_hangul(q) for q in df["prompts"].astype(str)]
        groups = [("all", [True] * len(df)),
                  ("korean (Ewha)", korean),
                  ("english (general)", [not x for x in korean])]
        if "category" in df.columns:
            cats = df["category"].astype(str).tolist()
            groups += [(c, [x == c for x in cats]) for c in sorted(set(cats))]
        print(f"  fallbacks: {sum(r.get('fallback', False) for _, r in results)}")
        for name, mask in groups:
            idx = [i for i, m in enumerate(mask) if m]
            if idx:
                acc = sum(pred[i] == gold[i] for i in idx) / len(idx)
                print(f"  acc {name:18s}: {acc * 100:5.1f}%  ({len(idx)} q)")


if __name__ == "__main__":
    main()
