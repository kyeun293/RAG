"""BookRAG runner (same CLI/output as base_taein/run.py).

    python index.py                       # once: build the BookIndex
    python run.py --input ../data/ewha_test_full.csv --output resp_ewha_bookrag.csv --workers 8
    python ../../eval.py resp_ewha_bookrag.csv ../data/ewha_test_full.csv

Each question uses at most config.MAX_LLM_CALLS_PER_QUESTION (=5) Solar calls;
the count is enforced by llm.CallBudget and recorded in the jsonl log.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from llm import CallBudget
from retrieve import BookRetriever, answer
from utils import DEFAULT_LETTER, extract_letter, format_response, option_letters


def answer_one(retriever, question):
    letters = option_letters(question)
    budget = CallBudget()
    record = {"valid_letters": letters}
    try:
        raw, trace = answer(retriever, question, letters, budget)
        record.update(trace=trace, raw=raw)
        letter = extract_letter(raw, letters)
        record["parse_ok"] = letter is not None
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        letter = None
    record["llm_calls"] = budget.used
    if letter is None:
        letter, record["fallback"] = DEFAULT_LETTER, True
    record["letter"] = letter
    return letter, record


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/ewha_test_full.csv")
    ap.add_argument("--output", default="resp_ewha_bookrag.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--kg", default="names", choices=list(config.KG_MODES) + ["none"],
                    help="which entity graph to use (none = tree only)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    if args.limit:
        df = df.head(args.limit)
    ids = (df["question_id"].astype(str).tolist() if "question_id" in df.columns
           else [f"row-{i:04d}" for i in range(1, len(df) + 1)])

    retriever = BookRetriever(kg_mode=args.kg)
    print(f"BookIndex: {len(retriever.ids)} nodes, "
          f"{len(retriever.kg.entities) if retriever.kg else 0} entities")

    log_path = args.log or os.path.join(
        "logs", os.path.splitext(os.path.basename(args.output))[0] + ".jsonl")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    done = {}
    if args.resume and os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in rec:
                done[rec["question_id"]] = rec
        print(f"Resuming: {len(done)} done")
    lock = threading.Lock()
    ckpt = open(log_path, "a" if args.resume else "w", encoding="utf-8")

    def work(item):
        qid, q = item
        if qid in done:
            return done[qid]["letter"], done[qid]
        t0 = time.time()
        letter, rec = answer_one(retriever, q)
        rec.update(question_id=qid, model=config.CHAT_MODEL, kg=args.kg,
                   seconds=round(time.time() - t0, 2))
        with lock:
            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()
            flag = " (fallback)" if rec.get("fallback") else ""
            print(f"{qid}: ({letter}) calls={rec['llm_calls']}{flag}", flush=True)
        return letter, rec

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(work, zip(ids, df["prompts"].astype(str))))
    ckpt.close()
    with open(log_path, "w", encoding="utf-8") as f:
        for _, rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    responses = [format_response(letter) for letter, _ in results]
    assert len(responses) == len(df)
    pd.DataFrame({"question_id": ids, "responses": responses}).to_csv(
        args.output, index=False, encoding="utf-8")
    calls = [r["llm_calls"] for _, r in results]
    print(f"Saved {len(responses)} -> {args.output}; log: {log_path}")
    print(f"  LLM calls/question: max {max(calls)}, mean {sum(calls) / len(calls):.2f}; "
          f"fallbacks {sum(r.get('fallback', False) for _, r in results)}")
    if "answers" in df.columns:
        gold = df["answers"].astype(str).str.strip().tolist()
        ok = [f"({l})" == g for (l, _), g in zip(results, gold)]
        print(f"  acc all: {100 * sum(ok) / len(ok):.1f}% ({len(ok)} q)")
        if "source" in df.columns:
            for s, grp in pd.Series(ok).groupby(df["source"].values):
                print(f"  acc {s:16s}: {100 * grp.mean():.1f}% ({len(grp)} q)")


if __name__ == "__main__":
    main()
