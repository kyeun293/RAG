"""Build an MMLU-Pro validation subset for the assignment domains.

Source : TIGER-Lab/MMLU-Pro (HuggingFace), test split parquet
         https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Output : CSV in the same format as data/test_samples.csv
         (prompts = "QUESTIONn) ...\n(A) ...", answers = "(X)") plus
         question_id (MMLU-Pro id) and category columns for per-domain analysis.

NOTE: this file is for *evaluation only*. Never index it as a retrieval source
(assignment: benchmark question dumps / answer keys are prohibited knowledge).

Usage (from base_taein/):
    python make_mmlu_pro_subset.py --per-domain 400 --seed 42
"""
import argparse
import os
import string
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
URL = ("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
       "data/test-00000-of-00001.parquet")
RAW = os.path.join(HERE, "data", "mmlu_pro_raw", "test.parquet")
DOMAINS = ["law", "psychology", "business", "philosophy", "history"]
LETTERS = string.ascii_uppercase


def _norm(text):
    return " ".join(str(text).split()).lower()


def load_raw():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    return pd.read_parquet(RAW)


def format_prompt(n, question, options):
    lines = [f"QUESTION{n}) {question.strip()}"]
    lines += [f"({LETTERS[i]}) {str(o).strip()}" for i, o in enumerate(options)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default=os.path.join(HERE, "data", "test_samples.csv"),
                    help="drop questions that also appear in this CSV")
    ap.add_argument("--output", default=os.path.join(HERE, "data", "mmlu_pro_subset.csv"))
    args = ap.parse_args()

    df = load_raw()
    df = df[df["category"].isin(DOMAINS)].copy()

    # Remove overlap with the practice set (match on normalized question stem).
    if args.exclude and os.path.exists(args.exclude):
        practice = " || ".join(_norm(p) for p in pd.read_csv(args.exclude)["prompts"])
        overlap = df["question"].map(lambda q: _norm(q)[:120] in practice)
        print(f"excluded {int(overlap.sum())} questions overlapping {args.exclude}")
        df = df[~overlap]

    parts = []
    for dom in DOMAINS:
        sub = df[df["category"] == dom]
        n = min(args.per_domain, len(sub))
        parts.append(sub.sample(n=n, random_state=args.seed))
        print(f"{dom:12s}: {n} / {len(sub)}")
    out = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out = pd.DataFrame({
        "question_id": [f"mmlupro-{i}" for i in out["question_id"]],
        "category": out["category"],
        "prompts": [format_prompt(n, q, o) for n, (q, o) in
                    enumerate(zip(out["question"], out["options"]), 1)],
        "answers": [f"({a})" for a in out["answer"]],
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"saved {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
