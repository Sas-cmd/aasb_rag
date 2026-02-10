import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

CORPUS_PATH = Path("data/corpus/corpus.jsonl")

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_corpus():
    corpus = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def score_page(query_tokens, page_tokens):
    page_counts = Counter(page_tokens)
    score = 0.0
    for t in query_tokens:
        if t in page_counts:
            score += math.log(1 + page_counts[t])
    return score


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    corpus = load_corpus()
    query_tokens = tokenize(query)

    scored = []

    for row in corpus:
        page_tokens = tokenize(row["text"])
        s = score_page(query_tokens, page_tokens)
        if s > 0:
            scored.append({
                "doc_id": row["doc_id"],
                "page": row["page"],
                "score": round(s, 3),
                "text": row["text"]
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    q = "definition of a lease"
    results = retrieve(q, top_k=5)

    print(f"\nQuery: {q}\n")
    for r in results:
        print(f"{r['doc_id']} – page {r['page']} (score={r['score']})")
        print(r["text"][:400], "\n")