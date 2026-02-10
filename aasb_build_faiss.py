import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


CORPUS_PATH = Path("data/corpus/corpus.jsonl")
INDEX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/meta.jsonl")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

FAISS_CANDIDATES = 20


_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_index = None
_meta: Optional[list] = None
_texts: Optional[list] = None


def _load():
    global _model, _reranker, _index, _meta, _texts

    if _model is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    _model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    _reranker = CrossEncoder(RERANKER_MODEL, device="cuda")

    _index = faiss.read_index(str(INDEX_PATH))

    _meta = []
    with META_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _meta.append(json.loads(line))

    _texts = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _texts.append(json.loads(line)["text"])

    if len(_meta) != len(_texts):
        raise RuntimeError(
            f"Metadata/text length mismatch: meta={len(_meta)} texts={len(_texts)}"
        )


def _faiss_retrieve(query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_emb = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if q_emb.shape[1] != _index.d:
        raise RuntimeError(
            f"Embedding dim {q_emb.shape[1]} does not match index dim {_index.d}. "
            "Rebuild FAISS index (run aasb_build_faiss.py)."
        )

    scores, idxs = _index.search(q_emb, top_k)
    return scores[0], idxs[0]


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    _load()

    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 5

    faiss_k = max(top_k, FAISS_CANDIDATES)
    faiss_scores, faiss_idxs = _faiss_retrieve(query, faiss_k)

    candidates = []
    for score, idx in zip(faiss_scores, faiss_idxs):
        if idx < 0:
            continue
        candidates.append((int(idx), float(score), _texts[idx]))

    if not candidates:
        return []

    pairs = [(query, c[2]) for c in candidates]
    rerank_scores = _reranker.predict(pairs)

    reranked = []
    for (idx, faiss_score, _), rr_score in zip(candidates, rerank_scores):
        reranked.append((idx, float(faiss_score), float(rr_score)))

    reranked.sort(key=lambda x: x[2], reverse=True)

    results = []
    for idx, faiss_score, rr_score in reranked[:top_k]:
        m = _meta[idx]
        results.append({
            "doc_id": m.get("doc_id"),
            "page": m.get("page"),
            "text": _texts[idx],
            "score": rr_score,
            "faiss_score": faiss_score,
        })

    return results