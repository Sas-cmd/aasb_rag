import json
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer


CORPUS_PATH = Path("data/corpus/corpus.jsonl")
INDEX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/meta.jsonl")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


_model = None
_index = None
_meta = None
_texts = None


def _load():
    global _model, _index, _meta, _texts

    if _model is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    _model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
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


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    _load()

    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 5

    q_emb = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if q_emb.shape[1] != _index.d:
        raise RuntimeError(
            f" {q_emb.shape[1]} does not match index dim {_index.d}. "
            "rebuild FAISS index."
        )

    scores, idxs = _index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue

        m = _meta[idx]
        results.append({
            "doc_id": m.get("doc_id"),
            "page": m.get("page"),
            "text": _texts[idx],
            "score": float(score),
        })

    return results