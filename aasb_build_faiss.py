# SECTION 1 — Imports (dependencies and typing)
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


# SECTION 2 — File Paths & Model Configuration
# These paths define where the corpus, FAISS index, and associated metadata live.
CORPUS_PATH = Path("data/corpus/corpus.jsonl")
INDEX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/meta.jsonl")

# Embedding model produces vector representations for semantic search.
# Reranker model re-scores candidate passages for better relevance ordering.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Number of initial ANN candidates fetched from FAISS before reranking.
FAISS_CANDIDATES = 20


# SECTION 3 — Module-Level Cache (lazy-loaded singletons)
# These globals are loaded once on first use to avoid repeated model/index IO and GPU init.
_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_index = None
_meta: Optional[list] = None
_texts: Optional[list] = None


# SECTION 4 — Loader: initialize models, index, and in-memory corpus/metadata
def _load():
    global _model, _reranker, _index, _meta, _texts

    # Guard: if we've already initialized, do nothing (keeps retrieval calls fast).
    if _model is not None:
        return

    # Defensive checks so failures happen early and with clear messages.
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    # Load embedding + reranking models onto GPU (assumes CUDA is available).
    _model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    _reranker = CrossEncoder(RERANKER_MODEL, device="cuda")

    # Load FAISS ANN index used for initial candidate retrieval.
    _index = faiss.read_index(str(INDEX_PATH))

    # Load per-chunk metadata (doc/page/etc.) aligned by row index with the FAISS index.
    _meta = []
    with META_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _meta.append(json.loads(line))

    # Load the raw chunk text aligned by row index with the FAISS index.
    _texts = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _texts.append(json.loads(line)["text"])

    # Sanity check: meta and text must be 1:1 and match index row semantics.
    if len(_meta) != len(_texts):
        raise RuntimeError(
            f"Metadata/text length mismatch: meta={len(_meta)} texts={len(_texts)}"
        )


# SECTION 5 — FAISS Retrieval: embed query then perform ANN search
def _faiss_retrieve(query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Convert the query into a normalized vector for cosine-similarity-style search.
    q_emb = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Safety: embedding dimensionality must match the index dimensionality.
    if q_emb.shape[1] != _index.d:
        raise RuntimeError(
            f"Embedding dim {q_emb.shape[1]} does not match index dim {_index.d}. "
            "Rebuild FAISS index (run aasb_build_faiss.py)."
        )

    # FAISS returns (scores, indices) for each query; here we only have one query.
    scores, idxs = _index.search(q_emb, top_k)
    return scores[0], idxs[0]


# SECTION 6 — Public API: retrieve top passages with reranking and metadata
def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    # Ensure everything is loaded (models, index, corpus, metadata).
    _load()

    # Input validation: empty or non-string queries return no results.
    if not isinstance(query, str) or not query.strip():
        return []

    # Input validation: enforce a sensible positive integer for top_k.
    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 5

    # Pull more candidates than we ultimately return, to give reranker enough options.
    faiss_k = max(top_k, FAISS_CANDIDATES)
    faiss_scores, faiss_idxs = _faiss_retrieve(query, faiss_k)

    # Build candidate list: (row_index, faiss_score, candidate_text).
    candidates = []
    for score, idx in zip(faiss_scores, faiss_idxs):
        if idx < 0:
            continue
        candidates.append((int(idx), float(score), _texts[idx]))

    if not candidates:
        return []

    # Reranker input format: list of (query, passage_text) pairs.
    pairs = [(query, c[2]) for c in candidates]
    rerank_scores = _reranker.predict(pairs)

    # Combine FAISS score and reranker score for transparency/debugging.
    reranked = []
    for (idx, faiss_score, _), rr_score in zip(candidates, rerank_scores):
        reranked.append((idx, float(faiss_score), float(rr_score)))

    # Final ordering is based on reranker score (higher = more relevant).
    reranked.sort(key=lambda x: x[2], reverse=True)

    # Build user-facing results with doc/page metadata + both scores.
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


# SECTION SUMMARY — What this code does (by section)
# 1) Imports:
#    Brings in JSON + filesystem helpers, FAISS for ANN search, NumPy for arrays,
#    and SentenceTransformers for both embedding (bi-encoder) and reranking (cross-encoder).
#
# 2) Paths & Configuration:
#    Defines the on-disk resources (corpus, index, metadata) and the specific BGE models used.
#    Sets how many candidates to fetch from FAISS before reranking.
#
# 3) Caches:
#    Uses module-level singletons so models/index/text are loaded once per process.
#
# 4) _load():
#    Validates files exist, loads embedding and reranker models (GPU), loads FAISS index,
#    reads metadata + corpus text into memory, and verifies alignment.
#
# 5) _faiss_retrieve():
#    Encodes the query into a normalized embedding and runs FAISS search to get candidate hits.
#    Includes a guard to catch mismatched embedding/index dimensions.
#
# 6) retrieve():
#    Public function that:
#      - validates inputs
#      - fetches FAISS candidates (fast approximate search)
#      - reranks candidates with a cross-encoder (more accurate relevance)
#      - returns top_k results including doc_id/page, passage text, and both scores.
