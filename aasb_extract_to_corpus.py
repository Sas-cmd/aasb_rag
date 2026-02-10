"""
Goal :
- 1 JSON object per page
- Skip page 1 and 2
- Preserve doc_id and page number
- Minimal cleaning

Next step:
- Chunk within pages
"""


# Imports

import json
from pathlib import Path
from typing import List, Dict

import fitz
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



# Paths and configuration

RAW_PDF_DIR = Path("data/raw_pdfs")          # Directory containing source PDFs
CORPUS_DIR = Path("data/corpus")            # Output directory
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = CORPUS_DIR / "corpus.jsonl"   # Output corpus (JSON Lines format)

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # Used only for its tokenizer



# Text cleaning and paragraph splitting

def clean_text(text: str) -> str:
    # Minimal, non-destructive cleaning to preserve original wording
    return (
        text.replace("\u00a0", " ")
            .replace("\t", " ")
            .strip()
    )


def split_paragraphs(text: str) -> List[str]:
    # Split text into paragraphs using blank lines
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


#
# Token-based chunking logic
#
def chunk_paragraphs_by_tokens(
    paragraphs: List[str],
    tokenizer,
    target_tokens: int = 300,
    overlap_tokens: int = 60,
) -> List[str]:
    chunks = []
    current = []
    current_tokens = 0

    for p in paragraphs:
        p_tokens = len(tokenizer.encode(p, add_special_tokens=False))

        if p_tokens == 0:
            continue

        # If a paragraph is too large, split it into sentences
        if p_tokens > target_tokens:
            sentences = [s.strip() for s in p.replace("\n", " ").split(". ") if s.strip()]
            for s in sentences:
                s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
                if s_tokens == 0:
                    continue

                if current_tokens + s_tokens > target_tokens and current:
                    chunks.append(" ".join(current).strip())

                    # Carry overlap from the previous chunk
                    carry = []
                    carry_tokens = 0
                    for item in reversed(current):
                        item_tokens = len(tokenizer.encode(item, add_special_tokens=False))
                        if carry_tokens + item_tokens > overlap_tokens:
                            break
                        carry.insert(0, item)
                        carry_tokens += item_tokens

                    current = carry[:]
                    current_tokens = carry_tokens

                current.append(s)
                current_tokens += s_tokens
            continue

        # Flush chunk if adding paragraph would exceed target size
        if current_tokens + p_tokens > target_tokens and current:
            chunks.append("\n\n".join(current).strip())

            carry = []
            carry_tokens = 0
            for item in reversed(current):
                item_tokens = len(tokenizer.encode(item, add_special_tokens=False))
                if carry_tokens + item_tokens > overlap_tokens:
                    break
                carry.insert(0, item)
                carry_tokens += item_tokens

            current = carry[:]
            current_tokens = carry_tokens

        current.append(p)
        current_tokens += p_tokens

    if current:
        chunks.append("\n\n".join(current).strip())

    return [c for c in chunks if c]



# PDF extraction and JSONL writing

def extract_pdf(pdf_path: Path, out_file, tokenizer, target_tokens: int, overlap_tokens: int):
    doc_id = pdf_path.stem  # Stable document identifier

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page_number = page_index + 1

            # Skip cover / front-matter pages
            if page_number <= 2:
                continue

            page = doc[page_index]
            text = clean_text(page.get_text("text"))

            if not text:
                continue

            paragraphs = split_paragraphs(text)
            if not paragraphs:
                continue

            chunks = chunk_paragraphs_by_tokens(
                paragraphs=paragraphs,
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )

            # Reset chunk numbering per page
            chunk_id = 0
            for chunk in chunks:
                chunk_id += 1

                record = {
                    "doc_id": doc_id,
                    "page": page_number,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")



# Corpus builder (main pipeline)

def build_corpus(target_tokens: int = 320, overlap_tokens: int = 60):
    pdf_files = sorted(RAW_PDF_DIR.glob("AASB_*.pdf"))

    if not pdf_files:
        raise RuntimeError("No PDFs found in data/raw_pdfs")

    # Load model only to access a compatible tokenizer
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    tokenizer = model.tokenizer

    with CORPUS_PATH.open("w", encoding="utf-8") as out_file:
        for pdf_path in tqdm(pdf_files, desc="Extracting + chunking PDFs"):
            extract_pdf(
                pdf_path=pdf_path,
                out_file=out_file,
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )

    print(f"Corpus written to {CORPUS_PATH}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Chunking: target_tokens={target_tokens}, overlap_tokens={overlap_tokens}")



# Script entry point

if __name__ == "__main__":
    build_corpus()



# SECTION-BY-SECTION EXPLANATION

#
# 1. Imports
#    Brings in standard utilities, PyMuPDF for PDF parsing, tqdm for progress
#    tracking, and SentenceTransformer solely to reuse its tokenizer.
#
# 2. Paths and configuration
#    Defines where raw PDFs live, where the processed corpus is written,
#    and enforces a consistent naming/output structure.
#
# 3. Text cleaning and paragraph splitting
#    Applies minimal whitespace normalization and splits extracted PDF text
#    into paragraph-like units using blank lines.
#
# 4. Token-based chunking logic
#    Groups paragraphs into chunks of approximately `target_tokens` using
#    the embedding model’s tokenizer. Overlap is preserved between chunks
#    to improve downstream retrieval continuity. Large paragraphs are
#    split into sentences as a fallback.
#
# 5. PDF extraction and JSONL writing
#    Iterates through each PDF page (skipping pages 1–2), extracts text,
#    chunks content within each page, and writes JSON records that preserve
#    document ID, page number, and chunk ID.
#
# 6. Corpus builder
#    Orchestrates the full pipeline: locating PDFs, initializing the tokenizer,
#    processing all documents, and writing a single JSONL corpus file.
#
# 7. Entry point
#    Allows the script to be run directly, producing a fully chunked corpus
#    ready for embedding, indexing, or retrieval workflows.
#
# ============================================================
