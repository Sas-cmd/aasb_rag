"""

- 1 JSON object per page
- Skip page 1 and 2 
- Need to preserve doc_id and page number
- Minimal cleaning 


Next steps
- chunk within pages



"""

import json
from pathlib import Path
from typing import List, Dict

import fitz
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


RAW_PDF_DIR = Path("data/raw_pdfs")
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = CORPUS_DIR / "corpus.jsonl"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def clean_text(text: str) -> str:
    return (
        text.replace("\u00a0", " ")
            .replace("\t", " ")
            .strip()
    )


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


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

        if p_tokens > target_tokens:
            sentences = [s.strip() for s in p.replace("\n", " ").split(". ") if s.strip()]
            for s in sentences:
                s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
                if s_tokens == 0:
                    continue

                if current_tokens + s_tokens > target_tokens and current:
                    chunks.append(" ".join(current).strip())
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

    chunks = [c for c in chunks if c and len(c.strip()) > 0]
    return chunks


def extract_pdf(pdf_path: Path, out_file, tokenizer, target_tokens: int, overlap_tokens: int):
    doc_id = pdf_path.stem

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page_number = page_index + 1

            if page_number <= 2:
                continue

            page = doc[page_index]
            text = page.get_text("text")
            text = clean_text(text)

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


def build_corpus(target_tokens: int = 320, overlap_tokens: int = 60):
    pdf_files = sorted(RAW_PDF_DIR.glob("AASB_*.pdf"))

    if not pdf_files:
        raise RuntimeError("No PDFs found in data/raw_pdfs")

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


if __name__ == "__main__":
    build_corpus()