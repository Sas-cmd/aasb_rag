# ============================================================
# AASB RAG CLI
# ============================================================

# -----------------------------
# SECTION 1 — Imports & Constants
# -----------------------------
import argparse
import textwrap
from typing import List, Dict, Tuple
import ollama
from aasb_faiss_retrieve import retrieve

# Local LLM model identifier (must exist in Ollama)
MODEL_NAME = "llama3:instruct"

# Thresholds used to decide whether retrieval is confident enough to answer directly
MIN_TOP_SCORE = 0.35
MIN_MARGIN = 0.03


# -----------------------------
# SECTION 2 — Context Construction
# -----------------------------
def build_context(pages: List[Dict]) -> str:
    # Combine retrieved chunks into a single authoritative context with citation headers
    blocks = []
    for p in pages:
        blocks.append(f"[{p['doc_id']} – page {p['page']}]\n{p['text']}")
    return "\n\n".join(blocks)


# -----------------------------
# SECTION 3 — Definition Detection Heuristics
# -----------------------------
def contains_definition(text: str) -> bool:
    # Heuristic check for definition-style language common in standards
    t = text.lower()
    triggers = [
        " is a contract that ",
        " means a contract ",
        " is defined as ",
        " conveys the right to control the use ",
        " a lease is ",
        " lease is a contract ",
    ]
    return any(tr in t for tr in triggers)


def filter_definition_override(question: str, pages: List[Dict]) -> List[Dict]:
    # For definitional questions, prefer chunks that actually contain definitions
    q = question.lower().strip()
    if q.startswith("what is") or "define" in q or "what does" in q:
        hits = [p for p in pages if contains_definition(p.get("text", ""))]
        if hits:
            return hits
    return pages


# -----------------------------
# SECTION 4 — Confidence & Ambiguity Scoring
# -----------------------------
def confidence_stats(pages: List[Dict]) -> Tuple[float, float]:
    # Compute top score and the gap between the top two results
    scores = [float(p.get("score", 0.0)) for p in pages if p.get("score") is not None]
    if not scores:
        return 0.0, 0.0
    scores_sorted = sorted(scores, reverse=True)
    top1 = scores_sorted[0]
    top2 = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
    return top1, (top1 - top2)


def needs_clarification_gate(pages: List[Dict], min_top_score: float, min_margin: float) -> bool:
    # Gate when evidence is weak or multiple chunks compete
    top1, margin = confidence_stats(pages)
    if top1 < min_top_score:
        return True
    if margin < min_margin:
        return True
    return False


# -----------------------------
# SECTION 5 — Prompt Construction
# -----------------------------
def build_prompt(question: str, context: str, gated: bool) -> str:
    # Build a strict prompt that forbids external knowledge and enforces citations
    if gated:
        return textwrap.dedent(f"""
        Output constraint (Mandatory):
        - Your response MUST start with exactly: "Answer:"
        - Do not write any other text before that.

        Hard rules:
        - Use only the authoritative text below.
        - Do not invent or rely on outside knowledge.
        - Citations MUST be copied EXACTLY from the bracketed headers in the authoritative text,
          e.g. "[AASB_016 – page 14]".
        - If you provide an Answer containing any factual claim, you MUST include at least 1 citation.

        Task:
        - Provide the best direct answer supported by the authoritative text.
        - If the authoritative text does not contain enough to answer, write exactly:
          "Answer: Insufficient evidence in retrieved text."
        - Then ask 1–3 targeted clarifying questions.

        Authoritative text:
        {context}
        """).strip()

    return textwrap.dedent(f"""
    Output constraints (Mandatory):
    - Your response MUST start with exactly one of these tokens:
      "Answer:" OR "Clarifying questions:"
    - Do not write any other text before that.

    Hard rules:
    - Use only the authoritative text below.
    - Do not invent or rely on outside knowledge.
    - Citations MUST be copied EXACTLY from the bracketed headers in the authoritative text,
      e.g. "[AASB_016 – page 14]".
    - If you provide an Answer containing any factual claim, you MUST include at least 1 citation.

    Decision rules:
    - If a definition exists in the authoritative text, answer directly and do NOT ask clarifying questions.
    - Ask clarifying questions only if genuinely ambiguous AND no definition exists in the text.

    Question:
    {question}

    Authoritative text:
    {context}
    """).strip()


# -----------------------------
# SECTION 6 — Core RAG Pipeline
# -----------------------------
def ask(
    question: str,
    top_k: int = 7,
    min_top_score: float = MIN_TOP_SCORE,
    min_margin: float = MIN_MARGIN,
) -> Tuple[str, List[Dict], Dict[str, float]]:
    pages = retrieve(question, top_k=top_k)

    if not pages:
        return "Answer:\nNo relevant authoritative text found.", [], {"top1": 0.0, "margin": 0.0}

    pages = filter_definition_override(question, pages)

    top1, margin = confidence_stats(pages)
    gated = needs_clarification_gate(pages, min_top_score=min_top_score, min_margin=min_margin)

    context = build_context(pages)
    prompt = build_prompt(question, context, gated=gated)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"], pages, {"top1": float(top1), "margin": float(margin)}


# -----------------------------
# SECTION 7 — Health Check
# -----------------------------
def run_health_check():
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )

    if response["message"]["content"].strip() != "OK":
        raise RuntimeError("LLM failed")

    print("model loaded, LLM responding.")


# -----------------------------
# SECTION 8 — Interactive CLI
# -----------------------------
def run_interactive(top_k: int, min_top_score: float, min_margin: float):
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("AASB RAG > ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not question or question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        answer, sources, conf = ask(
            question,
            top_k=top_k,
            min_top_score=min_top_score,
            min_margin=min_margin,
        )

        print()
        print(answer.strip())
        print(f"\nCONFIDENCE:\nTop score: {conf['top1']:.3f}\nMargin:    {conf['margin']:.3f}")

        print("\nSOURCES:\n")
        for s in sources:
            if "faiss_score" in s:
                print(
                    f"{s['doc_id']} – page {s['page']} "
                    f"(score={s['score']:.3f}, faiss={s['faiss_score']:.3f})"
                )
            else:
                print(
                    f"{s['doc_id']} – page {s['page']} "
                    f"(score={s['score']:.3f})"
                )

        print("\n" + "-" * 60 + "\n")


# -----------------------------
# SECTION 9 — Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="AASB RAG: FAISS semantic retrieval + LLaMA 3")

    parser.add_argument("--check", action="store_true", help="Run health check only")
    parser.add_argument("--top-k", type=int, default=7, help="Number of chunks to retrieve")
    parser.add_argument("--min-top-score", type=float, default=MIN_TOP_SCORE, help="Minimum top score before gating")
    parser.add_argument("--min-margin", type=float, default=MIN_MARGIN, help="Minimum top1-top2 margin before gating")

    args = parser.parse_args()

    if args.check:
        run_health_check()
    else:
        run_interactive(args.top_k, args.min_top_score, args.min_margin)


if __name__ == "__main__":
    main()


# ============================================================
# SECTION-BY-SECTION SUMMARY
# ============================================================
# Section 1 — Imports & Constants
# Defines dependencies, the LLM model, and retrieval confidence thresholds.
#
# Section 2 — Context Construction
# Builds a single authoritative context block with strict citation headers
# that the LLM must copy verbatim.
#
# Section 3 — Definition Detection Heuristics
# Detects definitional questions and prioritises chunks that contain
# explicit standard-style definitions.
#
# Section 4 — Confidence & Ambiguity Scoring
# Quantifies retrieval strength and ambiguity using top score and margin.
#
# Section 5 — Prompt Construction
# Enforces no hallucination, exact citation copying, and deterministic
# output formats, with stricter rules when confidence is low.
#
# Section 6 — Core RAG Pipeline
# Orchestrates retrieval, filtering, gating, prompt construction,
# and LLM invocation.
#
# Section 7 — Health Check
# Verifies that the LLM loads correctly and responds deterministically.
#
# Section 8 — Interactive CLI
# Provides a REPL interface with transparency into confidence and sources.
#
# Section 9 — Entry Point
# Handles CLI arguments and routes execution.
# ============================================================
