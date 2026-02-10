import argparse
import textwrap
from typing import List, Dict, Tuple
import ollama
from aasb_faiss_retrieve import retrieve


MODEL_NAME = "llama3:instruct"

MIN_TOP_SCORE = 0.35
MIN_MARGIN = 0.03


def build_context(pages: List[Dict]) -> str:
    blocks = []
    for p in pages:
        blocks.append(f"[{p['doc_id']} – page {p['page']}]\n{p['text']}")
    return "\n\n".join(blocks)


def contains_definition(text: str) -> bool:
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
    q = question.lower().strip()
    if q.startswith("what is") or "define" in q or "what does" in q:
        hits = [p for p in pages if contains_definition(p.get("text", ""))]
        if hits:
            return hits
    return pages


def confidence_stats(pages: List[Dict]) -> Tuple[float, float]:
    scores = [float(p.get("score", 0.0)) for p in pages if p.get("score") is not None]
    if not scores:
        return 0.0, 0.0
    scores_sorted = sorted(scores, reverse=True)
    top1 = scores_sorted[0]
    top2 = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
    return top1, (top1 - top2)


def needs_clarification_gate(pages: List[Dict], min_top_score: float, min_margin: float) -> bool:
    top1, margin = confidence_stats(pages)
    if top1 < min_top_score:
        return True
    if margin < min_margin:
        return True
    return False


def build_prompt(question: str, context: str, gated: bool) -> str:
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

        Output format (Mandatory):

        Answer:
        <1–6 sentences>

        Citations:
        - <copy one or more bracketed headers exactly, e.g. [AASB_016 – page 14]>

        Clarifying questions:
        1) <specific clarifier>
        2) <specific clarifier>


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

    Output format (choose ONE only):

    A) Direct answer exists:
    Answer:
    <concise answer, paraphrased from the text>

    Citations:
    - <copy one or more bracketed headers exactly, e.g. [AASB_016 – page 14]>

    B) Needs clarification:
    Clarifying questions:
    1) <specific clarifier>
    2) <specific clarifier>

    What can be stated from the provided text so far:
    <1–5 sentences, only if something relevant exists>

    Citations:
    - <copy one or more bracketed headers exactly, only if cited above>

    Question:
    {question}

    Authoritative text:
    {context}
    """).strip()


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


def run_health_check():
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )

    if response["message"]["content"].strip() != "OK":
        raise RuntimeError("LLM failed")

    print("model loaded, LLM responding.")


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