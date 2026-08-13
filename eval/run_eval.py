"""Reproducible evaluation harness for the RAG chatbot.

Runs a hand-labeled golden set (see golden_set.template.json /
GOLDEN_SET_GUIDE.md) through the ACTUAL production pipeline in
src/rag_engine.py -- same hybrid retrieval, same reranker, same score gate,
same generation prompt the Streamlit app uses -- and computes:

  Retrieval (computed twice -- see note below):
    Recall@K, Precision@K, MRR, Hit Rate

  Refusal quality (two variants):
    refusal_quality           : hard-refusal only (the pre-generation gate's
                                 canned REFUSAL_MESSAGE fired)
    effective_refusal_quality : hard refusal OR the model honestly admitted
                                 in its own words that the answer isn't in
                                 the document (checked via LLM judge, only
                                 with --llm-judge). The hard gate alone can
                                 undercount safe behavior: weak topical
                                 overlap can let a chunk through threshold,
                                 and the model can still correctly decline
                                 rather than guess -- that's equally safe,
                                 just a different code path.

  Latency:
    mean / median / p95 for condense, retrieval, rerank, generation, total

Retrieval metrics are computed at two points so you can see what the
reranker is actually buying you:
  - "raw"    : the hybrid vector+BM25 candidate pool, unranked-by-relevance
               cutoff at TOP_K_RETRIEVE
  - "served" : what's actually left after rerank + RERANK_SCORE_THRESHOLD +
               TOP_K_RERANK -- i.e. what generation actually sees

Usage:
    python eval/run_eval.py \\
        --golden-set eval/golden_set.json \\
        --pdf-dir /path/to/pdfs \\
        --output eval/results.json

Requires GROQ_API_KEY to be set (same as the app) for generation + the
question-condensing step.

--llm-judge / faithfulness+answer-relevance:
Computed in-process, in THIS environment, using the same pinned Groq LLM
src/rag_engine.py already uses -- no new dependencies. This is the primary
generation-quality metric for this project (see eval/llm_judge.py for the
full rationale).

--ragas (optional, secondary path, not required):
This flag additionally writes eval/ragas_input.json and prints the command
to run eval/run_ragas.py from a SEPARATE environment, if you still want to
try RAGAS's context_precision/context_recall on top of --llm-judge later.
Not needed to get a real faithfulness/relevance number -- --llm-judge above
already gives you that with zero extra setup.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from eval.llm_judge import judge_answer_relevance, judge_faithfulness, judge_is_soft_refusal  # noqa: E402
from eval.metrics import (  # noqa: E402
    aggregate,
    hit,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_quality,
)
from src import config  # noqa: E402
from src.document_loader import load_and_split_pdfs  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.rag_engine import (  # noqa: E402
    answer_question,
    get_embeddings,
    get_reranker,
    hybrid_retrieve,
    rerank,
)
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402


def build_indexes_uncached(pdf_dir: str):
    """Same logic as rag_engine.build_indexes but without the Streamlit
    cache decorator, since this script runs outside a Streamlit session."""
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    chunks = load_and_split_pdfs(pdf_paths)
    if not chunks:
        raise SystemExit("No extractable text found in any PDF in --pdf-dir")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.TOP_K_RETRIEVE

    return vectorstore, bm25_retriever, chunks


def run(
    golden_set_path: str,
    pdf_dir: str,
    output_path: str,
    use_ragas: bool,
    ragas_input_path: str,
    use_llm_judge: bool,
):
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"Loaded {len(golden_set)} golden-set questions from {golden_set_path}")
    print("Building indexes from", pdf_dir, "...")
    vectorstore, bm25_retriever, chunks = build_indexes_uncached(pdf_dir)
    print(f"Indexed {len(chunks)} chunks.\n")

    llm = get_llm()
    reranker = get_reranker()
    judge_llm = (
        get_llm(model_name=config.JUDGE_MODEL, reasoning_effort=config.JUDGE_REASONING_EFFORT)
        if use_llm_judge
        else None
    )

    per_question_rows = []
    ragas_questions, ragas_answers, ragas_contexts, ragas_ground_truths = [], [], [], []

    for item in golden_set:
        qid = item["id"]
        question = item["question"]
        relevant_ids = item.get("relevant_chunks", [])
        is_answerable = item.get("is_answerable", bool(relevant_ids))
        reference_answer = item.get("reference_answer", "")

        # ---- raw hybrid retrieval (pre-rerank) ----
        t0 = time.perf_counter()
        raw_candidates = hybrid_retrieve(question, vectorstore, bm25_retriever)
        raw_ms = (time.perf_counter() - t0) * 1000
        raw_ids = [d.metadata.get("chunk_id") for d in raw_candidates]

        # ---- reranked pool (pre-threshold, for MRR-of-reranker-order) ----
        t1 = time.perf_counter()
        reranked = rerank(question, raw_candidates, reranker)
        rerank_ms = (time.perf_counter() - t1) * 1000
        reranked_ids_all = [d.metadata.get("chunk_id") for d, _ in reranked]

        # ---- served set (post-threshold, post-top-k -- what generation sees) ----
        served = [
            (d, s) for d, s in reranked if s >= config.RERANK_SCORE_THRESHOLD
        ][: config.TOP_K_RERANK]
        if not served and reranked and reranked[0][1] >= config.RERANK_FLOOR_SCORE:
            # Mirrors the floor-fallback in rag_engine.answer_question --
            # keep this in sync with that function, or these retrieval
            # metrics will silently stop matching what the app actually
            # serves to generation.
            served = reranked[:1]
        served_ids = [d.metadata.get("chunk_id") for d, _ in served]

        # ---- full pipeline call (this is what the app actually does) ----
        result = answer_question(
            llm=llm,
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            question=question,
            chat_history=[],  # golden-set questions are evaluated standalone
        )

        refused = not result["grounded"]

        row = {
            "id": qid,
            "question": question,
            "is_answerable": is_answerable,
            "relevant_chunks": relevant_ids,
            "raw_retrieved_ids": raw_ids,
            "served_ids": served_ids,
            "reranked_ids_all": reranked_ids_all,
            "answer": result["answer"],
            "refused": refused,
            "timings": result["timings"],
            "raw_retrieval_ms": raw_ms,
            "rerank_ms_standalone": rerank_ms,
            # retrieval metrics (skip when there's nothing to score against,
            # e.g. genuinely unanswerable questions with no relevant chunks)
            "recall_raw": recall_at_k(raw_ids, relevant_ids) if relevant_ids else None,
            "precision_raw": precision_at_k(raw_ids, relevant_ids) if relevant_ids else None,
            "mrr_raw": reciprocal_rank(raw_ids, relevant_ids) if relevant_ids else None,
            "hit_raw": hit(raw_ids, relevant_ids) if relevant_ids else None,
            "recall_served": recall_at_k(served_ids, relevant_ids) if relevant_ids else None,
            "precision_served": precision_at_k(served_ids, relevant_ids) if relevant_ids else None,
            "mrr_served": reciprocal_rank(reranked_ids_all, relevant_ids) if relevant_ids else None,
            "hit_served": hit(served_ids, relevant_ids) if relevant_ids else None,
            "faithfulness_score": None,
            "faithfulness_unsupported_claims": None,
            "answer_relevance_score": None,
            "soft_refusal": None,
            "effective_refused": refused,
        }

        # LLM-judge generation metrics: only meaningful for turns that
        # actually generated grounded content, not refusals (a refusal has
        # no claims to check faithfulness on, and "relevance" of a correct
        # refusal is already captured by refusal_quality above).
        if use_llm_judge and not refused:
            context_text = "\n\n".join(d.page_content for d in result["sources"])
            faith = judge_faithfulness(judge_llm, question, context_text, result["answer"])
            relevance = judge_answer_relevance(judge_llm, question, result["answer"])
            row["faithfulness_score"] = faith["score"]
            row["faithfulness_unsupported_claims"] = faith.get("unsupported_claims")
            row["answer_relevance_score"] = relevance["score"]

        # Soft-refusal check: only for the specific edge case where a
        # question is genuinely unanswerable but the hard pre-generation
        # gate didn't fire (some weak topical overlap let a chunk through).
        # `refused` alone would undercount this turn even if the model
        # honestly admitted the info isn't in the document instead of
        # guessing -- see judge_is_soft_refusal's docstring.
        if use_llm_judge and is_answerable is False and not refused:
            soft_refusal = judge_is_soft_refusal(judge_llm, question, result["answer"])
            row["soft_refusal"] = soft_refusal
            if soft_refusal:
                row["effective_refused"] = True

        per_question_rows.append(row)

        if use_ragas and is_answerable:
            ragas_questions.append(question)
            ragas_answers.append(result["answer"])
            ragas_contexts.append([d.page_content for d in result["sources"]])
            ragas_ground_truths.append(reference_answer)

        print(f"[{qid}] refused={refused} "
              f"recall(raw/served)={row['recall_raw']}/{row['recall_served']} "
              f"faithfulness={row['faithfulness_score']} "
              f"total_ms={row['timings'].get('total_ms', 0):.0f}")

    # ---- aggregate retrieval metrics ----
    summary = {
        "n_questions": len(golden_set),
        "retrieval_raw": {
            "recall_at_k": aggregate([r["recall_raw"] for r in per_question_rows]),
            "precision_at_k": aggregate([r["precision_raw"] for r in per_question_rows]),
            "mrr": aggregate([r["mrr_raw"] for r in per_question_rows]),
            "hit_rate": aggregate([r["hit_raw"] for r in per_question_rows]),
        },
        "retrieval_served": {
            "recall_at_k": aggregate([r["recall_served"] for r in per_question_rows]),
            "precision_at_k": aggregate([r["precision_served"] for r in per_question_rows]),
            "mrr": aggregate([r["mrr_served"] for r in per_question_rows]),
            "hit_rate": aggregate([r["hit_served"] for r in per_question_rows]),
        },
        "refusal_quality": refusal_quality(
            [{"is_answerable": r["is_answerable"], "refused": r["refused"]} for r in per_question_rows]
        ),
        "effective_refusal_quality": refusal_quality(
            [{"is_answerable": r["is_answerable"], "refused": r["effective_refused"]} for r in per_question_rows]
        ) if use_llm_judge else None,
        "latency_ms": {
            "condense": aggregate([r["timings"].get("condense_ms") for r in per_question_rows]),
            "retrieval": aggregate([r["timings"].get("retrieval_ms") for r in per_question_rows]),
            "rerank": aggregate([r["timings"].get("rerank_ms") for r in per_question_rows]),
            "generation": aggregate([r["timings"].get("generation_ms") for r in per_question_rows]),
            "total": aggregate([r["timings"].get("total_ms") for r in per_question_rows]),
        },
        "ragas": None,  # never computed in this process -- see eval/run_ragas.py (optional, secondary)
        "generation_quality": {
            "faithfulness": aggregate([r["faithfulness_score"] for r in per_question_rows]),
            "answer_relevance": aggregate([r["answer_relevance_score"] for r in per_question_rows]),
        } if use_llm_judge else None,
    }

    if use_ragas:
        if ragas_questions:
            with open(ragas_input_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "questions": ragas_questions,
                        "answers": ragas_answers,
                        "contexts": ragas_contexts,
                        "ground_truths": ragas_ground_truths,
                    },
                    f,
                    indent=2,
                )
            print(
                f"\nWrote {len(ragas_questions)} answerable question/answer/context "
                f"tuples to {ragas_input_path}.\n"
                "Faithfulness / answer relevance / context precision / context "
                "recall are NOT computed in this process -- ragas needs a "
                "langchain-core version that's incompatible with the pinned "
                "0.2.x stack src/rag_engine.py runs on. Compute them from a "
                "SEPARATE environment (only ragas + datasets + langchain-groq "
                "installed there, nothing from src/) with:\n\n"
                f"    python eval/run_ragas.py --input {ragas_input_path} "
                "--output eval/ragas_results.json\n"
            )
        else:
            print("\n--ragas was set, but there were no answerable questions to score.")

    output = {"summary": summary, "per_question": per_question_rows}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFull results written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Compute faithfulness and answer-relevance via an LLM judge, "
             "in-process, using the same Groq LLM already in this "
             "environment. No new dependencies. Recommended: this is the "
             "primary generation-quality metric for this project.",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Optional, secondary: write eval/ragas_input.json for a "
             "separate RAGAS run via eval/run_ragas.py in another "
             "environment. Not required -- --llm-judge above already gives "
             "you faithfulness/relevance with zero extra setup.",
    )
    parser.add_argument("--ragas-input", default="eval/ragas_input.json")
    args = parser.parse_args()
    run(args.golden_set, args.pdf_dir, args.output, args.ragas, args.ragas_input, args.llm_judge)


if __name__ == "__main__":
    main()
