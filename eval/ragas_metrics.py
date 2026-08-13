"""Thin wrapper around RAGAS for the generation-quality metrics that need
LLM-as-judge: faithfulness, answer relevance, context precision, context
recall.

This module is optional. If `ragas` (and `datasets`) aren't installed,
`compute_ragas_metrics` returns None and run_eval.py reports those metrics
as "not computed" rather than inventing numbers.

Install with:  pip install ragas datasets
"""
from __future__ import annotations

from typing import Dict, List, Optional


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


def compute_ragas_metrics(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    llm=None,
    embeddings=None,
) -> Optional[Dict[str, float]]:
    """Runs RAGAS's faithfulness, answer_relevancy, context_precision, and
    context_recall over the given (question, answer, contexts, ground_truth)
    tuples. Returns a dict of metric_name -> score, or None if ragas isn't
    installed.

    `llm` / `embeddings` should be LangChain-wrapped models if you want RAGAS
    to use the same Groq LLM as the app (recommended for consistency) rather
    than defaulting to OpenAI, which requires OPENAI_API_KEY and would be a
    silent, uncontrolled swap in judge model.
    """
    if not ragas_available():
        return None

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    kwargs = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        **kwargs,
    )
    return dict(result)
