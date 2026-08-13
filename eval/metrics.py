"""Retrieval, refusal, and latency metrics.

These are computed directly from (retrieved_ids, relevant_ids) pairs, so
they need no LLM judge and no external library -- they're exact, not
estimated. Faithfulness / answer relevance / context relevance (which do
need judgment) live in ragas_metrics.py instead.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """|retrieved ∩ relevant| / |relevant|. Undefined (returns None) if there
    are no relevant ids for this question -- that's a golden-set bug, not a
    0."""
    if not relevant_ids:
        return None
    relevant_set = set(relevant_ids)
    hit = len(set(retrieved_ids) & relevant_set)
    return hit / len(relevant_set)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """|retrieved ∩ relevant| / |retrieved|. Undefined if nothing was
    retrieved (e.g. a correct refusal) -- callers should skip those rows
    rather than counting them as 0."""
    if not retrieved_ids:
        return None
    relevant_set = set(relevant_ids)
    hit = len(set(retrieved_ids) & relevant_set)
    return hit / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """1 / rank of the first relevant id in retrieved_ids (1-indexed), or 0
    if none of the retrieved ids are relevant."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def hit(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """1.0 if at least one retrieved id is relevant, else 0.0."""
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def aggregate(values: List[float]) -> Dict[str, float]:
    """Mean / median / p95 / n over a list of floats, skipping Nones."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "p95": None, "n": 0}
    ordered = sorted(clean)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "p95": ordered[p95_idx],
        "n": len(clean),
    }


def refusal_quality(rows: List[dict]) -> Dict[str, float]:
    """rows: list of {"is_answerable": bool, "refused": bool}.

    Returns:
      refusal_recall     : of the truly-unanswerable questions, fraction
                            correctly refused (higher is better)
      false_refusal_rate : of the truly-answerable questions, fraction
                            wrongly refused (lower is better)
    """
    unanswerable = [r for r in rows if not r["is_answerable"]]
    answerable = [r for r in rows if r["is_answerable"]]

    refusal_recall = (
        sum(1 for r in unanswerable if r["refused"]) / len(unanswerable)
        if unanswerable
        else None
    )
    false_refusal_rate = (
        sum(1 for r in answerable if r["refused"]) / len(answerable)
        if answerable
        else None
    )
    return {
        "refusal_recall": refusal_recall,
        "false_refusal_rate": false_refusal_rate,
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
    }
