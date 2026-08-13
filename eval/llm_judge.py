"""Lightweight, dependency-free LLM-as-judge generation metrics.

Computes faithfulness (groundedness) and answer relevance using the SAME
Groq LLM and SAME pinned langchain stack src/rag_engine.py already uses --
no new packages, no separate environment. This is the primary
generation-quality metric for this project now: ragas's own dependencies
proved incompatible with the pinned stack in practice (two separate,
confirmed failures -- an unresolvable langchain-core version conflict, then
an unrelated upstream bug in ragas itself importing a relocated
ChatVertexAI class). eval/run_ragas.py is still here if you want to try
ragas separately later, but it is no longer required for a generation
metric. This module has zero new dependencies.

Two metrics, each scored 0.0-1.0 by asking the judge LLM to reason and
return strict JSON:

  faithfulness      : are the answer's claims actually supported by the
                       retrieved context? This is the generation-side
                       counterpart to the chunk-level Recall@K/Precision@K
                       already computed in run_eval.py -- note that those
                       existing metrics are exact-match against a
                       hand-labeled golden set, and are arguably a MORE
                       rigorous "context relevance" signal than what ragas's
                       own context_precision/context_recall would have
                       given you (which are themselves LLM/embedding
                       approximations). Faithfulness is the one thing this
                       project genuinely didn't have a number for yet.
  answer_relevance  : does the answer actually address the question asked?

run_eval.py deliberately calls these with a SEPARATE, stronger LLM
(config.JUDGE_MODEL) than the one that generated the answers
(config.GROQ_MODEL) -- using the small production model to judge its own
answers produced visibly unreliable scores in practice (clearly on-topic,
correct answers scored 0.0 for relevance). A judge doesn't need to be fast
or cheap; it needs to be a more careful, consistent rater than the model
under evaluation.

If a judge call fails or returns unparseable output, the score is None --
never fabricated as 0 or 1 -- and it's logged, consistent with this
project's "never invent evaluation results" rule.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checking judge. You will be given a "
            "Context, a Question, and an Answer that was generated from "
            "that Context. Break the Answer down into its individual "
            "factual claims, and judge each claim as SUPPORTED (clearly "
            "backed by the Context) or UNSUPPORTED (not stated in the "
            "Context, even if plausible or generally true).\n\n"
            "Respond with ONLY a JSON object, no other text, in this exact "
            "shape:\n"
            '{{"total_claims": <int>, "supported_claims": <int>, '
            '"unsupported_claims": [<short strings>]}}\n'
            "If the Answer makes no factual claims at all (e.g. it is a "
            "refusal), return total_claims=0, supported_claims=0, "
            "unsupported_claims=[].",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}",
        ),
    ]
)

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict judge of answer relevance. Given a Question "
            "and an Answer, judge how directly the Answer addresses what "
            "was asked, on a 0.0-1.0 scale: 1.0 = fully and directly "
            "answers the question; 0.5 = partially answers or is somewhat "
            "off-topic; 0.0 = does not address the question at all. Being "
            "factually correct is not the point here -- only judge whether "
            "it is on-topic and responsive to what was asked.\n\n"
            'Respond with ONLY a JSON object: {{"relevance_score": <float '
            "between 0.0 and 1.0>}}",
        ),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ]
)

_SOFT_REFUSAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are checking whether an AI assistant's answer amounts to "
            "declining to answer because the requested information isn't "
            "available, even if it doesn't use an explicit refusal "
            "template. Given a Question and an Answer, judge whether the "
            "Answer states or implies the information is not present, not "
            "specified, or not found in the source document -- as opposed "
            "to actually providing the requested information, even "
            "partially.\n\n"
            'Respond with ONLY a JSON object: {{"is_soft_refusal": true or '
            "false}}.",
        ),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ]
)


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_faithfulness(llm, question: str, context: str, answer: str) -> Dict:
    """Returns {"score": float|None, "total_claims", "supported_claims",
    "unsupported_claims"}. score is None if there were 0 claims to judge
    (e.g. a refusal) or the judge call/parse failed -- never fabricated."""
    chain = _FAITHFULNESS_PROMPT | llm
    try:
        result = chain.invoke({"context": context, "question": question, "answer": answer})
        raw = getattr(result, "content", str(result))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Faithfulness judge call failed: %s", exc)
        return {"score": None, "total_claims": None, "unsupported_claims": [], "error": str(exc)}

    parsed = _extract_json(raw)
    if not parsed or "total_claims" not in parsed:
        logger.warning("Faithfulness judge returned unparseable output: %r", raw[:200])
        return {"score": None, "total_claims": None, "unsupported_claims": [], "raw": raw}

    total = parsed.get("total_claims", 0) or 0
    supported = parsed.get("supported_claims", 0) or 0
    score = (supported / total) if total else None
    return {
        "score": score,
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": parsed.get("unsupported_claims", []),
    }


def judge_answer_relevance(llm, question: str, answer: str) -> Dict:
    """Returns {"score": float|None}."""
    chain = _RELEVANCE_PROMPT | llm
    try:
        result = chain.invoke({"question": question, "answer": answer})
        raw = getattr(result, "content", str(result))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer-relevance judge call failed: %s", exc)
        return {"score": None, "error": str(exc)}

    parsed = _extract_json(raw)
    if not parsed or "relevance_score" not in parsed:
        logger.warning("Answer-relevance judge returned unparseable output: %r", raw[:200])
        return {"score": None, "raw": raw}

    score = parsed.get("relevance_score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    return {"score": score}


def judge_is_soft_refusal(llm, question: str, answer: str) -> Optional[bool]:
    """For rows where the hard pre-generation refusal gate did NOT fire, but
    the question was actually unanswerable: checks whether the generated
    answer nonetheless honestly declined (e.g. "the document doesn't
    specify...") rather than guessing. This exists because `refused` in
    run_eval.py only tracks the hard REFUSAL_MESSAGE path -- a model that
    honestly admits it doesn't know, in its own words, is exhibiting the
    same safe behavior but wouldn't be counted without this check, making
    refusal_recall look worse than the system's actual behavior.

    Returns None (not fabricated as True/False) if the judge call fails or
    output is unparseable.
    """
    chain = _SOFT_REFUSAL_PROMPT | llm
    try:
        result = chain.invoke({"question": question, "answer": answer})
        raw = getattr(result, "content", str(result))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Soft-refusal judge call failed: %s", exc)
        return None

    parsed = _extract_json(raw)
    if not parsed or "is_soft_refusal" not in parsed:
        logger.warning("Soft-refusal judge returned unparseable output: %r", raw[:200])
        return None
    return bool(parsed["is_soft_refusal"])
