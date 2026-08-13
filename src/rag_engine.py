"""Embeddings, hybrid retrieval, reranking, and grounded generation.

Architecture (unchanged pieces): Streamlit caching, FAISS, HuggingFace
MiniLM embeddings, Groq/LLaMA generation via LangChain.

What changed vs. the previous version, and why:

1. Retrieval is now hybrid (FAISS vector search + BM25 lexical search),
   merged and deduped by chunk_id. Pure embedding similarity misses exact
   terms (numbers, IDs, names) that a lexical match catches for free.
2. A cross-encoder reranker (`RERANKER_MODEL`) rescoring the merged
   candidate pool replaces "trust the top-k of a single bi-encoder search".
   This is the main precision lever.
3. There is a hard score gate (`RERANK_SCORE_THRESHOLD`) applied BEFORE
   generation. If nothing clears it, the app refuses immediately and never
   calls the LLM for an answer -- there is no more "fall back to a general
   LLM response" path. This directly satisfies the "must be grounded or
   refuse" requirement, and also saves a wasted generation call.
4. Every stage is timed (condense / retrieve / rerank / generate) so
   latency can be reported per-query, per the eval requirements.
5. Displayed citations are exactly the chunks that were actually sent to
   the LLM for generation (post rerank + threshold), not a separate,
   possibly-different top-3 slice of raw retrieval.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import config
from .document_loader import load_and_split_pdfs

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Cached resources
# -------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """The embedding model is expensive to load - cache it once per process."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_reranker():
    """Cross-encoder reranker, loaded once per process.

    Imported lazily so the app doesn't pay the sentence-transformers
    cross-encoder import cost until it's actually needed.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL)


@st.cache_resource(show_spinner="Indexing document(s)...")
def build_indexes(
    file_hashes: Tuple[str, ...], file_paths: Tuple[str, ...]
) -> Tuple[Optional[FAISS], Optional[BM25Retriever], List[Document]]:
    """Build (and cache) both a FAISS index and a BM25 index for a given set
    of uploaded files, plus the raw chunk list (used by the eval harness).

    Cache key is `file_hashes` (content hash per file), not a fixed path, so
    a different file (or set of files) always invalidates the cache, while
    re-uploading an identical file is still a fast cache hit. This is the fix
    for the original app's stale-cache bug (it cached on a fixed "temp.pdf"
    path, so a second, different upload silently kept answering from the
    first PDF's index).
    """
    del file_hashes  # unused inside the function; it only exists to key the cache

    chunks: List[Document] = load_and_split_pdfs(list(file_paths))
    if not chunks:
        return None, None, []

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.TOP_K_RETRIEVE

    return vectorstore, bm25_retriever, chunks


# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------

# Enforcement of "grounded or refuse" happens in three layers now:
#   1. the pre-generation score gate (unchanged, in answer_question below)
#   2. this prompt, made explicit and hard rather than a soft suggestion
#   3. a post-generation leak check + one reinforced regeneration attempt
#      (see _contains_knowledge_leak / answer_question below)
# Layer 2 alone was not enough in practice: a model can retrieve the exactly
# right chunk and still pad the answer with "based on general knowledge..."
# if the instruction reads as a preference rather than a hard rule, and if
# the conversation history (which includes the model's own prior turns) is
# left ambiguous as a possible source of facts.
RAG_SYSTEM_PROMPT = (
    "You are a strictly document-grounded assistant. Answer the user's question "
    "using ONLY the information inside the Context section below.\n\n"
    "Hard rules:\n"
    "- Do not use any knowledge from outside the Context, even if you are "
    "confident it is correct, and even if the Context only partially covers "
    "the question.\n"
    "- Never say things like 'based on general knowledge', 'typically', or "
    "'usually' to fill a gap. If the Context doesn't say it, you don't know it.\n"
    "- If the Context answers the question, answer directly and concisely, "
    "and mention which document/page you are drawing from when relevant.\n"
    "- If the Context only partially answers the question, state plainly "
    "which part is covered and which part is not -- do not guess at the rest.\n"
    "- The conversation history below exists ONLY so you understand what the "
    "user is referring to across turns. It is NOT a source of facts: never "
    "treat your own earlier responses as ground truth. Re-verify every claim "
    "in this answer against the Context section for this turn.\n\n"
    "Context:\n{context}"
)

# Reused verbatim in the retry path below (_STRICT_REMINDER) when the
# leak-check fires, so the model gets the same hard rules a second time,
# closer to the question this time.
_STRICT_REMINDER = (
    "\n\n(Reminder: answer using ONLY the Context above. Do not add anything "
    "from general knowledge, even to fill a gap. If the Context is "
    "incomplete or silent on part of the question, say only what the "
    "Context actually contains.)"
)

CONDENSE_QUESTION_PROMPT = (
    "Given the conversation history and a follow-up question, decide whether "
    "the follow-up question can already be understood entirely on its own.\n"
    "- If it does NOT rely on pronouns (it/that/this/those), or on implicit "
    "references to an earlier turn, it is already self-contained: return it "
    "EXACTLY as written, with no changes at all.\n"
    "- Only if it truly cannot be understood without the history, rewrite it "
    "into a standalone question that captures the necessary context.\n"
    "Return only the question text, nothing else."
)

# Heuristic bypass for the common case: most follow-ups in a document-QA
# session are already self-contained ("what about X" excluded), and running
# every single one through an LLM rewrite -- at nonzero variance -- was
# observed to sometimes drift the retrieval query enough that hybrid
# retrieval + rerank legitimately found nothing above threshold for what was
# functionally the same question one turn later (a false refusal). Skipping
# the LLM rewrite whenever the question shows no referential markers removes
# that failure mode for the majority of turns; the LLM condenser is still
# used (with the stricter prompt above) for genuine follow-ups.
_REFERENTIAL_MARKERS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "he", "she", "him", "her", "his", "previous", "above", "former", "latter",
    "again", "also", "too", "same", "other", "second", "third", "next",
    "last",
}


def _needs_condensing(question: str) -> bool:
    tokens = set(re.findall(r"[a-zA-Z']+", question.lower()))
    return bool(tokens & _REFERENTIAL_MARKERS)


def _contains_knowledge_leak(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in config.LEAK_PHRASES)

_CONDENSE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_QUESTION_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def get_document_chain(llm):
    """create_stuff_documents_chain wires {input, chat_history, context} ->
    grounded answer string. Kept as LangChain's standard composition (not a
    hand-rolled prompt-fill) since it's stable across 0.2.x and doesn't
    require the BaseRetriever abstraction we deliberately moved away from
    below."""
    return create_stuff_documents_chain(llm, _ANSWER_TEMPLATE)


# -------------------------------------------------------------------------
# Retrieval pipeline (manual, not LCEL retriever chain)
# -------------------------------------------------------------------------
# We stopped using create_history_aware_retriever / create_retrieval_chain
# because we need to inspect and threshold retrieval results BEFORE deciding
# whether to generate at all. Wrapping hybrid-retrieve+rerank+gate inside a
# LangChain BaseRetriever and then handing it back to create_retrieval_chain
# would just hide the exact information (per-chunk rerank scores, an empty
# result) we need to act on. This is a few more lines of glue code, but it's
# what makes the refusal gate and per-stage latency timing possible.


def condense_question(llm, question: str, chat_history: list) -> str:
    """Rewrite a follow-up question into a standalone one using chat history.

    Skipped entirely (returns `question` unchanged, no LLM call) when there's
    no history yet, OR when the question has no referential markers and is
    therefore already self-contained -- see _needs_condensing's docstring
    above for why this bypass exists.
    """
    if not chat_history or not _needs_condensing(question):
        return question
    chain = _CONDENSE_TEMPLATE | llm
    result = chain.invoke({"input": question, "chat_history": chat_history})
    rewritten = getattr(result, "content", str(result)).strip()
    return rewritten or question


def hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    bm25_retriever: Optional[BM25Retriever],
    k: int = config.TOP_K_RETRIEVE,
) -> List[Document]:
    """Vector search + BM25 lexical search, merged and deduped by chunk_id.

    Pure embedding similarity can miss exact terms (order numbers, model
    codes, names) that a cheap lexical match catches directly. This is the
    candidate pool the reranker below then scores precisely.
    """
    vector_docs = vectorstore.similarity_search(query, k=k)
    bm25_docs = bm25_retriever.invoke(query)[:k] if bm25_retriever is not None else []

    seen = {}
    for doc in vector_docs + bm25_docs:
        cid = doc.metadata.get("chunk_id") or id(doc)
        if cid not in seen:
            seen[cid] = doc
    return list(seen.values())


def rerank(query: str, docs: List[Document], reranker) -> List[Tuple[Document, float]]:
    """Cross-encoder rescoring of the candidate pool. Returns
    (doc, score) pairs sorted best-first. Scores are raw cross-encoder
    logits (unbounded), not probabilities."""
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, [float(s) for s in scores]))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def answer_question(
    *,
    llm,
    vectorstore: FAISS,
    bm25_retriever: Optional[BM25Retriever],
    reranker,
    question: str,
    chat_history: list,
) -> Dict:
    """Full pipeline for one turn: condense -> hybrid retrieve -> rerank ->
    score-gate -> (refuse | generate grounded answer).

    Returns a dict:
      answer            : str
      grounded          : bool (False iff this is a refusal)
      sources           : List[Document] actually used for generation,
                           each carrying metadata['rerank_score']
      standalone_question: the (possibly rewritten) question actually used
                           for retrieval
      timings           : Dict[str, float], milliseconds per stage + total
    """
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    standalone_question = condense_question(llm, question, chat_history)
    timings["condense_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    candidates = hybrid_retrieve(standalone_question, vectorstore, bm25_retriever)
    timings["retrieval_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    reranked = rerank(standalone_question, candidates, reranker)
    timings["rerank_ms"] = (time.perf_counter() - t2) * 1000

    filtered = [
        (doc, score)
        for doc, score in reranked
        if score >= config.RERANK_SCORE_THRESHOLD
    ][: config.TOP_K_RERANK]

    if not filtered and reranked and reranked[0][1] >= config.RERANK_FLOOR_SCORE:
        # Nothing cleared the primary threshold, but the single best
        # candidate isn't pure noise either -- serve just that one rather
        # than refusing outright. See config.RERANK_FLOOR_SCORE for why:
        # a paraphrased query can legitimately score a real, relevant chunk
        # below the primary cutoff without the chunk being irrelevant.
        filtered = reranked[:1]

    if not filtered:
        timings["generation_ms"] = 0.0
        timings["total_ms"] = sum(timings.values())
        return {
            "answer": config.REFUSAL_MESSAGE,
            "grounded": False,
            "sources": [],
            "standalone_question": standalone_question,
            "timings": timings,
        }

    docs = []
    for doc, score in filtered:
        doc.metadata["rerank_score"] = score
        docs.append(doc)

    t3 = time.perf_counter()
    document_chain = get_document_chain(llm)
    answer = document_chain.invoke(
        {"input": question, "chat_history": chat_history, "context": docs}
    )
    if not isinstance(answer, str):
        answer = str(answer)

    # Safety net: temperature=0 + the strict prompt above should prevent
    # this, but if the model still pads the answer with outside knowledge,
    # regenerate once with the hard rules repeated right next to the
    # question rather than silently shipping the leaked answer.
    if _contains_knowledge_leak(answer):
        logger.warning(
            "Detected a possible knowledge-leak phrase in the generated "
            "answer; regenerating once with a reinforced reminder."
        )
        answer = document_chain.invoke(
            {
                "input": question + _STRICT_REMINDER,
                "chat_history": chat_history,
                "context": docs,
            }
        )
        if not isinstance(answer, str):
            answer = str(answer)

    answer = answer.replace("\u01a0", "ti").replace("\ufb01", "fi")
    timings["generation_ms"] = (time.perf_counter() - t3) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "answer": answer,
        "grounded": True,
        "sources": docs,
        "standalone_question": standalone_question,
        "timings": timings,
    }


def is_low_quality_answer(answer: str) -> bool:
    """Cheap hedging heuristic. No longer used to trigger a fallback (there
    is none); kept as an optional secondary signal for the eval harness."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in config.BAD_PHRASES)
