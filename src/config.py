"""Centralized configuration for the RAG chatbot.

Keeping these as named constants (instead of magic numbers scattered through
app.py / rag_engine.py) is what makes it possible to tune retrieval quality
later without hunting through code, and is what the eval harness in
`eval/` tunes against.
"""
import os

# ---- Embedding & LLM settings ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq deprecated `llama-3.1-8b-instant` on 2026-06-17; shutdown 2026-08-16
# (confirmed at https://console.groq.com/docs/deprecations). After that date
# requests to the old model ID return errors, not warnings. `openai/gpt-oss-20b`
# is Groq's official recommended replacement.
#
# gpt-oss-20b is a reasoning model -- by default it can emit visible
# chain-of-thought alongside the final answer. `GROQ_REASONING_EFFORT` +
# `reasoning_format="hidden"` (applied in src/llm.py via model_kwargs) keep
# it behaving like a normal fast chat model for this project's purposes:
# `content` is the final answer only, never mixed with reasoning tokens --
# this matters because the leak-phrase check and citation display in
# rag_engine.py both assume `content` is clean prose, not reasoning + prose.
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = "low"  # fastest/cheapest; this task is extraction, not deep reasoning

# ---- LLM-judge (eval only, see eval/llm_judge.py) ----
# Deliberately a stronger model than GROQ_MODEL above -- standard practice
# for LLM-as-judge: the judge doesn't need to be fast or cheap, it needs to
# be a more careful, reliable rater than the model being judged. Using the
# small model as its own judge produced visibly unreliable scores in
# practice (correct, clearly on-topic answers scored 0.0 for relevance).
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_REASONING_EFFORT = "medium"

# 0.0, deliberately. At the previous default (non-zero) temperature, the
# exact same underlying question could be answered correctly on one turn and
# either padded with outside knowledge or wrongly refused on the next, from
# sampling variance alone. Determinism matters more than variety here.
LLM_TEMPERATURE = 0.0

# ---- Chunking ----
# 800/120 keeps chunks coherent (vs. 400/50, which fragments sentences and
# hurts recall with MiniLM + an 8B model) while still fitting comfortably in
# context.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_LENGTH = 30  # discard near-empty chunks (headers, page numbers, etc.)

# ---- Retrieval (first stage: cheap, high recall) ----
# We deliberately over-retrieve here (vector + BM25) and let the reranker
# below narrow down to what's actually relevant. A single bi-encoder top-k
# has a known precision ceiling; widening the candidate pool is what gives
# the reranker something to work with.
TOP_K_RETRIEVE = 15

# ---- Reranking (second stage: expensive, high precision) ----
# Cross-encoder reranker scores each (query, chunk) pair jointly, which is
# far more precise than bi-encoder cosine similarity. This is the single
# highest-leverage change for answer quality without touching the embedding
# model or index.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 4  # how many reranked chunks are actually shown to the LLM

# Cross-encoder ms-marco-MiniLM-L-6-v2 outputs an unbounded logit, not a
# probability, and it is NOT calibrated for your documents out of the box.
# Empirically: a chunk that shares an exact phrase with the query (e.g. query
# "soft skills" against a chunk literally headed "Soft Skills") scores high
# (5-7+), but a genuinely relevant chunk the query only paraphrases (e.g.
# "Key Responsibilities?" against a bullet list with no such header) can
# easily score close to, or below, 0. A single hard threshold at 0.0 was
# observed to correctly gate out unrelated content but ALSO wrongly refuse
# real, present content purely because the query didn't share vocabulary
# with it -- that's a false refusal, not a correct one.
#
# RERANK_SCORE_THRESHOLD is the primary cutoff: candidates at or above this
# are served to generation.
RERANK_SCORE_THRESHOLD = -2.0

# RERANK_FLOOR_SCORE is a secondary, much looser safety net: if NOTHING
# clears the primary threshold, but the single best candidate is still above
# this floor (i.e. not pure noise), that one candidate is served anyway
# rather than refusing outright. This is what lets a weak-but-real match
# survive a strict primary threshold, while a truly unrelated question
# (whose best candidate score is deep negative) still gets refused.
RERANK_FLOOR_SCORE = -6.0

# Tune both against eval/run_eval.py's Recall@K / Precision@K / Hit Rate /
# refusal-quality output on your own golden set rather than trusting these
# starting values -- they were chosen from the false-refusal pattern
# observed in testing, not from a calibrated study.

# ---- Refusal ----
# If nothing clears RERANK_SCORE_THRESHOLD, the app refuses BEFORE calling
# the LLM for generation at all -- there is no general-knowledge fallback.
REFUSAL_MESSAGE = (
    "I couldn't find this in the uploaded document(s). Try rephrasing the "
    "question, or upload a document that covers this topic."
)

# ---- Knowledge-leak detection ----
# Even with a strict prompt and temperature=0, an 8B model can still pad a
# grounded answer with outside knowledge (e.g. "based on general knowledge,
# ..."). This is a safety net, not the primary defense: if any of these
# phrases show up in a generated answer, rag_engine.answer_question()
# regenerates once with a reinforced reminder appended to the question.
LEAK_PHRASES = [
    "general knowledge",
    "based on general",
    "in general,",
    "typically,",
    "usually,",
    "commonly known",
    "as a rule,",
    "from what i know",
    "as far as i know",
]

# ---- Legacy fallback-detection heuristic ----
# No longer used to trigger a general-LLM fallback (that behavior has been
# removed). Kept only as an optional secondary signal inside the eval
# harness (a cheap "did the model hedge anyway" check alongside the real
# faithfulness metric).
BAD_PHRASES = [
    "don't know",
    "do not know",
    "not mentioned",
    "not provided",
    "no information",
    "cannot find",
    "i don't have access",
]

# ---- Storage ----
# Uploaded PDFs are cached on disk by content hash so re-uploading the same
# file doesn't rewrite it, and switching files doesn't collide on one path.
UPLOAD_DIR = os.path.join(os.getcwd(), ".uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
