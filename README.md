# Smart RAG Chatbot

A grounded RAG chatbot: hybrid (vector + BM25) retrieval, cross-encoder
reranking, a hard relevance gate before generation, and real conversational
memory over the documents. **No general-LLM fallback** — if nothing in the
uploaded documents clears the relevance threshold, the app refuses instead
of answering from parametric knowledge.

## Features

- Upload one or more PDFs and ask questions across all of them
- **Hybrid retrieval**: FAISS vector search + BM25 lexical search, merged
  and deduped, so exact terms (IDs, numbers, names) aren't missed by
  embedding similarity alone
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over
  the merged candidate pool, replacing raw bi-encoder top-k
- **Score-gated refusal**: if nothing clears `RERANK_SCORE_THRESHOLD`, the
  app refuses *before* calling the LLM for generation — not a post-hoc
  string-match on a generated answer
- True conversational memory — follow-up questions are understood using
  chat history via query condensation, not just replayed in the UI
- Citations shown are exactly the chunks that were sent to the LLM for
  generation (with their rerank score), not a separate raw top-k slice
- Per-query latency (retrieval / rerank / generation / total), shown in the
  UI and logged for the eval harness
- FAISS + BM25 indexes cached per file content hash, so a new upload always
  re-indexes correctly and re-uploading the same file is instant

## Project structure

```
rag_chatbot/
├── app.py                  # Streamlit UI only — no business logic
├── src/
│   ├── config.py            # All tunable constants (chunking, k, threshold, etc.)
│   ├── document_loader.py   # PDF loading, cleaning, chunking, chunk_id assignment
│   ├── rag_engine.py        # Hybrid retrieval, reranking, refusal gate, generation
│   ├── llm.py                # Groq client (cached)
│   └── utils.py              # Upload persistence / hashing
├── eval/
│   ├── metrics.py            # Recall@K, Precision@K, MRR, Hit Rate, refusal quality
│   ├── inspect_chunks.py     # Dump chunk_ids to help build a golden set
│   ├── run_eval.py           # Main evaluation runner (retrieval + refusal + latency)
│   ├── run_ragas.py          # Separate RAGAS runner -- run from its OWN venv, see below
│   ├── golden_set.template.json
│   ├── GOLDEN_SET_GUIDE.md   # How to build a golden set and read results
│   └── requirements-eval.txt # For the separate RAGAS venv only
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # or put it in .streamlit/secrets.toml
streamlit run app.py
```

## Evaluation

```bash
python eval/inspect_chunks.py --pdf-dir eval/sample_pdfs
# ... build eval/golden_set.json by hand, see eval/GOLDEN_SET_GUIDE.md ...
python eval/run_eval.py --golden-set eval/golden_set.json --pdf-dir eval/sample_pdfs --llm-judge
```

This computes, on your own labeled questions and documents, against the
**actual production pipeline** (not a mock): Recall@K, Precision@K, MRR, Hit
Rate (both raw retrieval and post-rerank, so you can see the reranker's
measured contribution), refusal quality, per-stage latency, and — with
`--llm-judge` — faithfulness and answer relevance. No numbers are
fabricated; anything not computed is reported as `null`, not 0.

`--llm-judge` runs entirely in this same environment, using the same Groq
LLM already pinned in `requirements.txt` — no new dependencies, no second
virtual environment. It's the primary generation-quality metric for this
project (see `eval/llm_judge.py` for the full rationale).

`ragas` (context_precision/context_recall on top of the above) is optional
and no longer required: its dependencies proved incompatible with this
project's pinned langchain stack in practice — two separate, confirmed
failures (an unresolvable `langchain-core` version conflict, then an
unrelated upstream bug in `ragas` itself). If you still want to try it,
`--ragas` writes `eval/ragas_input.json`, and `eval/run_ragas.py` runs it
from a second, separate virtual environment (`eval/requirements-eval.txt`)
— but `--llm-judge` above already gets you a real faithfulness/relevance
number with zero extra setup.

---

## What changed from the previous version, and why

This iteration keeps the existing architecture and technologies (Streamlit,
LangChain, FAISS, HuggingFace MiniLM, Groq/LLaMA) and focuses entirely on
retrieval/generation quality and provable evaluation.

1. **Removed the general-LLM fallback.** The old app fell back to
   `llm.invoke(prompt)` (ungrounded) whenever retrieval was empty or the
   generated answer string-matched a "hedging" phrase list. That's gone.
   Now there's a hard score gate on reranked retrieval results, checked
   *before* generation is even called; if nothing clears it, the app
   refuses with a fixed message and never asks the LLM to answer.
2. **Hybrid retrieval.** Added a BM25 lexical retriever alongside the
   existing FAISS vector retriever; results are merged and deduped by a new
   deterministic `chunk_id`. Pure embedding similarity was missing exact
   terms that lexical search catches directly.
3. **Cross-encoder reranking.** Vector+BM25 now over-retrieve
   (`TOP_K_RETRIEVE=15`) into a candidate pool, which a cross-encoder
   (`ms-marco-MiniLM-L-6-v2`) rescores before the top `TOP_K_RERANK=4` are
   kept. This is the main precision lever versus the old plain bi-encoder
   top-k.
4. **Citations grounded to actual usage.** Previously the UI showed the raw
   top-3 of retrieval, which wasn't necessarily what the answer was built
   from. Now it shows exactly the post-rerank, post-threshold chunks that
   were passed to the LLM, each with its rerank score.
5. **Manual retrieval pipeline instead of `create_history_aware_retriever` /
   `create_retrieval_chain`.** Those LCEL chains hide retrieval results
   inside the chain — you can't inspect scores or gate on them before
   generation. Replaced with explicit `condense_question` →
   `hybrid_retrieve` → `rerank` → threshold → `document_chain.invoke(...)`,
   which is what makes the refusal gate and per-stage latency timing
   possible. Conversational follow-up handling (question condensing using
   chat history) is preserved, just done explicitly instead of inside a
   retriever wrapper.
6. **Per-stage latency instrumentation** (condense / retrieval / rerank /
   generation / total), surfaced in the UI and logged by the eval harness.
7. **`chunk_id` metadata** (`<filename>::p<page>::<index>`), deterministic
   given a fixed chunking config, added at load time — this is the join key
   the eval harness uses to compare retrieved chunks against a hand-labeled
   golden set, and is new relative to the previous version.
8. **Reproducible evaluation harness** (`eval/`): Recall@K, Precision@K,
   MRR, Hit Rate (computed exactly, no LLM judge needed), refusal quality,
   per-stage latency stats, and an optional RAGAS wrapper for faithfulness /
   answer relevance / context precision / context recall. Nothing here
   ships with invented baseline numbers — the harness needs your own PDFs
   and a golden set you label (see `eval/GOLDEN_SET_GUIDE.md`), by design.
9. **No PDF, no chat.** The app no longer offers a general-knowledge chat
   mode when no PDF is uploaded — this is now strictly a document-grounded
   assistant, consistent with removing the fallback above.

### Known trade-offs / next steps worth considering

- `RERANK_SCORE_THRESHOLD = 0.0` in `src/config.py` is a reasonable
  starting point, not a calibrated value for your documents — tune it
  against `eval/run_eval.py` output (see `GOLDEN_SET_GUIDE.md`).
- Citations are shown at chunk granularity, not per-claim. A further step
  would be prompting the model to cite chunk IDs inline per sentence and
  parsing them back out, so a multi-fact answer can point to exactly which
  chunk backs which claim.
- FAISS + BM25 indexes are rebuilt (from Streamlit cache) per session — fine
  at this scale; a persisted index would matter if the same PDFs are reused
  across many sessions/users.
- MMR-style diversity on the reranked set (to reduce near-duplicate chunks
  from the same page) isn't implemented; worth adding if your documents have
  a lot of repeated boilerplate.
