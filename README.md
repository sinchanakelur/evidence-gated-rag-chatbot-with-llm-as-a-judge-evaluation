# Evidence-Gated RAG Chatbot with LLM-as-a-Judge Evaluation

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
- Precision: 29% → 87%
- Hit Rate: 100%
- Mean Faithfulness: 0.93
- Answer Relevance: 0.93
- Median Latency: 788 ms
- 18-question golden set
---


## Future Improvements

- Calibrate `RERANK_SCORE_THRESHOLD` using evaluation results for better refusal accuracy.
- Add per-claim citations by mapping generated statements to supporting chunk IDs.
- Persist FAISS and BM25 indexes for faster reuse across sessions.
- Add MMR-based diversity to reduce near-duplicate chunks in retrieved results.
