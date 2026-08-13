# Building a golden set

`run_eval.py` needs a hand-labeled golden set — there's no way around this;
retrieval/generation metrics are only meaningful relative to ground truth
you define, and this project intentionally does not fabricate one for you.

## Steps

1. **Pick 2–3 real PDFs** you'll actually use for your interview / portfolio
   demo. Put them in one folder, e.g. `eval/sample_pdfs/`.

2. **Inspect the chunks** so you can reference real `chunk_id`s:

   ```bash
   python eval/inspect_chunks.py --pdf-dir eval/sample_pdfs
   # or narrow it down:
   python eval/inspect_chunks.py --pdf-dir eval/sample_pdfs --grep "revenue"
   ```

   This prints every chunk with its `chunk_id`, e.g.
   `financials.pdf::p4::2`, and a text preview.

3. **Write 25–40 questions** against those documents, split roughly:
   - ~70% answerable questions, each with:
     - `question`: the question text
     - `is_answerable: true`
     - `relevant_chunks`: the `chunk_id`(s) from step 2 that actually
       contain the answer (use more than one if the answer spans chunks)
     - `reference_answer`: a short correct answer, used both as a sanity
       check and as RAGAS's `ground_truth` for faithfulness/relevance
   - ~30% deliberately **unanswerable** questions (plausible-sounding but
     not covered by the documents), each with:
     - `is_answerable: false`
     - `relevant_chunks: []`
     - `reference_answer: ""`

   This unanswerable slice is what lets `refusal_quality` mean anything —
   without it, you can't distinguish "the bot never refuses" from
   "the bot correctly never needs to."

4. Save as `eval/golden_set.json` (see `golden_set.template.json` for the
   exact shape — it's a plain JSON array of these objects).

## Running

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

python eval/run_eval.py \
    --golden-set eval/golden_set.json \
    --pdf-dir eval/sample_pdfs \
    --output eval/results.json \
    --llm-judge
```

`--llm-judge` computes faithfulness and answer relevance in-process, using
the same Groq LLM already pinned in `requirements.txt` — no new
dependencies. This is the primary generation-quality metric for this
project (see `eval/llm_judge.py`).

`ragas` (context_precision/context_recall on top of the above) is optional
and not required — its dependencies were confirmed incompatible with this
project's pinned langchain stack twice in practice (an unresolvable
`langchain-core` version conflict, then an unrelated upstream bug in `ragas`
itself). If you want to try it anyway, `--ragas` writes
`eval/ragas_input.json`, and a **second, separate** virtual environment runs
it independently of everything above:

```bash
python -m venv venv-eval
venv-eval\Scripts\activate        # Windows; use `source venv-eval/bin/activate` on macOS/Linux
pip install -r eval/requirements-eval.txt   # ragas, datasets, langchain-groq only
export GROQ_API_KEY=your_key_here            # separate shell/session, set it again

python eval/run_ragas.py --input eval/ragas_input.json --output eval/ragas_results.json
```

`eval/run_ragas.py` deliberately imports nothing from `src/`, so it has zero
dependency on the pinned 0.2.x stack and can safely live alongside a modern
`ragas`. Keep these two environments separate if you go this route — don't
`pip install ragas` into the environment that runs `app.py` / `run_eval.py`,
and don't run `pip install -r requirements.txt` inside `venv-eval`.

## Reading the output

- **`retrieval_raw` vs `retrieval_served`**: `raw` is the hybrid
  vector+BM25 pool before reranking; `served` is what's left after the
  cross-encoder rerank + score threshold — i.e. what generation actually
  sees. The delta between the two is your reranker's measured contribution.
  If `served` isn't clearly better than `raw` on Recall/Precision/MRR/Hit
  Rate, that's a signal to revisit `RERANK_SCORE_THRESHOLD` or
  `TOP_K_RERANK` in `src/config.py`, not to assume the reranker is helping.
- **`refusal_quality.refusal_recall`**: on genuinely unanswerable
  questions, fraction correctly refused. Low → `RERANK_SCORE_THRESHOLD` is
  too permissive.
- **`refusal_quality.false_refusal_rate`**: on genuinely answerable
  questions, fraction wrongly refused. High → threshold is too strict, or
  `TOP_K_RETRIEVE` is too small to surface the right chunk at all.
- **`generation_quality.faithfulness` / `.answer_relevance`**: only present
  with `--llm-judge`. Faithfulness is `supported_claims / total_claims` per
  answer (an LLM judge decomposes the answer into claims and checks each
  against the retrieved context), averaged across all non-refused answers.
  A low faithfulness score with a high `retrieval_served` recall/precision
  means retrieval found the right chunks but generation still drifted from
  them — worth revisiting the grounding prompt in `src/rag_engine.py`
  (`RAG_SYSTEM_PROMPT`) or `LLM_TEMPERATURE` in `src/config.py`, not
  retrieval.
- **`ragas`**: optional and secondary — see "Running" above. If you didn't
  run `--ragas` + `eval/run_ragas.py`, this is `null`, same as anything else
  not computed; not a 0.
- **`latency_ms`**: mean/median/p95 for each pipeline stage plus total,
  across the whole golden set.

## Re-tuning `RERANK_SCORE_THRESHOLD`

The cross-encoder returns an unbounded logit, not a calibrated probability,
so the default `0.0` in `src/config.py` is a starting guess, not a
validated value. After a first eval run, look at `per_question` in
`results.json`: for questions where `served_ids` is empty but
`is_answerable` is true, check the `rerank_score` the relevant chunk
would have gotten (visible in `reranked_ids_all`'s ordering / the raw
scores you can print from `rerank()` directly) — that tells you how much
to lower the threshold, and vice versa for false positives on unanswerable
questions.
