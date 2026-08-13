"""Standalone RAGAS metrics runner.

RUN THIS FROM A SEPARATE VIRTUAL ENVIRONMENT than the one you use for
app.py / run_eval.py. Do not `pip install ragas` into that environment.

Why: this project's RAG pipeline (src/rag_engine.py) is pinned to
langchain==0.2.16 / langchain-core==0.2.38, because that's the last version
langchain-groq==0.1.9 supports. ragas's own dependencies require a much
newer langchain-core (1.x+), which restructures the package -- for example
`langchain.chains.combine_documents` (used by src/rag_engine.py) moves into
a separate `langchain-classic` package and isn't importable from the 1.x
`langchain` package at all. Installing both dependency trees in one
environment is not possible without breaking one of them; this was tried and
confirmed broken (ModuleNotFoundError: No module named 'langchain.chains').

So this script is deliberately self-contained: it imports NOTHING from
src/, so it never touches the pinned 0.2.x stack, and can happily live in
an environment with a modern `ragas` + `langchain-core` instead.

Setup (one time):
    python -m venv venv-eval
    venv-eval\\Scripts\\activate        (Windows)   or   source venv-eval/bin/activate (macOS/Linux)
    pip install ragas datasets langchain-groq

Usage:
    # 1. From your main app environment, generate the input file:
    python eval/run_eval.py --golden-set eval/golden_set.json \\
        --pdf-dir eval/sample_pdf --ragas

    # 2. Switch to venv-eval, then:
    python eval/run_ragas.py --input eval/ragas_input.json \\
        --output eval/ragas_results.json

Requires GROQ_API_KEY to be set in venv-eval too (RAGAS uses an LLM judge
for faithfulness/answer_relevancy).
"""
from __future__ import annotations

import argparse
import json
import os


def build_groq_llm(model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0):
    """Builds its own Groq client rather than importing src.llm.get_llm --
    that import chain pulls in the pinned 0.2.x langchain stack this script
    is specifically meant to avoid. Whatever langchain-groq version
    ragas's resolver picked in this environment is used as-is."""
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set in this environment. Add it here too "
            "(it's separate from your main app environment's shell/session)."
        )
    return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval/ragas_input.json")
    parser.add_argument("--output", default="eval/ragas_results.json")
    args = parser.parse_args()

    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit(
            "ragas/datasets aren't installed in THIS environment. Run:\n"
            "  pip install -r eval/requirements-eval.txt\n"
            "in the separate venv you created for this script (not the one "
            "running app.py)."
        )
    except ModuleNotFoundError as exc:
        if "vertexai" in str(exc):
            raise SystemExit(
                "Import failed inside ragas itself: " + str(exc) + "\n"
                "This is a known upstream bug in ragas>=0.4 (ragas/llms/base.py "
                "hardcodes an import path for ChatVertexAI that was moved to "
                "langchain-google-vertexai in current langchain-community). "
                "Fix:\n"
                "  pip uninstall -y ragas\n"
                "  pip install ragas==0.3.9\n"
                "(eval/requirements-eval.txt already pins this -- re-run "
                "`pip install -r eval/requirements-eval.txt` if you installed "
                "ragas some other way)."
            )
        raise

    if not os.path.exists(args.input):
        raise SystemExit(
            f"{args.input} not found. Generate it first, from your main app "
            "environment, with:\n"
            "  python eval/run_eval.py --golden-set eval/golden_set.json "
            "--pdf-dir <your-pdf-dir> --ragas"
        )

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("questions"):
        raise SystemExit(f"{args.input} has no questions to score.")

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
            "question": data["questions"],
            "answer": data["answers"],
            "contexts": data["contexts"],
            "ground_truth": data["ground_truths"],
        }
    )

    print(f"Scoring {len(data['questions'])} question(s) with RAGAS...")
    llm = build_groq_llm()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )
    summary = dict(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
