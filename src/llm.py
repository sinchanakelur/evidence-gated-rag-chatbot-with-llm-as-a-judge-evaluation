"""Groq LLM client construction."""
import os

import streamlit as st
from langchain_groq import ChatGroq

from . import config


@st.cache_resource(show_spinner=False)
def get_llm(
    model_name: str = config.GROQ_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
    reasoning_effort: str = config.GROQ_REASONING_EFFORT,
) -> ChatGroq:
    """Reuse one Groq client per model across the whole session.

    The original app called `ChatGroq(...)` fresh on every single chat
    message - harmless functionally, but unnecessary client-construction
    overhead on every turn. Cached here instead. Cache key includes
    model_name/temperature/reasoning_effort, so eval/run_eval.py can request
    a second, separate client (a stronger judge model) without colliding
    with the production client's cache entry.

    temperature=0 (config.LLM_TEMPERATURE) is deliberate, not a default we
    happened to leave alone: at the previous default (non-zero) temperature,
    the same question could get answered correctly on one turn and refused
    or padded with outside knowledge on the next, purely from sampling
    variance -- both the grounding leak and the retrieval-drift bugs this
    fixes are much more likely at temperature > 0.

    model_kwargs passes reasoning_effort/reasoning_format straight through
    to Groq's API in the outgoing request body -- confirmed working with
    this pinned langchain-groq==0.1.9 client by inspecting its source
    (ChatGroq._default_params spreads self.model_kwargs directly into the
    request payload), even though 0.1.9 predates gpt-oss/reasoning models
    and has no dedicated field for them. reasoning_format="hidden" is not
    optional: without it, gpt-oss can emit visible chain-of-thought mixed
    into `content`, which would corrupt both the leak-phrase check in
    rag_engine.py and the strict-JSON parsing in eval/llm_judge.py.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or to "
            ".streamlit/secrets.toml as GROQ_API_KEY."
        )
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        model_kwargs={"reasoning_effort": reasoning_effort, "reasoning_format": "hidden"},
    )
