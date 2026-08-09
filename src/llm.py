"""Groq LLM client construction."""
import os

import streamlit as st
from langchain_groq import ChatGroq

from . import config


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str = config.GROQ_MODEL) -> ChatGroq:
    """Reuse one Groq client per model across the whole session.

    The original app called `ChatGroq(...)` fresh on every single chat
    message - harmless functionally, but unnecessary client-construction
    overhead on every turn. Cached here instead.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or to "
            ".streamlit/secrets.toml as GROQ_API_KEY."
        )
    return ChatGroq(groq_api_key=api_key, model_name=model_name)
