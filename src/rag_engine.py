"""Embeddings, vector store construction, and the conversational RAG chain."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import config
from .document_loader import load_and_split_pdfs

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """The embedding model is expensive to load - cache it once per process
    instead of (implicitly) reloading it on every vector store rebuild."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Indexing document(s)...")
def build_vectorstore(
    file_hashes: Tuple[str, ...], file_paths: Tuple[str, ...]
) -> Optional[FAISS]:
    """Build (and cache) a FAISS index for a given set of uploaded files.

    IMPORTANT FIX vs. the original app: the original cached this function on a
    fixed path ("temp.pdf"), so after the very first upload, @st.cache_resource
    would keep returning that first vector store forever - uploading a
    *different* PDF silently kept answering from the old one. Here the cache
    key is `file_hashes`, a content hash per file, so a different file (or set
    of files) always invalidates the cache, while re-uploading an identical
    file is still a fast cache hit.
    """
    del file_hashes  # unused inside the function; it only exists to key the cache

    chunks: List[Document] = load_and_split_pdfs(list(file_paths))
    if not chunks:
        return None

    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# System prompt keeps the model grounded in retrieved context and instructs it
# to admit gaps rather than hallucinate - this is what the post-hoc
# "bad phrase" check downstream is looking for.
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's uploaded "
    "document(s). Use ONLY the retrieved context below to answer, and mention "
    "which document/page you are drawing from when relevant. If the answer is "
    "not contained in the context, say plainly that you couldn't find it in "
    "the document rather than guessing.\n\nContext:\n{context}"
)

CONDENSE_QUESTION_PROMPT = (
    "Given the conversation history and a follow-up question, rewrite the "
    "follow-up question as a standalone question that captures all necessary "
    "context from the history. Return only the rewritten question, nothing else."
)


def build_conversational_rag_chain(vectorstore: FAISS, llm):
    """Build a history-aware conversational RAG chain.

    This is the piece that gives the bot *real* memory over the documents: the
    retriever first reformulates the incoming question using chat history
    (e.g. "what about page 2?" -> "what does page 2 of report.pdf say?"), then
    retrieves against that reformulated query, then answers grounded in the
    retrieved chunks plus history. The original app only replayed old messages
    in the UI - it never fed them back into retrieval or generation, so
    follow-up questions like "and the second one?" wouldn't actually work.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        ChatPromptTemplate.from_messages(
            [
                ("system", CONDENSE_QUESTION_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        ),
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware_retriever, document_chain)


def is_low_quality_answer(answer: str) -> bool:
    """Heuristic fallback signal: if the grounded answer reads like a
    non-answer, prefer a general LLM response instead of showing it."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in config.BAD_PHRASES)
