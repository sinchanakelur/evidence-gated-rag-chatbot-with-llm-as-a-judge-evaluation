# -------- IMPORTS --------
import logging
import warnings

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.llm import get_llm
from src.rag_engine import answer_question, build_indexes, get_reranker
from src.utils import persist_uploaded_files

# -------- SETTINGS --------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# -------- UI --------
st.title("Smart RAG Chatbot")
st.caption("Chat with your PDFs — answers are grounded in the documents, or refused")

# -------- SIDEBAR --------
st.sidebar.header("Controls")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

with st.sidebar.expander("How answers are decided"):
    st.markdown(
        "1. Retrieve candidates via **vector search + BM25** (hybrid).\n"
        "2. **Rerank** candidates with a cross-encoder.\n"
        "3. If nothing clears the relevance threshold, **refuse** — no "
        "general-knowledge fallback is used.\n"
        "4. Otherwise generate an answer grounded only in the surviving "
        "chunks, shown under *Sources*."
    )

# -------- CHAT MEMORY (display) --------
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"**{s['source']} — page {s['page']}** "
                f"(rerank score: {s['rerank_score']:.3f})"
            )
            st.write(s["snippet"])


def _render_timings(timings):
    st.caption(
        f"retrieval {timings.get('retrieval_ms', 0):.0f} ms · "
        f"rerank {timings.get('rerank_ms', 0):.0f} ms · "
        f"generation {timings.get('generation_ms', 0):.0f} ms · "
        f"total {timings.get('total_ms', 0):.0f} ms"
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
        if msg.get("timings"):
            _render_timings(msg["timings"])

# -------- REQUIRE DOCUMENTS --------
# No general-chat fallback: this app only answers from uploaded documents.
if not uploaded_files:
    st.info("Upload one or more PDFs in the sidebar to start asking questions.")
    st.stop()

# -------- INDEXES --------
vectorstore = bm25_retriever = None
try:
    file_paths, file_hashes = persist_uploaded_files(uploaded_files)
    vectorstore, bm25_retriever, _chunks = build_indexes(
        tuple(file_hashes), tuple(file_paths)
    )
    if vectorstore is None:
        st.sidebar.warning(
            "No usable text could be extracted from the uploaded PDF(s)."
        )
except Exception as exc:  # noqa: BLE001 - surfaced to the user
    logging.exception("Failed to index uploaded PDFs")
    st.sidebar.error(f"Failed to process uploaded PDF(s): {exc}")

if vectorstore is None:
    st.stop()

reranker = get_reranker()

# -------- INPUT --------
prompt = st.chat_input("Ask something about your document(s)...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prior turns as LangChain message objects so condense_question can
    # actually use them to rewrite follow-ups.
    chat_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    response_text = ""
    sources_for_display = []
    timings = {}

    try:
        llm = get_llm()
        with st.spinner("Retrieving, reranking, and generating..."):
            result = answer_question(
                llm=llm,
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                reranker=reranker,
                question=prompt,
                chat_history=chat_history,
            )

        response_text = result["answer"]
        timings = result["timings"]

        seen = set()
        for doc in result["sources"]:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources_for_display.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "rerank_score": doc.metadata.get("rerank_score", 0.0),
                    "snippet": doc.page_content[:300] + "...",
                }
            )

    except Exception as exc:  # noqa: BLE001 - shown to user, and logged
        logging.exception("Chat turn failed")
        response_text = f"Error: {exc}"

    # -------- DISPLAY --------
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if sources_for_display:
            _render_sources(sources_for_display)
        if timings:
            _render_timings(timings)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": sources_for_display,
            "timings": timings,
        }
    )
