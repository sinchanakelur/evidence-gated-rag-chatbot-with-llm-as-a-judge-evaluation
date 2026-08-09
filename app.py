# -------- IMPORTS --------
import logging
import warnings

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.llm import get_llm
from src.rag_engine import (
    build_conversational_rag_chain,
    build_vectorstore,
    is_low_quality_answer,
)
from src.utils import persist_uploaded_files

# -------- SETTINGS --------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# -------- UI --------
st.title("Smart RAG Chatbot")
st.caption("Chat with your PDFs or ask anything")

# -------- SIDEBAR --------
st.sidebar.header("Controls")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

# -------- CHAT MEMORY (display) --------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} — page {s['page']}**")
                    st.write(s["snippet"])

# -------- VECTOR STORE --------
vectorstore = None
if uploaded_files:
    try:
        file_paths, file_hashes = persist_uploaded_files(uploaded_files)
        vectorstore = build_vectorstore(tuple(file_hashes), tuple(file_paths))
        if vectorstore is None:
            st.sidebar.warning(
                "No usable text could be extracted from the uploaded PDF(s)."
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        logging.exception("Failed to index uploaded PDFs")
        st.sidebar.error(f"Failed to process uploaded PDF(s): {exc}")

# -------- INPUT --------
prompt = st.chat_input("Ask something...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Convert prior turns into LangChain message objects so the chain can
    # actually use them (previously, history was only ever re-rendered in the
    # UI — it was never given back to the retriever or the LLM).
    chat_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    response_text = ""
    sources_for_display = []

    try:
        llm = get_llm()

        with st.spinner("Thinking..."):
            # -------- IF PDF(S) EXIST --------
            if vectorstore is not None:
                rag_chain = build_conversational_rag_chain(vectorstore, llm)
                result = rag_chain.invoke(
                    {"input": prompt, "chat_history": chat_history}
                )

                answer = result.get("answer", "")
                if not isinstance(answer, str):
                    answer = str(answer)
                answer = answer.replace("Ɵ", "ti").replace("ﬁ", "fi")

                retrieved_docs = result.get("context", [])

                # -------- SMART FILTER --------
                use_pdf_answer = bool(retrieved_docs) and not is_low_quality_answer(
                    answer
                )

                if use_pdf_answer:
                    response_text = answer

                    seen = set()
                    for doc in retrieved_docs[:3]:
                        key = (doc.metadata.get("source"), doc.metadata.get("page"))
                        if key in seen:
                            continue
                        seen.add(key)
                        sources_for_display.append(
                            {
                                "source": doc.metadata.get("source", "unknown"),
                                "page": doc.metadata.get("page", "?"),
                                "snippet": doc.page_content[:300] + "...",
                            }
                        )
                else:
                    response_text = llm.invoke(prompt).content

            else:
                # -------- NORMAL CHAT --------
                response_text = llm.invoke(prompt).content

    except Exception as exc:  # noqa: BLE001 - shown to user, and logged
        logging.exception("Chat turn failed")
        response_text = f"Error: {exc}"

    # -------- DISPLAY --------
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if sources_for_display:
            with st.expander("Sources"):
                for s in sources_for_display:
                    st.markdown(f"**{s['source']} — page {s['page']}**")
                    st.write(s["snippet"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": sources_for_display,
        }
    )
