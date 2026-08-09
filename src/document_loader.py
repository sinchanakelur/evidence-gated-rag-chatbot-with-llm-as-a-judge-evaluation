"""PDF loading, text cleaning and chunking utilities.

Separated from the vector-store logic so it can be unit tested (and reused)
independently of Streamlit / FAISS / embeddings.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

logger = logging.getLogger(__name__)


def file_hash(file_bytes: bytes) -> str:
    """Stable content hash, used as a cache key for a single uploaded file."""
    return hashlib.md5(file_bytes).hexdigest()


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\x00", "")
    return " ".join(text.split())  # collapse repeated whitespace


def load_and_split_pdfs(file_paths: List[str]) -> List[Document]:
    """Load one or more PDFs, clean the text and split into retrieval-sized chunks.

    Every chunk carries 'source' (original filename) and 'page' (1-indexed)
    metadata, which is what lets the UI show real page-wise citations instead
    of just a raw text snippet.

    A PDF that fails to load (corrupt file, scanned image-only PDF, etc.) is
    skipped with a warning rather than crashing the whole app, and the rest of
    the batch still gets indexed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_chunks: List[Document] = []

    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            documents = loader.load()
        except Exception as exc:  # noqa: BLE001 - intentionally broad, logged below
            logger.warning("Failed to load %s: %s", path, exc)
            continue

        if not documents:
            logger.warning("No extractable text in %s (scanned/image-only PDF?)", path)
            continue

        filename = Path(path).name
        for doc in documents:
            doc.metadata["source"] = filename
            # PyPDFLoader's 'page' is 0-indexed; make it human-readable.
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1

        try:
            chunks = splitter.split_documents(documents)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to split %s: %s", path, exc)
            continue

        for chunk in chunks:
            text = chunk.page_content
            if not text or not isinstance(text, str):
                continue
            cleaned = _clean_text(text)
            if len(cleaned) < config.MIN_CHUNK_LENGTH:
                continue
            chunk.page_content = cleaned
            all_chunks.append(chunk)

    return all_chunks
