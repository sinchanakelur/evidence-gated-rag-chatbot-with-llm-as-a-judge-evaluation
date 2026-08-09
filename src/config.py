"""Centralized configuration for the RAG chatbot.

Keeping these as named constants (instead of magic numbers scattered through
app.py) is what makes it possible to tune retrieval quality later without
hunting through the UI code.
"""
import os

# ---- Embedding & LLM settings ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# ---- Chunking ----
# Bumped from 400/50 -> 800/120. 400-char chunks are quite small for
# MiniLM + an 8B model: they fragment sentences and hurt recall. 800/120
# keeps chunks coherent while still fitting comfortably in context.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_LENGTH = 30  # discard near-empty chunks (headers, page numbers, etc.)

# ---- Retrieval ----
TOP_K = 4

# ---- Fallback detection ----
# Used as a last-resort signal that the PDF-grounded answer was a non-answer,
# so we can fall back to a general LLM response instead of showing "I don't know".
BAD_PHRASES = [
    "don't know",
    "do not know",
    "not mentioned",
    "not provided",
    "no information",
    "cannot find",
    "i don't have access",
]

# ---- Storage ----
# Uploaded PDFs are cached on disk by content hash so re-uploading the same
# file doesn't rewrite it, and switching files doesn't collide on one path.
UPLOAD_DIR = os.path.join(os.getcwd(), ".uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
