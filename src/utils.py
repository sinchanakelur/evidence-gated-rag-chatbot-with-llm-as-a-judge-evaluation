"""Small filesystem helpers for handling uploaded PDFs."""
from __future__ import annotations

import os
from typing import List, Tuple

from . import config
from .document_loader import file_hash


def persist_uploaded_files(uploaded_files) -> Tuple[List[str], List[str]]:
    """Save uploaded Streamlit files to disk, keyed by content hash.

    Re-uploading an identical file reuses the copy already on disk instead of
    rewriting it, and different files never collide on the same path (the
    original app always wrote to a single "temp.pdf", which is also why its
    vector-store cache silently went stale after the first upload).

    Returns (file_paths, file_hashes), in the same order as uploaded_files.
    """
    paths, hashes = [], []
    for f in uploaded_files:
        data = f.getvalue()
        h = file_hash(data)
        path = os.path.join(config.UPLOAD_DIR, f"{h}_{f.name}")
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(data)
        paths.append(path)
        hashes.append(h)
    return paths, hashes
