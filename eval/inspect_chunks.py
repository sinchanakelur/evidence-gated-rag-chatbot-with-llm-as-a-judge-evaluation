"""Dump every chunk (id, source, page, text preview) for a folder of PDFs.

Use this to build eval/golden_set.json by hand: run it against the PDFs you
plan to evaluate with, read through the chunks, and copy the `chunk_id`
values that actually answer each question you write into
`relevant_chunks`.

Usage:
    python eval/inspect_chunks.py --pdf-dir /path/to/pdfs
    python eval/inspect_chunks.py --pdf-dir /path/to/pdfs --grep "revenue"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.document_loader import load_and_split_pdfs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True, help="Folder of PDFs to inspect")
    parser.add_argument(
        "--grep", default=None, help="Only show chunks containing this substring (case-insensitive)"
    )
    args = parser.parse_args()

    pdf_paths = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {args.pdf_dir}")
        return

    chunks = load_and_split_pdfs(pdf_paths)
    print(f"{len(chunks)} chunks across {len(pdf_paths)} file(s)\n")

    needle = args.grep.lower() if args.grep else None
    for chunk in chunks:
        if needle and needle not in chunk.page_content.lower():
            continue
        preview = chunk.page_content[:200].replace("\n", " ")
        print(f"[{chunk.metadata['chunk_id']}]")
        print(f"  {preview}...")
        print()


if __name__ == "__main__":
    main()
