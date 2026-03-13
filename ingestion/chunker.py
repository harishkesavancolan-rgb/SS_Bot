"""
chunker.py
----------
Loads a PDF and splits it into overlapping text chunks using
LangChain's RecursiveCharacterTextSplitter.

Split priority order:
  1. Paragraph  \n\n   (most natural)
  2. Newline    \n
  3. Sentence   .
  4. Word space
  5. Character  (last resort)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import pdfplumber                                           # pip install pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pip install langchain-text-splitters


# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 1000  # max characters per chunk
CHUNK_OVERLAP = 200   # characters shared between consecutive chunks


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id   : str
    doc_id     : str
    page_number: int
    text       : str
    metadata   : dict = field(default_factory=dict)


# ── Core splitting logic ──────────────────────────────────────────────────────

def _get_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.

    It will try to split on paragraphs first, then newlines,
    then sentences, then words, then characters — whichever
    produces chunks under chunk_size first.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        separators    = ["\n\n", "\n", ". ", " ", ""],  # priority order
    )


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_pdf(
    pdf_path  : str,
    doc_id    : str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap   : int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Extract text from every page of *pdf_path*, split it using
    RecursiveCharacterTextSplitter, and return a list of Chunk objects.

    Parameters
    ----------
    pdf_path   : path to the PDF file
    doc_id     : logical document identifier (defaults to the file stem)
    chunk_size : maximum characters per chunk
    overlap    : characters shared between consecutive chunks
    """
    path     = Path(pdf_path)
    doc_id   = doc_id or path.stem
    splitter = _get_splitter(chunk_size, overlap)

    chunks: List[Chunk] = []
    chunk_index = 0

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            raw_text = raw_text.strip()

            if not raw_text:
                continue

            # RecursiveCharacterTextSplitter works on plain strings
            fragments = splitter.split_text(raw_text)

            for fragment in fragments:
                fragment = fragment.strip()
                if not fragment:
                    continue

                chunks.append(
                    Chunk(
                        chunk_id    = f"{doc_id}::chunk_{chunk_index:04d}",
                        doc_id      = doc_id,
                        page_number = page_num,
                        text        = fragment,
                        metadata    = {
                            "source"     : f"{doc_id}.pdf",
                            "page_number": page_num,
                        },
                    )
                )
                chunk_index += 1

    print(f"[chunker] '{path.name}' → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")
    return chunks


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    result   = chunk_pdf(pdf_file)

    for c in result[:3]:
        print(json.dumps({
            "chunk_id"   : c.chunk_id,
            "page_number": c.page_number,
            "preview"    : c.text[:120] + "…",
        }, indent=2))