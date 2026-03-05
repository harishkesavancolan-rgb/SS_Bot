"""
store.py
--------
Stores embedded chunks into Pinecone vector database.

Pinecone replaces OpenSearch as our vector store.
Same interface — just a different backend.
"""

import os
from typing import List
from pinecone import Pinecone, ServerlessSpec


# ── Config ────────────────────────────────────────────────────────────────────

INDEX_NAME    = "rag-vectors"
EMBEDDING_DIM = 1024          # must match Titan v2 output dimension


# ── Client factory ────────────────────────────────────────────────────────────

def _get_pinecone_client() -> Pinecone:
    """
    Creates a Pinecone client using the API key from environment variables.

    Never hardcode your API key in code — always use environment variables!
    Set it locally like this:
        Windows: set PINECONE_API_KEY=pcsk-xxxxxxxx
        Mac/Linux: export PINECONE_API_KEY=pcsk-xxxxxxxx
    In Lambda we set it as an environment variable in the AWS Console.
    """
    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY environment variable is not set. "
            "Get your key from app.pinecone.io → API Keys"
        )

    return Pinecone(api_key=api_key)


# ── Index management ──────────────────────────────────────────────────────────

def ensure_index(pc: Pinecone, index_name: str = INDEX_NAME) -> None:
    """
    Creates the Pinecone index if it doesn't already exist.
    Safe to call every time — won't overwrite existing data.
    """
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name in existing_indexes:
        print(f"[store] index '{index_name}' already exists — skipping creation")
        return

    print(f"[store] creating index '{index_name}'...")
    pc.create_index(
        name      = index_name,
        dimension = EMBEDDING_DIM,
        metric    = "cosine",
        spec      = ServerlessSpec(
            cloud  = "aws",
            region = "us-east-1",
        )
    )
    print(f"[store] index '{index_name}' created ✓")


# ── Store embeddings ──────────────────────────────────────────────────────────

def store_embeddings(
    records    : List[dict],
    index_name : str = INDEX_NAME,
    batch_size : int = 50,
) -> None:
    """
    Upserts embedded chunk records into Pinecone.

    Each record must have:
        chunk_id  : unique ID string
        embedding : list of 1024 floats
        text      : original chunk text
        doc_id    : source document ID
        page_number: page number in the PDF
        metadata  : dict of additional info

    Parameters
    ----------
    records    : output from embedder.embed_chunks()
    index_name : Pinecone index to store into
    batch_size : number of vectors per upsert call
    """
    pc    = _get_pinecone_client()
    ensure_index(pc, index_name)
    index = pc.Index(index_name)

    # Build Pinecone upsert format
    # Each vector = (id, embedding, metadata)
    # We store the text in metadata so we can retrieve it at query time
    vectors = []
    for record in records:
        vectors.append({
            "id"      : record["chunk_id"],
            "values"  : record["embedding"],
            "metadata": {
                "text"       : record["text"],
                "doc_id"     : record["doc_id"],
                "page_number": record["page_number"],
                "source"     : record.get("metadata", {}).get("source", ""),
            }
        })

    # Upsert in batches
    # Pinecone recommends batches of 50-100 vectors at a time
    total   = len(vectors)
    success = 0

    for i in range(0, total, batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        success += len(batch)
        print(f"[store] upserted {success}/{total} vectors")

    print(f"[store] ✅ done — {success} vectors stored in '{index_name}'")


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ingestion.chunker  import chunk_pdf
    from ingestion.embedder import embed_chunks

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.store <pdf_path>")
        sys.exit(1)

    chunks  = chunk_pdf(sys.argv[1])
    records = embed_chunks(chunks)
    store_embeddings(records)