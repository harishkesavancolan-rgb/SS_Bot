"""
ingest.py
---------
One-command orchestrator: PDF → chunks → embeddings → OpenSearch.

Usage
-----
    python ingest.py <pdf_path> <opensearch_host>

Example
-------
    python ingest.py story.pdf my-domain.us-east-1.es.amazonaws.com
"""

import sys
from chunker  import chunk_pdf
from embedder import embed_chunks
from store    import store_embeddings


def ingest(pdf_path: str, opensearch_host: str) -> None:
    print("=" * 60)
    print(f"  INGEST PIPELINE")
    print(f"  PDF  : {pdf_path}")
    print(f"  Host : {opensearch_host}")
    print("=" * 60)

    # 1️⃣  Chunk
    chunks = chunk_pdf(pdf_path)

    # 2️⃣  Embed
    records = embed_chunks(chunks)

    # 3️⃣  Store
    store_embeddings(records, host=opensearch_host)

    print("\n✅  Ingest complete!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <pdf_path> <opensearch_host>")
        sys.exit(1)

    ingest(sys.argv[1], sys.argv[2])