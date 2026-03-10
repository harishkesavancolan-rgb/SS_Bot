"""
store.py
--------
Stores embedded chunks into PostgreSQL using the pgvector extension.

pgvector turns PostgreSQL into a vector database —
allowing us to search chunks by semantic similarity.

Table structure:
    chunks
    ├── id          (auto increment)
    ├── chunk_id    (unique string ID e.g. "story::chunk_0001")
    ├── doc_id      (which document this came from)
    ├── page_number (which page in the PDF)
    ├── text        (the actual chunk text)
    ├── embedding   (1024-dimensional vector from Titan v2)
    └── metadata    (JSON — source filename etc.)
"""

import os
import json
from typing import List

import psycopg2
from psycopg2.extras import execute_values


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1024     # must match Titan v2 output dimension


# ── Database connection ───────────────────────────────────────────────────────

def _get_connection():
    """
    Creates a PostgreSQL connection using environment variables.

    Environment variables needed:
        DB_HOST      → your RDS endpoint
        DB_NAME      → ragdb
        DB_USER      → postgres
        DB_PASSWORD  → your password
        DB_PORT      → 5432 (default)

    Never hardcode these values — always use environment variables!
    Set them locally:
        Windows: set DB_HOST=rag-db.xxxxxxxxxx.us-east-1.rds.amazonaws.com
        Mac/Linux: export DB_HOST=rag-db.xxxxxxxxxx.us-east-1.rds.amazonaws.com
    In Lambda, set them as environment variables in the AWS Console.
    """
    return psycopg2.connect(
        host     = os.environ.get("DB_HOST"),
        dbname   = os.environ.get("DB_NAME",     "ragdb"),
        user     = os.environ.get("DB_USER",     "postgres"),
        password = os.environ.get("DB_PASSWORD"),
        port     = int(os.environ.get("DB_PORT", "5432")),
        sslmode  = "require",    # always use SSL with RDS
    )


# ── Table setup ───────────────────────────────────────────────────────────────

def ensure_table(conn) -> None:
    """
    Creates the chunks table and vector index if they don't exist.
    Safe to call every time — won't overwrite existing data.

    We create:
        1. The chunks table to store text + vectors
        2. An ivfflat index for fast vector similarity search
    """
    with conn.cursor() as cur:

        # Enable pgvector extension (safe to run even if already enabled)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create the chunks table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          SERIAL PRIMARY KEY,
                chunk_id    TEXT UNIQUE NOT NULL,
                doc_id      TEXT NOT NULL,
                page_number INTEGER,
                text        TEXT NOT NULL,
                embedding   vector({EMBEDDING_DIM}),
                metadata    JSONB
            );
        """)

        # Create vector similarity search index
        # ivfflat = fast approximate nearest neighbour search
        # lists=100 is a good default for small-medium datasets
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        conn.commit()
        print("[store] table and index ready ✓")


# ── Store embeddings ──────────────────────────────────────────────────────────

def store_embeddings(
    records    : List[dict],
    user_id    : str = "default_user",
    batch_size : int = 50,
) -> None:
    """
    Inserts embedded chunk records into PostgreSQL.

    Uses INSERT ... ON CONFLICT DO NOTHING so re-running
    on the same PDF never creates duplicates.

    Parameters
    ----------
    records    : output from embedder.embed_chunks()
    user_id    : owner of these chunks (isolates per user)
    batch_size : number of rows per insert call
    """
    if not records:
        print("[store] no records to store")
        return

    conn = _get_connection()

    try:
        ensure_table(conn)

        with conn.cursor() as cur:
            total   = len(records)
            success = 0

            for i in range(0, total, batch_size):
                batch = records[i:i + batch_size]

                rows = [
                    (
                        record["chunk_id"],
                        record["doc_id"],
                        record["page_number"],
                        record["text"],
                        record["embedding"],
                        json.dumps(record.get("metadata", {})),
                        user_id,              # ← save user_id per chunk
                    )
                    for record in batch
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, page_number, text, embedding, metadata, user_id)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    rows,
                )

                success += len(batch)
                print(f"[store] inserted {success}/{total} chunks")

            conn.commit()
            print(f"[store] ✅ done — {success} chunks stored for user '{user_id}'")

    finally:
        conn.close()


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