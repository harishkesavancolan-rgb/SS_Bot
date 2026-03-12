"""
store.py
--------
Stores embedded chunks into PostgreSQL using the pgvector extension.

Table structure:
    chunks
    ├── id          (auto increment)
    ├── chunk_id    (unique string ID e.g. "story::chunk_0001")
    ├── doc_id      (which document this came from)
    ├── user_id     (which user owns this chunk)
    ├── session_id  (which chat session this chunk belongs to)
    ├── page_number (which page in the PDF)
    ├── text        (the actual chunk text)
    ├── embedding   (1024-dimensional vector from Titan v2)
    └── metadata    (JSON — source filename etc.)

session_id isolation:
    Each chunk is tagged with the session it was uploaded in.
    Retrieval filters by both user_id AND session_id so
    documents from one chat never appear in another.
"""

import os
import json
from typing import List

import psycopg2
from psycopg2.extras import execute_values


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1024


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
    """
    return psycopg2.connect(
        host     = os.environ.get("DB_HOST"),
        dbname   = os.environ.get("DB_NAME",     "ragdb"),
        user     = os.environ.get("DB_USER",     "postgres"),
        password = os.environ.get("DB_PASSWORD"),
        port     = int(os.environ.get("DB_PORT", "5432")),
        sslmode  = "require",
    )


# ── Table setup ───────────────────────────────────────────────────────────────

def ensure_table(conn) -> None:
    """
    Creates the chunks table and vector index if they don't exist.
    Safe to call every time — won't overwrite existing data.

    session_id column added so retrieval can be scoped
    per chat session, not just per user.
    """
    with conn.cursor() as cur:

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          SERIAL PRIMARY KEY,
                chunk_id    TEXT UNIQUE NOT NULL,
                doc_id      TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                session_id  TEXT NOT NULL DEFAULT 'default_session',
                page_number INTEGER,
                text        TEXT NOT NULL,
                embedding   vector({EMBEDDING_DIM}),
                metadata    JSONB
            );
        """)

        # Index for vector similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        # Index on (user_id, session_id) for fast WHERE filtering
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_user_session_idx
            ON chunks (user_id, session_id);
        """)

        conn.commit()
        print("[store] table and index ready ✓")


# ── Store embeddings ──────────────────────────────────────────────────────────

def store_embeddings(
    records    : List[dict],
    user_id    : str = "default_user",
    session_id : str = "default_session",   # ← NEW: ties chunks to a session
    batch_size : int = 50,
) -> None:
    """
    Inserts embedded chunk records into PostgreSQL.

    Each chunk is tagged with both user_id and session_id so
    retrieval can be scoped to exactly one chat session.

    Uses INSERT ... ON CONFLICT DO NOTHING so re-running
    on the same PDF never creates duplicates.

    Parameters
    ----------
    records    : output from embedder.embed_chunks()
    user_id    : owner of these chunks
    session_id : chat session these chunks belong to
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
                        user_id,
                        session_id,           # ← stored per chunk
                    )
                    for record in batch
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, page_number, text, embedding, metadata, user_id, session_id)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    rows,
                )

                success += len(batch)
                print(f"[store] inserted {success}/{total} chunks")

            conn.commit()
            print(f"[store] ✅ done — {success} chunks stored for user '{user_id}' / session '{session_id}'")

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