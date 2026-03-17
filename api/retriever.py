"""
api/retriever.py
----------------
Retrieves relevant chunks for a user question using:
  1. pgvector similarity search  (fast, approximate)
  2. Offline cross-encoder rerank (no API call, runs locally)

Why two steps?
  Vector search is fast but imprecise — returns top K candidates.
  Reranking is slower but reads the question carefully
  and picks the truly relevant chunks from those candidates.
  Together they give fast AND accurate retrieval.

Session isolation:
  All retrieval is scoped to a specific session_id so
  documents uploaded in one chat never bleed into another.
"""

import os
os.environ["HF_HOME"] = "/app/hf_cache"

# Must be set BEFORE importing sentence_transformers
# HuggingFace reads this env var at import time to decide where to cache models
# /app/hf_cache is baked into the Docker image at build time — no download on cold start


import json
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict
from sentence_transformers import CrossEncoder


# ── Config ────────────────────────────────────────────────────────────────────

MIN_SIMILARITY_SCORE = 0.2
EMBEDDING_DIM        = 1024
VECTOR_SEARCH_TOP_K  = 5
RERANK_TOP_N         = 5
TITAN_MODEL_ID       = "amazon.titan-embed-text-v2:0"
AWS_REGION           = os.environ.get("AWS_REGION", "us-east-1")

# Loaded once at container startup — reused across all invocations
# Model lives at /app/hf_cache inside the Docker image (baked in at build time)
_RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ── Database connection ───────────────────────────────────────────────────────

def _get_connection():
    """Connect to RDS PostgreSQL using environment variables."""
    return psycopg2.connect(
        host     = os.environ.get("DB_HOST"),
        dbname   = os.environ.get("DB_NAME",    "ragdb"),
        user     = os.environ.get("DB_USER",    "postgres"),
        password = os.environ.get("DB_PASSWORD"),
        port     = int(os.environ.get("DB_PORT", "5432")),
        sslmode  = "require",
    )


# ── Step 1: Embed the question ────────────────────────────────────────────────

def embed_question(question: str) -> List[float]:
    """
    Converts the user's question into a 1024-dim vector
    using the same Titan model we used for chunks.

    Why same model?
    Because the question and chunks must live in the
    same vector space to be comparable.
    """
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    body = json.dumps({"inputText": question})
    response = client.invoke_model(
        modelId     = TITAN_MODEL_ID,
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )

    return json.loads(response["body"].read())["embedding"]


# ── Step 2: Vector search in pgvector ────────────────────────────────────────

def vector_search(question_vector, user_id, session_id, top_k=VECTOR_SEARCH_TOP_K):
    """
    Searches pgvector for the most similar chunks to the question vector.
    Scoped to user_id + session_id so sessions are fully isolated.
    Filters out chunks below MIN_SIMILARITY_SCORE threshold.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    chunk_id,
                    doc_id,
                    page_number,
                    text,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM chunks
                WHERE user_id = %s AND session_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (question_vector, user_id, session_id, question_vector, top_k))

            results = [dict(row) for row in cur.fetchall()]

    finally:
        conn.close()

    # Filter out chunks that are not relevant enough
    results = [r for r in results if r["similarity_score"] >= MIN_SIMILARITY_SCORE]

    print(f"[retriever] vector search returned {len(results)} candidates")
    return results


# ── Step 3: Offline rerank ────────────────────────────────────────────────────

def rerank(question: str, chunks: List[Dict], top_n: int = RERANK_TOP_N) -> List[Dict]:
    """
    Reranks chunks using a local cross-encoder model.
    No API call — runs entirely inside the Lambda container.

    Cross-encoder reads the question and each chunk together as a pair
    and outputs a relevance score — more accurate than vector similarity alone.

    If fewer than top_n chunks are passed in, all of them are returned.
    Python list slicing handles this safely — no error thrown.
    """
    if not chunks:
        return []

    # Pair question with each chunk text
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # Score all pairs in one pass — runs on CPU inside Lambda
    scores = _RERANKER.predict(pairs)

    # Attach rerank score to each chunk
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    # Sort by score descending, keep top_n
    # If len(chunks) < top_n, slice safely returns all chunks
    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    reranked = reranked[:top_n]

    print(f"[retriever] reranking kept top {len(reranked)} chunks")
    return reranked


# ── Step 4: Build source hyperlinks ──────────────────────────────────────────

def build_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Builds source reference objects for each chunk.

    These are displayed as hyperlinks in the chat UI:
        [ArtOfWar.pdf | Page 3 | Score: 0.89]

    Clicking the hyperlink shows the exact chunk text
    used to generate the answer — not the whole PDF.

    Score prefers rerank_score over similarity_score since
    rerank is more accurate.
    """
    sources = []

    for chunk in chunks:
        metadata  = chunk.get("metadata", {})
        pdf_title = metadata.get("source", chunk["doc_id"])
        page_num  = chunk.get("page_number", "?")
        score     = round(chunk.get("rerank_score", chunk.get("similarity_score", 0)), 4)

        sources.append({
            "chunk_id"   : chunk["chunk_id"],
            "pdf_title"  : pdf_title,
            "page_number": page_num,
            "score"      : score,
            "text"       : chunk["text"],
            "display"    : f"{pdf_title} | Page {page_num} | Score: {score}",
        })

    return sources


# ── Public API ────────────────────────────────────────────────────────────────

async def retrieve(question: str, user_id: str, session_id: str) -> Dict:
    """
    Full retrieval pipeline:
      1. Embed question → 1024-dim vector
      2. Vector search  → top K chunks scoped to user + session
      3. Rerank         → scored and sorted by cross-encoder
      4. Build sources  → display objects for the frontend

    If rerank fails for any reason, falls back to vector search order.
    If no chunks found, returns empty lists — LLM handles casual conversation.
    """
    # 1. Embed
    question_vector = embed_question(question)

    # 2. Vector search — scoped to user_id + session_id
    chunks = vector_search(question_vector, user_id, session_id)

    if not chunks:
        return {"chunks": [], "sources": []}

    # 3. Rerank — falls back to vector search order if model fails
    try:
        chunks = rerank(question, chunks)
    except Exception as e:
        print(f"[retriever] Rerank failed, using vector search order: {e}")

    # 4. Build sources
    sources = build_sources(chunks)

    return {
        "chunks" : chunks,
        "sources": sources,
    }