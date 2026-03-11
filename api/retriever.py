"""
api/retriever.py
----------------
Retrieves relevant chunks for a user question using:
  1. pgvector similarity search  (fast, approximate)
  2. Cohere Rerank               (slow, accurate)

Why two steps?
  Vector search is fast but imprecise — returns 20 candidates.
  Reranking is slower but reads the question carefully
  and picks the truly relevant chunks from those 20.
  Together they give fast AND accurate retrieval.
"""

import os
import json
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_DIM        = 1024
VECTOR_SEARCH_TOP_K  = 5     # fetch top 5 directly, no reranking
RERANK_TOP_N         = 5     # how many to keep after reranking
COHERE_MODEL_ID      = "cohere.rerank-v3-5:0"
TITAN_MODEL_ID       = "amazon.titan-embed-text-v2:0"
AWS_REGION           = os.environ.get("AWS_REGION", "us-east-1")


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

def vector_search(
    question_vector : List[float],
    user_id         : str,
    top_k           : int = VECTOR_SEARCH_TOP_K,
) -> List[Dict]:
    """
    Searches pgvector for the most similar chunks to the question.

    Uses cosine distance (<=>operator) to find nearest neighbours.
    Filters by user_id so users only see their own PDFs.

    Returns a list of chunks with similarity scores.
    """
    conn = _get_connection()

    try:
        # RealDictCursor returns rows as dicts instead of tuples
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
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                question_vector,
                user_id,
                question_vector,
                top_k,
            ))

            results = [dict(row) for row in cur.fetchall()]

    finally:
        conn.close()

    print(f"[retriever] vector search returned {len(results)} candidates")
    return results


# ── Step 3: Rerank with Cohere ────────────────────────────────────────────────

def rerank(
    question : str,
    chunks   : List[Dict],
    top_n    : int = RERANK_TOP_N,
) -> List[Dict]:
    """
    Reranks chunks using Cohere Rerank 3.5 via bedrock-agent-runtime.

    Uses the dedicated rerank() API — NOT invoke_model().
    Cohere Rerank requires bedrock-agent-runtime client.
    """
    if not chunks:
        return []

    # Must use bedrock-agent-runtime NOT bedrock-runtime for reranking
    client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

    try:
        response = client.rerank(
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{COHERE_MODEL_ID}"
                    },
                    "numberOfResults": top_n,
                }
            },
            sources=[
                {
                    "type"              : "INLINE",
                    "inlineDocumentSource": {
                        "type"         : "TEXT",
                        "textDocument" : {"text": chunk["text"]},
                    }
                }
                for chunk in chunks
            ],
            queries=[
                {"type": "TEXT", "textQuery": {"text": question}}
            ],
        )
    except Exception as e:
        print(f"[retriever] ❌ rerank error: {type(e).__name__}: {str(e)}")
        raise

    # Map rerank scores back to original chunks
    reranked_chunks = []
    for result in response["results"]:
        chunk = chunks[result["index"]].copy()
        chunk["rerank_score"]     = round(result["relevanceScore"], 4)
        chunk["similarity_score"] = round(chunk["similarity_score"], 4)
        reranked_chunks.append(chunk)

    print(f"[retriever] reranking kept top {len(reranked_chunks)} chunks")
    return reranked_chunks


# ── Step 4: Build source hyperlinks ──────────────────────────────────────────

def build_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Builds source reference objects for each chunk.

    These are displayed as hyperlinks in the chat UI:
        [ArtOfWar.pdf | Page 3 | Score: 0.89]

    Clicking the hyperlink shows the EXACT chunk text
    that was used to generate the answer — not the whole PDF.

    The frontend uses 'chunk_id' to fetch and display
    the specific chunk text in a popup/side panel.
    """
    sources = []

    for chunk in chunks:
        metadata  = chunk.get("metadata", {})
        pdf_title = metadata.get("source", chunk["doc_id"])
        page_num  = chunk.get("page_number", "?")
        score     = round(chunk.get("similarity_score", 0), 4)

        sources.append({
            "chunk_id"   : chunk["chunk_id"],    # unique ID for this chunk
            "pdf_title"  : pdf_title,
            "page_number": page_num,
            "score"      : score,
            "text"       : chunk["text"],        # ← the exact chunk text
            "display"    : f"{pdf_title} | Page {page_num} | Score: {score}",
            # Frontend generates the hyperlink using chunk_id
            # Clicking it displays chunk["text"] in a popup
        })

    return sources


# ── Public API ────────────────────────────────────────────────────────────────

async def retrieve(question: str, user_id: str) -> Dict:
    """
    Retrieval pipeline:
      1. Embed question
      2. Vector search → top 5 chunks
      3. Build sources

    Reranking skipped — vector search top 5 is accurate
    enough for personal scale use.
    """
    # 1. Embed
    question_vector = embed_question(question)

    # 2. Vector search — returns top 5 directly
    chunks = vector_search(question_vector, user_id)

    if not chunks:
        return {"chunks": [], "sources": []}

    # 3. Build sources
    sources = build_sources(chunks)

    return {
        "chunks" : chunks,
        "sources": sources,
    }