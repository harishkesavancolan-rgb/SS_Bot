"""
embedder.py
-----------
Calls AWS Bedrock Titan Embed v2 to turn text chunks into vectors.
"""

import json
import time
import boto3
from typing import List

from ingestion.chunker import Chunk


# ── Config ────────────────────────────────────────────────────────────────────

TITAN_MODEL_ID  = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM   = 1024       # Titan v2 native dimension
BATCH_DELAY_SEC = 0.05       # small pause between API calls to avoid throttling


# ── Client ────────────────────────────────────────────────────────────────────

def _get_bedrock_client(region: str = "us-east-1"):
    """
    Returns a Bedrock runtime client.
    Credentials are resolved automatically via the standard boto3 chain:
      1. Environment variables  (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
      2. ~/.aws/credentials
      3. IAM instance/task role  ← recommended in production
    """
    return boto3.client("bedrock-runtime", region_name=region)


# ── Core embedding call ───────────────────────────────────────────────────────

def embed_text(text: str, client) -> List[float]:
    """
    Send a single string to Titan Embed v2 and return its embedding vector.
    """
    body = json.dumps({
        "inputText": text,
        # Optional Titan v2 parameters:
        # "dimensions": 512,       # 256 | 512 | 1024  (default 1024)
        # "normalize": True,
    })

    response = client.invoke_model(
        modelId     = TITAN_MODEL_ID,
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


# ── Batch helper ──────────────────────────────────────────────────────────────

def embed_chunks(
    chunks : List[Chunk],
    region : str = "us-east-1",
    delay  : float = BATCH_DELAY_SEC,
) -> List[dict]:
    """
    Embed every Chunk and return a list of enriched dicts ready for storage.

    Each dict contains all chunk fields plus an 'embedding' key.
    """
    client = _get_bedrock_client(region)
    results = []

    for i, chunk in enumerate(chunks):
        print(f"[embedder] embedding {i + 1}/{len(chunks)}  ({chunk.chunk_id})")

        vector = embed_text(chunk.text, client)

        results.append({
            "chunk_id"   : chunk.chunk_id,
            "doc_id"     : chunk.doc_id,
            "page_number": chunk.page_number,
            "text"       : chunk.text,
            "embedding"  : vector,
            "metadata"   : chunk.metadata,
        })

        if delay:
            time.sleep(delay)

    print(f"[embedder] done — {len(results)} vectors (dim={EMBEDDING_DIM})")
    return results


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from chunker import chunk_pdf

    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    chunks  = chunk_pdf(pdf_file)
    records = embed_chunks(chunks[:3])     # embed only first 3 for the test

    for r in records:
        print(f"{r['chunk_id']}  →  vector[0:5] = {r['embedding'][:5]}")