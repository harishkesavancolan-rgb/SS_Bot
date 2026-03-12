"""
ingestion/ingest.py
--------------------
Orchestrator — works in TWO modes:

  1. LOCAL mode (manual testing on your laptop):
       python -m ingestion.ingest <pdf_path>

  2. LAMBDA mode (triggered automatically by S3):
       AWS Lambda calls handler(event, context)
       event contains the S3 bucket + file key

S3 key format (NEW):
    user_id/session_id/filename.pdf

    This allows chunks to be tagged with both user_id and
    session_id so retrieval is fully isolated per chat session.
"""

import os
import sys
import json
import boto3
import tempfile

from ingestion.chunker  import chunk_pdf
from ingestion.embedder import embed_chunks
from ingestion.store    import store_embeddings


# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path   : str,
    doc_id     : str,
    user_id    : str = "default_user",
    session_id : str = "default_session",   # ← NEW
) -> None:
    """
    Runs the full ingestion pipeline on a PDF file.

    Chunks are tagged with both user_id and session_id so
    the retriever can scope searches to a specific chat session.
    """
    print(f"[ingest] Starting pipeline for: {doc_id} (user: {user_id} / session: {session_id})")

    # 1. Chunk
    chunks = chunk_pdf(pdf_path, doc_id=doc_id)

    # 2. Embed
    records = embed_chunks(chunks, region=AWS_REGION)

    # 3. Store into PostgreSQL with user_id + session_id
    store_embeddings(records, user_id=user_id, session_id=session_id)

    print(f"[ingest] ✅ Pipeline complete for: {doc_id}")


# ── Lambda Handler ────────────────────────────────────────────────────────────

def handler(event, context):
    """
    AWS Lambda entry point.

    S3 key format: user_id/session_id/filename.pdf

    We extract user_id and session_id from the key path,
    download the PDF, run the pipeline, then clean up.

    Handles legacy keys (user_id/filename.pdf) gracefully
    by falling back to 'default_session'.
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]

        from urllib.parse import unquote_plus
        key = unquote_plus(key)

        # Parse S3 key to extract user_id, session_id, doc_id
        # Expected format: user_id/session_id/filename.pdf
        parts = key.split("/")

        if len(parts) == 3:
            # Normal case: user_id/session_id/filename.pdf
            user_id    = parts[0]
            session_id = parts[1]
            doc_id     = parts[2].replace(".pdf", "")
        elif len(parts) == 2:
            # Legacy fallback: user_id/filename.pdf
            user_id    = parts[0]
            session_id = "default_session"
            doc_id     = parts[1].replace(".pdf", "")
        else:
            # No folder prefix at all
            user_id    = "default_user"
            session_id = "default_session"
            doc_id     = key.replace(".pdf", "")

        print(f"[ingest] S3 event: s3://{bucket}/{key} → user={user_id} session={session_id} doc={doc_id}")

        # Download PDF to Lambda's /tmp folder (max 512MB)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            print(f"[ingest] Downloading to {tmp_path}")
            s3_client.download_file(bucket, key, tmp_path)

        try:
            run_pipeline(
                pdf_path   = tmp_path,
                doc_id     = doc_id,
                user_id    = user_id,
                session_id = session_id,
            )
        finally:
            os.remove(tmp_path)
            print(f"[ingest] Cleaned up temp file")

    return {
        "statusCode": 200,
        "body": json.dumps("Ingestion complete")
    }


# ── Local mode ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run locally for testing:
      python -m ingestion.ingest <pdf_path> [user_id] [session_id]
    """
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.ingest <pdf_path> [user_id] [session_id]")
        sys.exit(1)

    pdf_path   = sys.argv[1]
    user_id    = sys.argv[2] if len(sys.argv) > 2 else "default_user"
    session_id = sys.argv[3] if len(sys.argv) > 3 else "default_session"

    doc_id = os.path.basename(pdf_path).replace(".pdf", "")
    run_pipeline(pdf_path=pdf_path, doc_id=doc_id, user_id=user_id, session_id=session_id)