"""
ingestion/ingest.py
--------------------
Orchestrator — works in TWO modes:

  1. LOCAL mode (manual testing on your laptop):
       python -m ingestion.ingest <pdf_path> <opensearch_host>

  2. LAMBDA mode (triggered automatically by S3):
       AWS Lambda calls handler(event, context)
       event contains the S3 bucket + file key
"""

import os
import sys
import json
import boto3
import tempfile

from ingestion.chunker  import chunk_pdf
from ingestion.embedder import embed_chunks
from ingestion.store    import store_embeddings


# ── Config (read from environment variables set in Lambda) ────────────────────
# We never hardcode these values — Lambda will have them as env vars
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "")
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(pdf_path: str, doc_id: str) -> None:
    """
    Runs the full ingestion pipeline on a PDF file.
    Works the same whether called locally or from Lambda.
    """
    print(f"[ingest] Starting pipeline for: {doc_id}")

    # 1. Chunk
    chunks = chunk_pdf(pdf_path, doc_id=doc_id)

    # 2. Embed
    records = embed_chunks(chunks, region=AWS_REGION)

    # 3. Store
    store_embeddings(records, host=OPENSEARCH_HOST, region=AWS_REGION)

    print(f"[ingest] ✅ Pipeline complete for: {doc_id}")


# ── Lambda Handler ────────────────────────────────────────────────────────────

def handler(event, context):
    """
    AWS Lambda entry point.

    When you upload a PDF to S3, Lambda receives an event like:
    {
        "Records": [{
            "s3": {
                "bucket": { "name": "my-rag-pdfs" },
                "object": { "key": "story.pdf" }
            }
        }]
    }

    We download the PDF to a temp folder, run the pipeline, then clean up.
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    # S3 can send multiple files in one event — loop through each
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]     # e.g. "story.pdf"
        doc_id = key.replace(".pdf", "").replace("/", "_")

        print(f"[ingest] Received S3 event: s3://{bucket}/{key}")

        # Download PDF to a temporary file
        # Lambda has a /tmp folder we can write to (max 512MB)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            print(f"[ingest] Downloading to {tmp_path}")
            s3_client.download_file(bucket, key, tmp_path)

        try:
            run_pipeline(pdf_path=tmp_path, doc_id=doc_id)
        finally:
            # Always clean up the temp file even if pipeline fails
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
      python -m ingestion.ingest <pdf_path> <opensearch_host>
    """
    if len(sys.argv) < 3:
        print("Usage: python -m ingestion.ingest <pdf_path> <opensearch_host>")
        sys.exit(1)

    pdf_path       = sys.argv[1]
    OPENSEARCH_HOST = sys.argv[2]

    doc_id = os.path.basename(pdf_path).replace(".pdf", "")
    run_pipeline(pdf_path=pdf_path, doc_id=doc_id)