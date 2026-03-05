"""
store.py
--------
Creates an OpenSearch k-NN index (if needed) and bulk-indexes
embedded chunks produced by embedder.py.
"""

import json
from typing import List

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth
import boto3


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_DIM  = 1024           # must match Titan v2 output dimension
INDEX_NAME     = "rag_chunks"
REGION         = "us-east-1"
SERVICE        = "es"           # use "aoss" for OpenSearch Serverless


# ── Index mapping ─────────────────────────────────────────────────────────────

INDEX_BODY = {
    "settings": {
        "index": {
            "knn"                  : True,
            "knn.algo_param.ef_search": 512,
        }
    },
    "mappings": {
        "properties": {
            "chunk_id"   : {"type": "keyword"},
            "doc_id"     : {"type": "keyword"},
            "page_number": {"type": "integer"},
            "text"       : {"type": "text"},
            "metadata"   : {"type": "object"},
            "embedding"  : {
                "type"      : "knn_vector",
                "dimension" : EMBEDDING_DIM,
                "method"    : {
                    "name"       : "hnsw",
                    "space_type" : "cosinesimil",   # cosine similarity
                    "engine"     : "nmslib",
                    "parameters" : {
                        "ef_construction": 512,
                        "m"              : 16,
                    },
                },
            },
        }
    },
}


# ── Client factory ────────────────────────────────────────────────────────────

def _get_opensearch_client(host: str, region: str = REGION) -> OpenSearch:
    """
    Builds an OpenSearch client that signs every request with AWS SigV4.

    *host* should be the bare domain, e.g.:
        "my-domain.us-east-1.es.amazonaws.com"
    (no https:// prefix, no trailing slash)
    """
    credentials = boto3.Session().get_credentials()
    awsauth     = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        SERVICE,
        session_token=credentials.token,
    )

    return OpenSearch(
        hosts              = [{"host": host, "port": 443}],
        http_auth          = awsauth,
        use_ssl            = True,
        verify_certs       = True,
        connection_class   = RequestsHttpConnection,
        timeout            = 60,
    )


# ── Index management ──────────────────────────────────────────────────────────

def ensure_index(client: OpenSearch, index: str = INDEX_NAME) -> None:
    """Create the k-NN index only if it doesn't already exist."""
    if client.indices.exists(index=index):
        print(f"[store] index '{index}' already exists — skipping creation")
        return

    client.indices.create(index=index, body=INDEX_BODY)
    print(f"[store] index '{index}' created ✓")


# ── Bulk indexing ─────────────────────────────────────────────────────────────

def _build_actions(records: List[dict], index: str):
    """Generator that yields bulk-index action dicts."""
    for record in records:
        yield {
            "_index" : index,
            "_id"    : record["chunk_id"],
            "_source": {
                "chunk_id"   : record["chunk_id"],
                "doc_id"     : record["doc_id"],
                "page_number": record["page_number"],
                "text"       : record["text"],
                "embedding"  : record["embedding"],
                "metadata"   : record.get("metadata", {}),
            },
        }


def store_embeddings(
    records    : List[dict],
    host       : str,
    index      : str  = INDEX_NAME,
    region     : str  = REGION,
    chunk_size : int  = 50,        # docs per bulk request
) -> None:
    """
    Index a list of embedded-chunk dicts into OpenSearch.

    Parameters
    ----------
    records    : output from embedder.embed_chunks()
    host       : OpenSearch domain endpoint (no https://)
    index      : target index name
    region     : AWS region
    chunk_size : number of documents per bulk request
    """
    client = _get_opensearch_client(host, region)
    ensure_index(client, index)

    success, failed = helpers.bulk(
        client,
        _build_actions(records, index),
        chunk_size  = chunk_size,
        raise_on_error = False,
    )

    print(f"[store] indexed {success} docs  |  failed {len(failed)}")
    if failed:
        for err in failed[:5]:
            print("  ✗", json.dumps(err))


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from chunker  import chunk_pdf
    from embedder import embed_chunks

    if len(sys.argv) < 3:
        print("Usage: python store.py <pdf_path> <opensearch_host>")
        sys.exit(1)

    pdf_file, os_host = sys.argv[1], sys.argv[2]

    chunks  = chunk_pdf(pdf_file)
    records = embed_chunks(chunks)
    store_embeddings(records, host=os_host)