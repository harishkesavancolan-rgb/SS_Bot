"""
tests/test_retriever.py
------------------------
Tests for api/retriever.py — all AWS and DB calls are MOCKED.

What we're checking:
  - embed_question() returns a 1024-dim vector
  - vector_search() queries pgvector correctly
  - rerank() calls Cohere and reorders chunks
  - build_sources() builds correct source objects
  - retrieve() orchestrates all steps correctly
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from api.retriever import (
    embed_question,
    vector_search,
    rerank,
    build_sources,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_chunk(chunk_id="doc::chunk_0001", page=3, score=0.89):
    return {
        "chunk_id"        : chunk_id,
        "doc_id"          : "ArtOfWar",
        "page_number"     : page,
        "text"            : "All warfare is based on deception.",
        "metadata"        : {"source": "ArtOfWar.pdf"},
        "similarity_score": score,
    }


def _make_fake_bedrock_response(data: dict):
    """Mimics boto3 bedrock response body."""
    body = MagicMock()
    body.read.return_value = json.dumps(data).encode("utf-8")
    return {"body": body}


# ── Test: embed_question() ────────────────────────────────────────────────────

class TestEmbedQuestion:

    @patch("api.retriever.boto3.client")
    def test_returns_1024_dim_vector(self, mock_boto3):
        """embed_question() must return a 1024-dim vector."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_bedrock_response(
            {"embedding": [0.01] * 1024}
        )

        result = embed_question("What is deception?")

        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("api.retriever.boto3.client")
    def test_calls_titan_model(self, mock_boto3):
        """embed_question() must use Titan embed model."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_bedrock_response(
            {"embedding": [0.01] * 1024}
        )

        embed_question("test question")

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        assert "titan-embed" in call_kwargs["modelId"]


# ── Test: vector_search() ─────────────────────────────────────────────────────

class TestVectorSearch:

    @patch("api.retriever._get_connection")
    def test_returns_list_of_chunks(self, mock_get_conn):
        """vector_search() should return a list of chunk dicts."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [_make_fake_chunk()]
        mock_get_conn.return_value = mock_conn

        results = vector_search([0.01] * 1024, user_id="user_123")

        assert isinstance(results, list)

    @patch("api.retriever._get_connection")
    def test_filters_by_user_id(self, mock_get_conn):
        """vector_search() must filter chunks by user_id."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        vector_search([0.01] * 1024, user_id="user_123")

        query = mock_cursor.execute.call_args[0][0]
        assert "user_id" in query

    @patch("api.retriever._get_connection")
    def test_closes_connection(self, mock_get_conn):
        """Connection must always be closed after search."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        vector_search([0.01] * 1024, user_id="user_123")

        mock_conn.close.assert_called_once()


# ── Test: rerank() ────────────────────────────────────────────────────────────

class TestRerank:

    @patch("api.retriever.boto3.client")
    def test_returns_top_n_chunks(self, mock_boto3):
        """rerank() should return at most RERANK_TOP_N chunks."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.rerank.return_value = {
            "rerankingResults": [
                {"index": 0, "relevanceScore": 0.95},
                {"index": 1, "relevanceScore": 0.87},
            ]
        }

        chunks  = [_make_fake_chunk(f"doc::chunk_{i:04d}") for i in range(5)]
        results = rerank("What is deception?", chunks, top_n=2)

        assert len(results) == 2

    @patch("api.retriever.boto3.client")
    def test_adds_rerank_score(self, mock_boto3):
        """Each reranked chunk must have a rerank_score field."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.rerank.return_value = {
            "rerankingResults": [{"index": 0, "relevanceScore": 0.95}]
        }

        chunks  = [_make_fake_chunk()]
        results = rerank("test", chunks, top_n=1)

        assert "rerank_score" in results[0]

    def test_empty_chunks_returns_empty(self):
        """rerank() with empty input should return empty list."""
        results = rerank("test", [])
        assert results == []


# ── Test: build_sources() ─────────────────────────────────────────────────────

class TestBuildSources:

    def test_returns_correct_fields(self):
        """Each source must have all required fields."""
        chunks  = [_make_fake_chunk()]
        sources = build_sources(chunks)

        required = {"chunk_id", "pdf_title", "page_number", "score", "text", "display"}
        assert required.issubset(sources[0].keys())

    def test_display_format(self):
        """display field must follow 'title | Page X | Score: Y' format."""
        chunks  = [_make_fake_chunk(page=3, score=0.89)]
        sources = build_sources(chunks)

        assert "Page 3"  in sources[0]["display"]
        assert "Score"   in sources[0]["display"]

    def test_text_is_exact_chunk_text(self):
        """text field must be the exact chunk text — not the whole PDF."""
        chunks  = [_make_fake_chunk()]
        sources = build_sources(chunks)

        assert sources[0]["text"] == "All warfare is based on deception."

    def test_no_s3_link(self):
        """Sources must NOT contain an S3 link to the whole PDF."""
        chunks  = [_make_fake_chunk()]
        sources = build_sources(chunks)

        assert "link" not in sources[0]