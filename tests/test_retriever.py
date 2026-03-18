"""
tests/test_retriever.py
------------------------
Tests for api/retriever.py — all AWS, DB, and ML model calls are MOCKED.

What we're checking:
  - embed_question() returns a 1024-dim vector
  - vector_search() filters by both user_id AND session_id
  - rerank() reorders chunks using local cross-encoder
  - build_sources() builds correct source objects
  - retrieve() passes session_id through the pipeline
"""

import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# ── Mock sentence_transformers BEFORE importing api.retriever ─────────────────
# CrossEncoder loads an ~80MB model at import time.
# We must mock the entire module before retriever.py is imported,
# otherwise the model tries to download and the test crashes.
_mock_reranker_instance = MagicMock()
_mock_reranker_instance.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]

_mock_cross_encoder_class = MagicMock(return_value=_mock_reranker_instance)

sys.modules["sentence_transformers"] = MagicMock(
    CrossEncoder=_mock_cross_encoder_class
)

# Now safe to import retriever
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

        results = vector_search([0.01] * 1024, user_id="user_123", session_id="session_abc")

        assert isinstance(results, list)

    @patch("api.retriever._get_connection")
    def test_filters_by_user_id(self, mock_get_conn):
        """vector_search() must filter chunks by user_id."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        vector_search([0.01] * 1024, user_id="user_123", session_id="session_abc")

        query = mock_cursor.execute.call_args[0][0]
        assert "user_id" in query

    @patch("api.retriever._get_connection")
    def test_filters_by_session_id(self, mock_get_conn):
        """vector_search() must filter by session_id to isolate chat sessions."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        vector_search([0.01] * 1024, user_id="user_123", session_id="session_abc")

        query = mock_cursor.execute.call_args[0][0]
        assert "session_id" in query

    @patch("api.retriever._get_connection")
    def test_session_id_passed_as_param(self, mock_get_conn):
        """session_id must be passed as a query parameter, not interpolated."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        vector_search([0.01] * 1024, user_id="user_123", session_id="session_abc")

        params = mock_cursor.execute.call_args[0][1]
        assert "session_abc" in params

    @patch("api.retriever._get_connection")
    def test_closes_connection(self, mock_get_conn):
        """Connection must always be closed after search."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        vector_search([0.01] * 1024, user_id="user_123", session_id="session_abc")

        mock_conn.close.assert_called_once()


# ── Test: rerank() ────────────────────────────────────────────────────────────
# rerank() now uses a local CrossEncoder — no boto3/AWS calls
# The model is mocked via sys.modules at the top of this file

class TestRerank:

    def test_returns_top_n_chunks(self):
        """rerank() should return at most top_n chunks."""
        chunks  = [_make_fake_chunk(f"doc::chunk_{i:04d}") for i in range(5)]
        results = rerank("What is deception?", chunks, top_n=2)

        assert len(results) == 2

    def test_adds_rerank_score(self):
        """Each reranked chunk must have a rerank_score field."""
        chunks  = [_make_fake_chunk()]
        results = rerank("test", chunks, top_n=1)

        assert "rerank_score" in results[0]

    def test_empty_chunks_returns_empty(self):
        """rerank() with empty input should return empty list."""
        results = rerank("test", [])

        assert results == []

    def test_sorted_by_score_descending(self):
        """rerank() must return chunks sorted by rerank_score highest first."""
        chunks  = [_make_fake_chunk(f"doc::chunk_{i:04d}") for i in range(5)]
        results = rerank("test", chunks, top_n=5)
        scores  = [r["rerank_score"] for r in results]

        assert scores == sorted(scores, reverse=True)

    def test_fewer_chunks_than_top_n(self):
        """rerank() with fewer chunks than top_n must return all chunks safely."""
        chunks  = [_make_fake_chunk("doc::chunk_0001"), _make_fake_chunk("doc::chunk_0002")]
        results = rerank("test", chunks, top_n=5)

        assert len(results) == 2


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

        assert "Page 3" in sources[0]["display"]
        assert "Score"  in sources[0]["display"]

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


# ── Test: retrieve() — session isolation ─────────────────────────────────────

class TestRetrieve:

    @pytest.mark.asyncio
    @patch("api.retriever.vector_search")
    @patch("api.retriever.embed_question")
    async def test_passes_session_id_to_vector_search(
        self, mock_embed, mock_vector_search
    ):
        """retrieve() must pass session_id to vector_search for isolation."""
        from api.retriever import retrieve

        mock_embed.return_value         = [0.01] * 1024
        mock_vector_search.return_value = []

        await retrieve("What is deception?", user_id="user_123", session_id="session_abc")

        call_kwargs = mock_vector_search.call_args
        assert "session_abc" in call_kwargs[0] or call_kwargs[1].get("session_id") == "session_abc"

    @pytest.mark.asyncio
    @patch("api.retriever.vector_search")
    @patch("api.retriever.embed_question")
    async def test_returns_empty_when_no_chunks(self, mock_embed, mock_vector_search):
        """retrieve() must return empty chunks/sources when nothing found."""
        from api.retriever import retrieve

        mock_embed.return_value         = [0.01] * 1024
        mock_vector_search.return_value = []

        result = await retrieve("test", user_id="user_123", session_id="session_abc")

        assert result == {"chunks": [], "sources": []}