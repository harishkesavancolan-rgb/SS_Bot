"""
tests/test_embedder.py
-----------------------
Tests for embedder.py — AWS Bedrock is MOCKED (no real API calls).

What we're checking:
  - embed_text() returns a list of floats (a vector)
  - embed_chunks() returns one record per chunk
  - Each record has the required keys: chunk_id, text, embedding
  - The embedding dimension matches what Titan v2 returns (1024)
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from ingestion.chunker  import Chunk
from ingestion.embedder import embed_text, embed_chunks, EMBEDDING_DIM


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_chunk(chunk_id="doc::chunk_0001", text="Once upon a time."):
    """Creates a minimal Chunk for testing."""
    return Chunk(
        chunk_id    = chunk_id,
        doc_id      = "test_doc",
        page_number = 1,
        text        = text,
        metadata    = {"source": "test.pdf", "page_number": 1},
    )


def _make_fake_bedrock_response(dim: int = 1024):
    """
    Mimics what boto3 returns when we call invoke_model on Titan.
    The real response has a 'body' that we .read() and JSON-parse.
    """
    fake_vector  = [0.01] * dim           # a vector of 1024 floats
    body_content = json.dumps({"embedding": fake_vector}).encode("utf-8")

    mock_body = MagicMock()
    mock_body.read.return_value = body_content

    return {"body": mock_body}


# ── Test: embed_text() ────────────────────────────────────────────────────────

class TestEmbedText:

    def test_returns_list_of_floats(self):
        """embed_text() should return a plain Python list of numbers."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_fake_bedrock_response()

        result = embed_text("Hello world", mock_client)

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_correct_dimension(self):
        """The returned vector must have exactly EMBEDDING_DIM values."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_fake_bedrock_response(EMBEDDING_DIM)

        result = embed_text("Some story text", mock_client)

        assert len(result) == EMBEDDING_DIM

    def test_calls_correct_model(self):
        """embed_text() must call the Titan model, not some other model."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_fake_bedrock_response()

        embed_text("test", mock_client)

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        assert "titan-embed-text-v2" in call_kwargs["modelId"]


# ── Test: embed_chunks() ──────────────────────────────────────────────────────

class TestEmbedChunks:

    @patch("ingestion.embedder._get_bedrock_client")
    def test_returns_one_record_per_chunk(self, mock_get_client):
        """embed_chunks() should return exactly as many records as input chunks."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_fake_bedrock_response()
        mock_get_client.return_value = mock_client

        chunks  = [_make_fake_chunk(f"doc::chunk_{i:04d}") for i in range(5)]
        records = embed_chunks(chunks, delay=0)   # delay=0 speeds up tests

        assert len(records) == 5

    @patch("ingestion.embedder._get_bedrock_client")
    def test_each_record_has_required_keys(self, mock_get_client):
        """Every record must contain chunk_id, text, and embedding."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_fake_bedrock_response()
        mock_get_client.return_value = mock_client

        chunks  = [_make_fake_chunk()]
        records = embed_chunks(chunks, delay=0)

        required_keys = {"chunk_id", "doc_id", "page_number", "text", "embedding", "metadata"}
        for record in records:
            assert required_keys.issubset(record.keys())

    @patch("ingestion.embedder._get_bedrock_client")
    def test_chunk_id_is_preserved(self, mock_get_client):
        """The chunk_id in the output must match the input chunk."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_fake_bedrock_response()
        mock_get_client.return_value = mock_client

        chunk   = _make_fake_chunk(chunk_id="my_story::chunk_0042")
        records = embed_chunks([chunk], delay=0)

        assert records[0]["chunk_id"] == "my_story::chunk_0042"

    @patch("ingestion.embedder._get_bedrock_client")
    def test_empty_input_returns_empty_list(self, mock_get_client):
        """Passing an empty list should return an empty list, not crash."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        records = embed_chunks([], delay=0)
        assert records == []
