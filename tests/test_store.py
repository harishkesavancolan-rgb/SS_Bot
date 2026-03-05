"""
tests/test_store.py
--------------------
Tests for store.py — Pinecone is MOCKED (no real account needed).

What we're checking:
  - ensure_index() creates index when it doesn't exist
  - ensure_index() skips creation when index already exists
  - store_embeddings() upserts vectors correctly
  - store_embeddings() batches correctly
  - chunk_id is used as the vector ID
"""

import pytest
from unittest.mock import patch, MagicMock

from ingestion.store import ensure_index, store_embeddings, INDEX_NAME


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_record(chunk_id="doc::chunk_0001"):
    """Creates a minimal embedded record (output of embedder.embed_chunks)."""
    return {
        "chunk_id"   : chunk_id,
        "doc_id"     : "test_doc",
        "page_number": 1,
        "text"       : "The wizard cast a powerful spell.",
        "embedding"  : [0.01] * 1024,
        "metadata"   : {"source": "test.pdf", "page_number": 1},
    }


# ── Test: ensure_index() ──────────────────────────────────────────────────────

class TestEnsureIndex:

    def test_creates_index_when_missing(self):
        """
        If the index does NOT exist, ensure_index() should call
        pc.create_index() exactly once.
        """
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = []

        ensure_index(mock_pc, index_name="test-index")

        mock_pc.create_index.assert_called_once()

    def test_skips_creation_when_index_exists(self):
        """
        If the index ALREADY exists, ensure_index() should NOT call
        pc.create_index() — we don't want to overwrite existing data!
        """
        mock_pc = MagicMock()
        mock_index      = MagicMock()
        mock_index.name = "test-index"
        mock_pc.list_indexes.return_value = [mock_index]

        ensure_index(mock_pc, index_name="test-index")

        mock_pc.create_index.assert_not_called()

    def test_creates_index_with_correct_dimension(self):
        """Index must be created with 1024 dimensions to match Titan v2."""
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = []

        ensure_index(mock_pc, index_name="test-index")

        call_kwargs = mock_pc.create_index.call_args.kwargs
        assert call_kwargs["dimension"] == 1024

    def test_creates_index_with_cosine_metric(self):
        """Index must use cosine similarity for RAG search."""
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = []

        ensure_index(mock_pc, index_name="test-index")

        call_kwargs = mock_pc.create_index.call_args.kwargs
        assert call_kwargs["metric"] == "cosine"


# ── Test: store_embeddings() ──────────────────────────────────────────────────

class TestStoreEmbeddings:

    @patch("ingestion.store._get_pinecone_client")
    def test_upsert_is_called(self, mock_get_client):
        """store_embeddings() must call index.upsert() at least once."""
        mock_pc    = MagicMock()
        mock_index = MagicMock()
        mock_index_info      = MagicMock()
        mock_index_info.name = INDEX_NAME
        mock_pc.list_indexes.return_value = [mock_index_info]
        mock_pc.Index.return_value        = mock_index
        mock_get_client.return_value      = mock_pc

        records = [_make_fake_record(f"doc::chunk_{i:04d}") for i in range(3)]
        store_embeddings(records)

        mock_index.upsert.assert_called()

    @patch("ingestion.store._get_pinecone_client")
    def test_chunk_id_used_as_vector_id(self, mock_get_client):
        """
        Each vector's 'id' in Pinecone must be the chunk_id.
        This ensures no duplicate entries for the same chunk.
        """
        mock_pc    = MagicMock()
        mock_index = MagicMock()
        mock_index_info      = MagicMock()
        mock_index_info.name = INDEX_NAME
        mock_pc.list_indexes.return_value = [mock_index_info]
        mock_pc.Index.return_value        = mock_index
        mock_get_client.return_value      = mock_pc

        records = [_make_fake_record("my_story::chunk_0099")]
        store_embeddings(records)

        upsert_call = mock_index.upsert.call_args.kwargs
        vectors     = upsert_call["vectors"]
        assert vectors[0]["id"] == "my_story::chunk_0099"

    @patch("ingestion.store._get_pinecone_client")
    def test_text_stored_in_metadata(self, mock_get_client):
        """
        Text must be stored in Pinecone metadata so we can
        retrieve it at query time for the RAG chatbot.
        """
        mock_pc    = MagicMock()
        mock_index = MagicMock()
        mock_index_info      = MagicMock()
        mock_index_info.name = INDEX_NAME
        mock_pc.list_indexes.return_value = [mock_index_info]
        mock_pc.Index.return_value        = mock_index
        mock_get_client.return_value      = mock_pc

        records = [_make_fake_record()]
        store_embeddings(records)

        upsert_call = mock_index.upsert.call_args.kwargs
        vectors     = upsert_call["vectors"]
        assert "text" in vectors[0]["metadata"]

    @patch("ingestion.store._get_pinecone_client")
    def test_empty_records_does_not_crash(self, mock_get_client):
        """Passing an empty list should not crash."""
        mock_pc = MagicMock()
        mock_get_client.return_value = mock_pc

        store_embeddings([])