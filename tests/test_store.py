"""
tests/test_store.py
--------------------
Tests for store.py — OpenSearch is MOCKED (no real cluster needed).

What we're checking:
  - ensure_index() creates an index when one doesn't exist
  - ensure_index() skips creation when index already exists
  - store_embeddings() calls bulk indexing
  - Documents are built with the correct structure
"""

import pytest
from unittest.mock import patch, MagicMock, call

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
        client.indices.create() exactly once.
        """
        mock_client = MagicMock()
        # Simulate: index does not exist
        mock_client.indices.exists.return_value = False

        ensure_index(mock_client, index="test_index")

        mock_client.indices.create.assert_called_once()

    def test_skips_creation_when_index_exists(self):
        """
        If the index ALREADY exists, ensure_index() should NOT call
        client.indices.create() — we don't want to overwrite existing data!
        """
        mock_client = MagicMock()
        # Simulate: index already exists
        mock_client.indices.exists.return_value = True

        ensure_index(mock_client, index="test_index")

        mock_client.indices.create.assert_not_called()

    def test_checks_correct_index_name(self):
        """ensure_index() must check for the right index name."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        ensure_index(mock_client, index="my_custom_index")

        mock_client.indices.exists.assert_called_once_with(index="my_custom_index")


# ── Test: store_embeddings() ──────────────────────────────────────────────────

class TestStoreEmbeddings:

    @patch("ingestion.store.helpers.bulk")
    @patch("ingestion.store._get_opensearch_client")
    def test_bulk_is_called(self, mock_get_client, mock_bulk):
        """store_embeddings() must call bulk indexing at least once."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_get_client.return_value = mock_client
        mock_bulk.return_value = (3, [])    # (success_count, failed_list)

        records = [_make_fake_record(f"doc::chunk_{i:04d}") for i in range(3)]
        store_embeddings(records, host="fake-host.us-east-1.es.amazonaws.com")

        mock_bulk.assert_called_once()

    @patch("ingestion.store.helpers.bulk")
    @patch("ingestion.store._get_opensearch_client")
    def test_empty_records_still_calls_bulk(self, mock_get_client, mock_bulk):
        """Passing an empty list should not crash."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_get_client.return_value = mock_client
        mock_bulk.return_value = (0, [])

        store_embeddings([], host="fake-host.us-east-1.es.amazonaws.com")

        # Should still reach bulk without throwing an exception
        mock_bulk.assert_called_once()

    @patch("ingestion.store.helpers.bulk")
    @patch("ingestion.store._get_opensearch_client")
    def test_uses_chunk_id_as_document_id(self, mock_get_client, mock_bulk):
        """
        Each document indexed in OpenSearch should use chunk_id as its _id.
        This ensures we never create duplicate entries for the same chunk.
        """
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_get_client.return_value = mock_client
        mock_bulk.return_value = (1, [])

        records = [_make_fake_record("my_story::chunk_0099")]
        store_embeddings(records, host="fake-host.us-east-1.es.amazonaws.com")

        # Grab what was passed to bulk and check the action dicts
        bulk_call_args = mock_bulk.call_args
        actions = list(bulk_call_args[0][1])   # second positional arg is the generator

        assert actions[0]["_id"] == "my_story::chunk_0099"
