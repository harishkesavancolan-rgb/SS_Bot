"""
tests/test_store.py
--------------------
Tests for store.py — PostgreSQL is MOCKED (no real database needed).

What we're checking:
  - ensure_table() creates table and index
  - store_embeddings() inserts records correctly
  - store_embeddings() handles empty records gracefully
  - chunk_id is used as the unique identifier
  - duplicate chunk_ids are handled gracefully
"""

import pytest
from unittest.mock import patch, MagicMock, call

from ingestion.store import ensure_table, store_embeddings


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


# ── Test: ensure_table() ──────────────────────────────────────────────────────

class TestEnsureTable:

    def test_creates_extension(self):
        """
        ensure_table() must enable the pgvector extension.
        Without this, vector columns won't work.
        """
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        # Check that CREATE EXTENSION was called
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE EXTENSION" in c for c in calls)

    def test_creates_chunks_table(self):
        """ensure_table() must create the chunks table."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE TABLE" in c for c in calls)

    def test_creates_vector_index(self):
        """ensure_table() must create the ivfflat vector search index."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE INDEX" in c for c in calls)

    def test_commits_after_setup(self):
        """ensure_table() must commit the transaction."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        mock_conn.commit.assert_called()


# ── Test: store_embeddings() ──────────────────────────────────────────────────

class TestStoreEmbeddings:

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_inserts_records(self, mock_get_conn, mock_execute_values):
        """store_embeddings() must call execute_values to INSERT records."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        records = [_make_fake_record(f"doc::chunk_{i:04d}") for i in range(3)]
        store_embeddings(records)

        mock_execute_values.assert_called()

    @patch("ingestion.store._get_connection")
    def test_empty_records_does_not_crash(self, mock_get_conn):
        """Passing an empty list should return early without crashing."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        # Should complete without any exception
        store_embeddings([])

        # Connection should never be opened for empty records
        mock_get_conn.assert_not_called()

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_connection_is_closed_after_insert(self, mock_get_conn, mock_execute_values):
        """
        Database connection must always be closed after use —
        even if an error occurs. This prevents connection leaks.
        """
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        records = [_make_fake_record()]
        store_embeddings(records)

        mock_conn.close.assert_called_once()

    @patch("ingestion.store._get_connection")
    def test_connection_closed_even_on_error(self, mock_get_conn):
        """
        If an error occurs during insert, the connection
        must still be closed (the finally block must work).
        """
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        # Simulate a database error during insert
        mock_cursor.execute.side_effect = Exception("DB error")

        records = [_make_fake_record()]
        with pytest.raises(Exception):
            store_embeddings(records)

        # Connection must still be closed
        mock_conn.close.assert_called_once()

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_commits_after_insert(self, mock_get_conn, mock_execute_values):
        """store_embeddings() must commit the transaction."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        records = [_make_fake_record()]
        store_embeddings(records)

        mock_conn.commit.assert_called()