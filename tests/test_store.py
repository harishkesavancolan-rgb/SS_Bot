"""
tests/test_store.py
--------------------
Tests for store.py — PostgreSQL is MOCKED.

What we're checking:
  - ensure_table() creates table with session_id column and indexes
  - store_embeddings() inserts records with session_id
  - session_id is stored per chunk for retrieval isolation
  - Empty records handled gracefully
  - Connection always closed
"""

import pytest
from unittest.mock import patch, MagicMock, call

from ingestion.store import ensure_table, store_embeddings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_record(chunk_id="doc::chunk_0001"):
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
        """ensure_table() must enable the pgvector extension."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

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

    def test_table_includes_session_id_column(self):
        """chunks table must include a session_id column."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("session_id" in c for c in calls)

    def test_creates_vector_index(self):
        """ensure_table() must create the ivfflat vector search index."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE INDEX" in c for c in calls)

    def test_creates_user_session_index(self):
        """ensure_table() must create a (user_id, session_id) composite index."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        ensure_table(mock_conn)

        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("user_id" in c and "session_id" in c for c in calls)

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

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_session_id_stored_in_rows(self, mock_get_conn, mock_execute_values):
        """session_id must be included in every inserted row."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        records = [_make_fake_record()]
        store_embeddings(records, user_id="user_123", session_id="session_abc")

        # Grab the rows passed to execute_values
        rows = mock_execute_values.call_args[0][2]
        assert any("session_abc" in row for row in rows)

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_different_sessions_store_separately(self, mock_get_conn, mock_execute_values):
        """
        Calling store_embeddings with different session_ids must
        tag rows with the correct session_id each time.
        """
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        store_embeddings([_make_fake_record("doc::chunk_0001")], session_id="session_A")
        rows_A = mock_execute_values.call_args[0][2]
        assert any("session_A" in row for row in rows_A)

        store_embeddings([_make_fake_record("doc::chunk_0002")], session_id="session_B")
        rows_B = mock_execute_values.call_args[0][2]
        assert any("session_B" in row for row in rows_B)

    @patch("ingestion.store._get_connection")
    def test_empty_records_does_not_crash(self, mock_get_conn):
        """Passing an empty list should return early without crashing."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        store_embeddings([])

        mock_get_conn.assert_not_called()

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_connection_is_closed_after_insert(self, mock_get_conn, mock_execute_values):
        """Database connection must always be closed after use."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        store_embeddings([_make_fake_record()])

        mock_conn.close.assert_called_once()

    @patch("ingestion.store._get_connection")
    def test_connection_closed_even_on_error(self, mock_get_conn):
        """Connection must still be closed even if an error occurs."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        mock_cursor.execute.side_effect = Exception("DB error")

        with pytest.raises(Exception):
            store_embeddings([_make_fake_record()])

        mock_conn.close.assert_called_once()

    @patch("ingestion.store.execute_values")
    @patch("ingestion.store._get_connection")
    def test_commits_after_insert(self, mock_get_conn, mock_execute_values):
        """store_embeddings() must commit the transaction."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        store_embeddings([_make_fake_record()])

        mock_conn.commit.assert_called()