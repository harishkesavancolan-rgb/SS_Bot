"""
tests/test_chat.py
-------------------
Tests for api/chat.py — FastAPI endpoints.

What we're checking:
  - GET  /           → health check returns 200
  - POST /sessions/new → creates session with ID
  - POST /chat        → returns answer + sources
  - GET  /sessions    → lists user sessions
  - POST /upload      → rejects non-PDF, requires session_id
"""

import pytest
import pytest_asyncio
from httpx             import AsyncClient, ASGITransport
from unittest.mock     import patch, MagicMock, AsyncMock

from api.chat import app


# ── Test: GET / ───────────────────────────────────────────────────────────────

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_returns_200(self):
        """Health check endpoint must return 200."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_running_status(self):
        """Health check must confirm API is running."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")

        assert "running" in response.json()["status"].lower()


# ── Test: POST /sessions/new ──────────────────────────────────────────────────

class TestNewSession:

    @pytest.mark.asyncio
    @patch("api.chat._get_connection")
    async def test_returns_session_id(self, mock_get_conn):
        """POST /sessions/new must return a session_id."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/sessions/new",
                json={"user_id": "user_123"}
            )

        assert response.status_code == 200
        assert "session_id" in response.json()

    @pytest.mark.asyncio
    @patch("api.chat._get_connection")
    async def test_session_ids_are_unique(self, mock_get_conn):
        """Each call must return a different session_id."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.post("/sessions/new", json={"user_id": "user_123"})
            r2 = await client.post("/sessions/new", json={"user_id": "user_123"})

        assert r1.json()["session_id"] != r2.json()["session_id"]


# ── Test: POST /chat ──────────────────────────────────────────────────────────

class TestChat:

    @pytest.mark.asyncio
    @patch("api.chat.retrieve")
    @patch("api.chat.generate_answer")
    @patch("api.chat.get_session_history")
    @patch("api.chat.save_message")
    async def test_returns_answer_and_sources(
        self, mock_save, mock_history, mock_generate, mock_retrieve
    ):
        """POST /chat must return answer and sources."""
        mock_history.return_value  = []
        mock_retrieve.return_value = {
            "chunks" : [{"chunk_id": "doc::chunk_0001", "text": "test", "page_number": 1, "doc_id": "ArtOfWar", "rerank_score": 0.9, "metadata": {"source": "ArtOfWar.pdf"}}],
            "sources": [{"chunk_id": "doc::chunk_0001", "display": "ArtOfWar.pdf | Page 1 | Score: 0.9", "text": "test", "pdf_title": "ArtOfWar.pdf", "page_number": 1, "score": 0.9}],
        }
        mock_generate.return_value = "Sun Tzu believed deception was key."

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/chat", json={
                "question"  : "What is deception?",
                "user_id"   : "user_123",
                "session_id": "session_abc",
            })

        assert response.status_code == 200
        assert "answer"  in response.json()
        assert "sources" in response.json()

    @pytest.mark.asyncio
    @patch("api.chat.retrieve")
    @patch("api.chat.get_session_history")
    async def test_retrieve_called_with_session_id(self, mock_history, mock_retrieve):
        """POST /chat must pass session_id to retrieve() for isolation."""
        mock_history.return_value  = []
        mock_retrieve.return_value = {"chunks": [], "sources": []}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/chat", json={
                "question"  : "What is deception?",
                "user_id"   : "user_123",
                "session_id": "session_abc",
            })

        # Verify session_id was passed to retrieve
        call_kwargs = mock_retrieve.call_args.kwargs
        assert call_kwargs.get("session_id") == "session_abc"
    @pytest.mark.asyncio
    @patch("api.chat.retrieve")
    @patch("api.chat.generate_answer")
    @patch("api.chat.get_session_history")
    @patch("api.chat.save_message")
    async def test_returns_answer_when_no_chunks(
        self, mock_save, mock_history, mock_generate, mock_retrieve
    ):
        """POST /chat must respond naturally even when no chunks found (e.g. greetings)."""
        mock_history.return_value  = []
        mock_retrieve.return_value = {"chunks": [], "sources": []}
        mock_generate.return_value = "Hello! How can I help you?"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/chat", json={
                "question"  : "hi",
                "user_id"   : "user_123",
                "session_id": "session_abc",
            })

        assert response.status_code == 200
        assert response.json()["answer"] == "Hello! How can I help you?"
        assert response.json()["sources"] == []

# ── Test: POST /upload ────────────────────────────────────────────────────────

class TestUpload:

    @pytest.mark.asyncio
    async def test_rejects_non_pdf(self):
        """POST /upload must reject non-PDF files."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/upload?user_id=user_123&session_id=session_abc",
                files={"file": ("test.txt", b"hello", "text/plain")},
            )

        assert response.status_code == 400

    @pytest.mark.asyncio
    @patch("api.chat.boto3.client")
    async def test_accepts_pdf_with_session_id(self, mock_boto3):
        """POST /upload must accept PDF files and include session_id in S3 key."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/upload?user_id=user_123&session_id=session_abc",
                files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
            )

        assert response.status_code == 200
        assert "uploaded successfully" in response.json()["message"]

    @pytest.mark.asyncio
    @patch("api.chat.boto3.client")
    async def test_s3_key_includes_session_id(self, mock_boto3):
        """S3 key must be user_id/session_id/filename.pdf for isolation."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/upload?user_id=user_123&session_id=session_abc",
                files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
            )

        # S3 key must contain both user_id and session_id
        call_args = mock_s3.upload_fileobj.call_args
        s3_key    = call_args[0][2]   # third positional arg is the key
        assert "user_123"    in s3_key
        assert "session_abc" in s3_key
        assert "test.pdf"    in s3_key

    @pytest.mark.asyncio
    async def test_upload_missing_session_id_fails(self):
        """POST /upload without session_id must fail — prevents orphaned uploads."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/upload?user_id=user_123",   # no session_id
                files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
            )

        assert response.status_code == 422   # FastAPI validation error