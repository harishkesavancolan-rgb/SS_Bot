"""
tests/test_llm.py
------------------
Tests for api/llm.py — Bedrock is MOCKED.

What we're checking:
  - generate_answer() calls Claude Haiku
  - Chat history is included in messages
  - build_response() formats correctly
  - Response contains answer + sources with correct fields
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from api.llm import generate_answer, build_response, _build_prompt


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_chunk(chunk_id="doc::chunk_0001"):
    return {
        "chunk_id"    : chunk_id,
        "doc_id"      : "ArtOfWar",
        "page_number" : 3,
        "text"        : "All warfare is based on deception.",
        "rerank_score": 0.89,
        "metadata"    : {"source": "ArtOfWar.pdf"},
    }


def _make_fake_claude_response(answer: str):
    """Mimics boto3 Titan Text response."""
    body = MagicMock()
    body.read.return_value = json.dumps({
        "results": [{"outputText": answer}]
    }).encode("utf-8")
    return {"body": body}


def _make_fake_source():
    return {
        "chunk_id"   : "ArtOfWar::chunk_0042",
        "display"    : "ArtOfWar.pdf | Page 3 | Score: 0.89",
        "text"       : "All warfare is based on deception.",
        "pdf_title"  : "ArtOfWar.pdf",
        "page_number": 3,
        "score"      : 0.89,
    }


# ── Test: _build_prompt() ─────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_includes_question(self):
        """Prompt must include the user's question."""
        prompt = _build_prompt("What is deception?", [_make_fake_chunk()])
        assert "What is deception?" in prompt

    def test_includes_chunk_text(self):
        """Prompt must include the chunk text."""
        prompt = _build_prompt("test", [_make_fake_chunk()])
        assert "All warfare is based on deception." in prompt

    def test_includes_page_number(self):
        """Prompt must include page number for reference."""
        prompt = _build_prompt("test", [_make_fake_chunk()])
        assert "Page 3" in prompt

    def test_multiple_chunks_all_included(self):
        """All chunks must appear in the prompt."""
        chunks = [
            _make_fake_chunk("doc::chunk_0001"),
            _make_fake_chunk("doc::chunk_0002"),
        ]
        prompt = _build_prompt("test", chunks)
        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt


# ── Test: generate_answer() ───────────────────────────────────────────────────

class TestGenerateAnswer:

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_returns_string(self, mock_boto3):
        """generate_answer() must return a string."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_claude_response(
            "Deception is fundamental to warfare."
        )

        result = await generate_answer("What is deception?", [_make_fake_chunk()])

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_calls_claude_haiku(self, mock_boto3):
        """generate_answer() must use Claude 3 Haiku model."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_claude_response("answer")

        await generate_answer("test", [_make_fake_chunk()])

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        assert "titan" in call_kwargs["modelId"].lower()

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_includes_chat_history(self, mock_boto3):
        """Chat history must be included in messages sent to Claude."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_claude_response("answer")

        history = [
            {"role": "user",      "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        await generate_answer("follow up question", [_make_fake_chunk()], history)

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        roles = [m["role"] for m in body["messages"]]
        assert "user"      in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_works_without_history(self, mock_boto3):
        """generate_answer() must work fine with no chat history."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_claude_response("answer")

        result = await generate_answer("test", [_make_fake_chunk()], None)

        assert isinstance(result, str)


# ── Test: build_response() ────────────────────────────────────────────────────

class TestBuildResponse:

    def test_contains_answer_and_sources(self):
        """build_response() must return both answer and sources."""
        response = build_response("test answer", [_make_fake_source()])

        assert "answer"  in response
        assert "sources" in response

    def test_answer_is_preserved(self):
        """The answer text must not be modified."""
        response = build_response("exact answer text", [_make_fake_source()])
        assert response["answer"] == "exact answer text"

    def test_source_has_required_fields(self):
        """Each source must have chunk_id, display, text, score."""
        response = build_response("answer", [_make_fake_source()])
        source   = response["sources"][0]

        assert "chunk_id"    in source
        assert "display"     in source
        assert "text"        in source
        assert "score"       in source
        assert "page_number" in source

    def test_source_has_no_s3_link(self):
        """Sources must not contain S3 links to whole PDFs."""
        response = build_response("answer", [_make_fake_source()])
        assert "link" not in response["sources"][0]