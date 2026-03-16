"""
tests/test_llm.py
------------------
Tests for api/llm.py — Bedrock is MOCKED.

What we're checking:
  - generate_answer() calls Mistral Mixtral 8x7B
  - _build_prompt() formats Mistral chat template correctly
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


def _make_fake_mistral_response(answer: str):
    """Mimics boto3 Mistral Mixtral response format."""
    body = MagicMock()
    body.read.return_value = json.dumps({
        "outputs": [{"text": answer}]
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

    def test_uses_mistral_inst_template(self):
        """Prompt must use Mistral [INST] chat template."""
        prompt = _build_prompt("test", [_make_fake_chunk()])
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_empty_chunks_returns_basic_prompt(self):
        """With no chunks, prompt must still be valid Mistral format."""
        prompt = _build_prompt("What is deception?", [])
        assert "[INST]" in prompt
        assert "What is deception?" in prompt

    def test_includes_doc_id(self):
        """Prompt must include the doc_id for traceability."""
        prompt = _build_prompt("test", [_make_fake_chunk()])
        assert "ArtOfWar" in prompt

    def test_includes_rerank_score(self):
        """Prompt must include the chunk relevance score."""
        prompt = _build_prompt("test", [_make_fake_chunk()])
        assert "0.89" in prompt


# ── Test: generate_answer() ───────────────────────────────────────────────────


class TestGenerateAnswer:

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_returns_string(self, mock_boto3):
        """generate_answer() must return a string."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response(
            "Deception is fundamental to warfare."
        )

        result = await generate_answer("What is deception?", [_make_fake_chunk()])

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_calls_mistral_mixtral(self, mock_boto3):
        """generate_answer() must use Mistral Mixtral 8x7B model."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response("answer")

        await generate_answer("test", [_make_fake_chunk()])

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        assert call_kwargs["modelId"] == "mistral.mixtral-8x7b-v1:0"

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_sends_native_prompt_format(self, mock_boto3):
        """Request body must use Mistral native prompt format, not messages."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response("answer")

        await generate_answer("test", [_make_fake_chunk()])

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert "prompt" in body
        assert "messages" not in body

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_prompt_contains_question(self, mock_boto3):
        """The prompt sent to Mistral must contain the user's question."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response("answer")

        await generate_answer("What is the speed of light?", [_make_fake_chunk()])

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert "What is the speed of light?" in body["prompt"]

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_uses_low_temperature(self, mock_boto3):
        """Temperature must be low (≤ 0.2) for grounded, factual answers."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response("answer")

        await generate_answer("test", [_make_fake_chunk()])

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["temperature"] <= 0.2

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_works_without_history(self, mock_boto3):
        """generate_answer() must work fine with no chat history."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response("answer")

        result = await generate_answer("test", [_make_fake_chunk()], None)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_graceful_error_handling(self, mock_boto3):
        """generate_answer() must return a fallback string on Bedrock error."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("Bedrock unavailable")

        result = await generate_answer("test", [_make_fake_chunk()])

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("api.llm.boto3.client")
    async def test_sets_correct_content_type(self, mock_boto3):
        """invoke_model must be called with JSON content type headers."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.invoke_model.return_value = _make_fake_mistral_response("answer")

        await generate_answer("test", [_make_fake_chunk()])

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        assert call_kwargs["contentType"] == "application/json"
        assert call_kwargs["accept"] == "application/json"


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
        """Each source must have chunk_id, display, text, score, page_number."""
        response = build_response("answer", [_make_fake_source()])
        source   = response["sources"][0]

        assert "chunk_id"    in source
        assert "display"     in source
        assert "text"        in source
        assert "score"       in source
        assert "page_number" in source

    def test_source_has_pdf_title(self):
        """Each source must include pdf_title for provenance."""
        response = build_response("answer", [_make_fake_source()])
        assert "pdf_title" in response["sources"][0]

    def test_source_has_no_s3_link(self):
        """Sources must not contain S3 links to whole PDFs."""
        response = build_response("answer", [_make_fake_source()])
        assert "link" not in response["sources"][0]

    def test_multiple_sources_all_included(self):
        """All sources passed in must appear in the response."""
        sources = [_make_fake_source(), _make_fake_source()]
        response = build_response("answer", sources)
        assert len(response["sources"]) == 2

    def test_empty_sources_returns_empty_list(self):
        """build_response() must handle zero sources without error."""
        response = build_response("answer", [])
        assert response["sources"] == []