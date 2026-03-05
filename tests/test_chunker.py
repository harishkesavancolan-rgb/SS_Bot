"""
tests/test_chunker.py
----------------------
Tests for the updated chunker.py (LangChain RecursiveCharacterTextSplitter).

What we're checking:
  - chunk_pdf() returns a list of Chunk objects
  - Chunk IDs are unique
  - No empty chunks are produced
  - Page numbers are recorded correctly
  - Empty pages produce no chunks
  - Chunks respect the size limit (roughly)
"""

import pytest
from unittest.mock import patch, MagicMock

from ingestion.chunker import chunk_pdf, _get_splitter, Chunk


# ── Test: _get_splitter() ─────────────────────────────────────────────────────

class TestGetSplitter:

    def test_returns_splitter_instance(self):
        """_get_splitter() should return a RecursiveCharacterTextSplitter."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = _get_splitter(chunk_size=500, chunk_overlap=100)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_short_text_is_one_chunk(self):
        """Text shorter than chunk_size should come back as a single chunk."""
        splitter = _get_splitter(chunk_size=500, chunk_overlap=100)
        result   = splitter.split_text("This is a short sentence.")
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than chunk_size must be split into multiple chunks."""
        splitter  = _get_splitter(chunk_size=100, chunk_overlap=20)
        long_text = "The dragon soared above the mountains. " * 20
        result    = splitter.split_text(long_text)
        assert len(result) > 1

    def test_no_chunk_exceeds_size(self):
        """Each chunk should be at or under chunk_size characters (with small buffer)."""
        chunk_size = 100
        splitter   = _get_splitter(chunk_size=chunk_size, chunk_overlap=20)
        long_text  = "Once upon a time in a magical kingdom far away. " * 30
        chunks     = splitter.split_text(long_text)
        for chunk in chunks:
            assert len(chunk) <= chunk_size * 1.2   # 20% tolerance

    def test_no_empty_chunks(self):
        """The splitter should never return empty strings."""
        splitter  = _get_splitter(chunk_size=100, chunk_overlap=20)
        long_text = "A brave knight rode through the dark forest. " * 20
        chunks    = splitter.split_text(long_text)
        for chunk in chunks:
            assert chunk.strip() != ""


# ── Test: chunk_pdf() ─────────────────────────────────────────────────────────

class TestChunkPdf:
    """
    pdfplumber is MOCKED — no real PDF file needed.
    We control exactly what text each fake page returns.
    """

    def _make_mock_page(self, text: str):
        """Creates a fake pdfplumber page that returns *text*."""
        page = MagicMock()
        page.extract_text.return_value = text
        return page

    @patch("ingestion.chunker.pdfplumber.open")
    def test_returns_list_of_chunk_objects(self, mock_open):
        """chunk_pdf() must return a list of Chunk dataclass instances."""
        fake_page = self._make_mock_page(
            "The witch stirred her cauldron slowly. "
            "The potion bubbled and hissed loudly. "
            "Outside the storm raged on forever. " * 10
        )
        mock_open.return_value.__enter__.return_value.pages = [fake_page]

        chunks = chunk_pdf("fake.pdf", doc_id="test_doc")

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    @patch("ingestion.chunker.pdfplumber.open")
    def test_chunk_ids_are_unique(self, mock_open):
        """Every chunk must have a unique chunk_id — no duplicates."""
        fake_page = self._make_mock_page(
            "A long story about knights and castles. " * 30
        )
        mock_open.return_value.__enter__.return_value.pages = [fake_page]

        chunks = chunk_pdf("fake.pdf", doc_id="test_doc")
        ids    = [c.chunk_id for c in chunks]

        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found!"

    @patch("ingestion.chunker.pdfplumber.open")
    def test_empty_page_produces_no_chunks(self, mock_open):
        """A page with no extractable text should yield zero chunks."""
        fake_page = self._make_mock_page("")
        mock_open.return_value.__enter__.return_value.pages = [fake_page]

        chunks = chunk_pdf("fake.pdf", doc_id="empty_doc")
        assert chunks == []

    @patch("ingestion.chunker.pdfplumber.open")
    def test_page_numbers_are_recorded(self, mock_open):
        """Each chunk must know which page it came from."""
        page1 = self._make_mock_page("First page about magic spells. " * 5)
        page2 = self._make_mock_page("Second page about dark forests. " * 5)
        mock_open.return_value.__enter__.return_value.pages = [page1, page2]

        chunks       = chunk_pdf("fake.pdf", doc_id="test_doc")
        page_numbers = {c.page_number for c in chunks}

        assert 1 in page_numbers
        assert 2 in page_numbers

    @patch("ingestion.chunker.pdfplumber.open")
    def test_doc_id_is_set_on_all_chunks(self, mock_open):
        """Every chunk must carry the correct doc_id."""
        fake_page = self._make_mock_page("A short story about the sea. " * 10)
        mock_open.return_value.__enter__.return_value.pages = [fake_page]

        chunks = chunk_pdf("fake.pdf", doc_id="my_story")

        assert all(c.doc_id == "my_story" for c in chunks)

    @patch("ingestion.chunker.pdfplumber.open")
    def test_metadata_contains_source_and_page(self, mock_open):
        """Each chunk's metadata should have 'source' and 'page_number'."""
        fake_page = self._make_mock_page("The hero crossed the river bravely. " * 10)
        mock_open.return_value.__enter__.return_value.pages = [fake_page]

        chunks = chunk_pdf("fake.pdf", doc_id="test_doc")

        for chunk in chunks:
            assert "source"      in chunk.metadata
            assert "page_number" in chunk.metadata

    @patch("ingestion.chunker.pdfplumber.open")
    def test_chunk_ids_follow_sequential_naming(self, mock_open):
        """Chunk IDs should follow the pattern doc_id::chunk_0000, 0001..."""
        fake_page = self._make_mock_page("A tale of two cities far apart. " * 20)
        mock_open.return_value.__enter__.return_value.pages = [fake_page]

        chunks = chunk_pdf("fake.pdf", doc_id="tale")

        assert chunks[0].chunk_id == "tale::chunk_0000"
        assert chunks[1].chunk_id == "tale::chunk_0001"