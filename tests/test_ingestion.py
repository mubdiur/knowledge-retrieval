"""Tests for the ingestion pipeline components."""

import pytest
import tempfile
import os

from app.ingestion.parser import FileParser
from app.ingestion.chunker import AdaptiveChunker
from app.ingestion.extractor import EntityExtractor


class TestFileParser:
    def test_parse_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world\nThis is a test file.")
            f.flush()
            result = FileParser.parse(f.name)
            assert result["content"] == "Hello world\nThis is a test file."
            assert result["doc_type"] != "unknown"
            os.unlink(f.name)

    def test_parse_unsupported(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            f.flush()
            with pytest.raises(ValueError):
                FileParser.parse(f.name)
            os.unlink(f.name)

    def test_doc_type_inference(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Runbook for Payment Gateway\nSteps to recover.")
            f.flush()
            result = FileParser.parse(f.name)
            assert result["doc_type"] == "runbook"
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2026-01-01 ERROR something failed")
            f.flush()
            result = FileParser.parse(f.name)
            assert result["doc_type"] == "log"
            os.unlink(f.name)


class TestAdaptiveChunker:
    def setup_method(self):
        self.chunker = AdaptiveChunker(max_size=200, min_size=20, overlap=20)

    def test_empty_text(self):
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   ") == []

    def test_short_text_single_chunk(self):
        chunks = self.chunker.chunk("Short text under limit.")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Short text under limit."

    def test_chunk_respects_max_size(self):
        text = " ".join(["word"] * 500)
        chunks = self.chunker.chunk(text)
        for c in chunks:
            assert len(c["content"]) <= 200

    def test_chunk_metadata(self):
        chunks = self.chunker.chunk("Test content.", metadata={"source": "test"})
        assert chunks[0]["metadata"] == {"source": "test"}
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["token_count"] > 0


class TestEntityExtractor:
    def test_extract_hostnames(self):
        text = "Host web-01.prod is down. Also db-01.staging is slow."
        entities = EntityExtractor.extract(text)
        assert "web-01.prod" in entities["hostnames"]
        assert "db-01.staging" in entities["hostnames"]

    def test_extract_ip_addresses(self):
        text = "Server 10.0.1.20 is unreachable from 192.168.1.1"
        entities = EntityExtractor.extract(text)
        assert "10.0.1.20" in entities["ip_addresses"]
        assert "192.168.1.1" in entities["ip_addresses"]

    def test_extract_error_codes(self):
        text = "ERR-5002: Database connection failed. FATAL-1001 occurred."
        entities = EntityExtractor.extract(text)
        assert any("ERR5002" in e or "ERR-5002" in e for e in entities["error_codes"])

    def test_extract_timestamps(self):
        text = "Incident at 2026-04-15T14:32:00Z was critical."
        entities = EntityExtractor.extract(text)
        assert any("2026-04-15" in t for t in entities["timestamps"])

    def test_extract_severity(self):
        text = "This is a critical incident with major impact on production."
        entities = EntityExtractor.extract(text)
        assert "critical" in entities["severities"]
        assert "major" in entities["severities"]

    def test_summarize(self):
        text = "This is the first sentence. This is the second one about an incident. And a third here. Fourth too."
        summary = EntityExtractor.summarize(text, max_sentences=2)
        assert "first sentence" in summary
        assert "second one" in summary
        assert "Fourth" not in summary
