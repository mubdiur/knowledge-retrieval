"""Adaptive chunker — splits text into semantically meaningful chunks."""

import logging
import re
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AdaptiveChunker:
    """Chunk text using adaptive boundaries rather than fixed-size slices.

    Strategy:
    1. Split on natural boundaries (headings, blank lines, paragraphs)
    2. Merge until minimum size, stop at maximum size
    3. Overlap only at sentence boundaries
    """

    def __init__(
        self,
        max_size: int = 1024,
        min_size: int = 128,
        overlap: int = 128,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into chunks with metadata.

        Returns list of {index, content, token_count, metadata}.
        """
        if not text or not text.strip():
            return []

        segments = self._split_into_segments(text)
        chunks = self._merge_segments(segments)
        return [
            {
                "chunk_index": i,
                "content": chunk,
                "token_count": self._estimate_tokens(chunk),
                "metadata": metadata or {},
            }
            for i, chunk in enumerate(chunks)
        ]

    def _split_into_segments(self, text: str) -> list[str]:
        """Split text at natural boundaries: headings, blank lines, paragraphs."""
        segments = []

        # Try markdown headings first
        if re.search(r"^#{1,6}\s", text, re.MULTILINE):
            parts = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)
            segments.extend(p for p in parts if p.strip())
        else:
            # Split on double newlines (paragraphs)
            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                # If paragraph is too long, split on single newlines
                if len(para) > self.max_size:
                    for line in para.split("\n"):
                        line = line.strip()
                        if line:
                            segments.append(line)
                else:
                    segments.append(para)

        # If no segments found (unlikely), use the whole text
        if not segments:
            segments = [text]

        return segments

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge segments into chunks respecting min/max boundaries."""
        chunks = []
        current = ""

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # If segment alone exceeds max size, split it
            if len(seg) > self.max_size:
                # Flush current first
                if current:
                    chunks.append(current)
                    current = ""
                # Split long segment into smaller chunks at sentence boundaries
                chunks.extend(self._split_long(seg))
                continue

            # If adding this segment would exceed max, start new chunk
            if current and len(current) + len(seg) + 1 > self.max_size:
                if len(current) >= self.min_size:
                    chunks.append(current)
                    # Carry overlap from the end of current
                    current = self._get_overlap(current)
                else:
                    # Current too short, merge anyway
                    current += "\n\n" + seg
                    continue

            if current:
                current += "\n\n" + seg
            else:
                current = seg

        # Flush remaining
        if current and len(current) >= self.min_size:
            chunks.append(current)
        elif current:
            # Too small but non-empty — append to last chunk or keep solo
            if chunks:
                chunks[-1] += "\n\n" + current
            else:
                chunks.append(current)

        return chunks

    def _split_long(self, text: str) -> list[str]:
        """Split a long text into chunks at sentence boundaries.

        Falls back to word-boundary splitting when no sentence endings exist.
        """
        # Try sentence boundaries first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) == 1:
            # No sentence boundaries — split at word boundaries
            words = text.split()
            chunks = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > self.max_size:
                    if current:
                        chunks.append(current)
                    current = word
                else:
                    current = current + " " + word if current else word
            if current:
                chunks.append(current)
            return chunks if chunks else [text[:self.max_size]]

        chunks = []
        current = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) + 1 > self.max_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = current + " " + sent if current else sent
        if current:
            chunks.append(current)
        return chunks if chunks else [text[:self.max_size]]

    def _get_overlap(self, text: str, overlap_chars: int | None = None) -> str:
        """Extract the last N characters for overlap, preferring sentence boundaries."""
        n = overlap_chars or self.overlap
        if len(text) <= n:
            return ""

        overlap_text = text[-n:]
        # Try to find a sentence boundary within the overlap
        match = re.search(r"(?<=[.!?])\s+", overlap_text)
        if match:
            return overlap_text[match.end():]
        return overlap_text

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
