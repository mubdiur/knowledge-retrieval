"""BM25 keyword search over ingested documents.

An in-memory BM25 index rebuilt on startup from the database chunk store.
For production, consider Elasticsearch or Meilisearch instead.
"""

import json
import math
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

DEFAULT_INDEX_PATH = "data/bm25_index.json"


class BM25Index:
    """Simple BM25 implementation for keyword search over chunks.

    Supports incremental updates and persistence to disk.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, index_path: str | None = None):
        self.k1 = k1
        self.b = b
        self._documents: list[dict[str, Any]] = []
        self._doc_terms: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._num_docs: int = 0
        self._ready = False
        self._index_path = Path(index_path or DEFAULT_INDEX_PATH)
        self._load()

    # ── Tokenization ──────────────────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower()
        # Split on non-alphanumeric, keep minimal tokens
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)
        return [t for t in tokens if len(t) > 1]

    # ── Index Management ──────────────────────────────────────────────────────

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build or rebuild the BM25 index from a list of doc dicts with 'content' key."""
        self._documents = documents
        self._doc_terms = []
        total_len = 0

        for doc in documents:
            content = doc.get("content", "")
            tokens = self.tokenize(content)
            self._doc_terms.append(Counter(tokens))
            total_len += len(tokens)

        self._num_docs = len(documents)
        self._avg_doc_len = total_len / max(self._num_docs, 1)

        # Compute IDF
        doc_freq: Counter = Counter()
        for term_counts in self._doc_terms:
            for term in term_counts:
                doc_freq[term] += 1

        self._idf = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log(
                1 + (self._num_docs - df + 0.5) / (df + 0.5)
            )

        self._ready = True
        self._save()
        logger.info("BM25 index built: %d docs, %d terms", self._num_docs, len(self._idf))

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Incrementally add documents to the BM25 index without full rebuild.

        Recomputes IDF and avg doc length to include new documents.
        """
        if not documents:
            return

        new_total_len = self._avg_doc_len * self._num_docs if self._num_docs else 0

        for doc in documents:
            content = doc.get("content", "")
            tokens = self.tokenize(content)
            self._documents.append(doc)
            self._doc_terms.append(Counter(tokens))
            new_total_len += len(tokens)

        self._num_docs = len(self._documents)
        self._avg_doc_len = new_total_len / max(self._num_docs, 1)

        # Recompute IDF with all documents
        doc_freq: Counter = Counter()
        for term_counts in self._doc_terms:
            for term in term_counts:
                doc_freq[term] += 1

        self._idf = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log(
                1 + (self._num_docs - df + 0.5) / (df + 0.5)
            )

        self._ready = True
        self._save()
        logger.info("BM25 index updated: %d docs (+%d), %d terms",
                     self._num_docs, len(documents), len(self._idf))

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist index to disk."""
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "k1": self.k1,
                "b": self.b,
                "documents": self._documents,
                "avg_doc_len": self._avg_doc_len,
                "num_docs": self._num_docs,
                "ready": self._ready,
            }
            self._index_path.write_text(json.dumps(data, default=str), encoding="utf-8")
            logger.debug("BM25 index saved: %d docs", self._num_docs)
        except Exception as e:
            logger.warning("BM25 save failed: %s", e)

    def _load(self) -> None:
        """Load index from disk if it exists."""
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self.k1 = data.get("k1", self.k1)
            self.b = data.get("b", self.b)
            self._documents = data.get("documents", [])
            self._avg_doc_len = data.get("avg_doc_len", 0.0)
            self._num_docs = data.get("num_docs", 0)
            self._ready = data.get("ready", False)

            # Rebuild doc_terms and idf from loaded documents
            self._doc_terms = []
            total_len = 0
            for doc in self._documents:
                tokens = self.tokenize(doc.get("content", ""))
                self._doc_terms.append(Counter(tokens))
                total_len += len(tokens)

            # Recompute IDF
            doc_freq: Counter = Counter()
            for term_counts in self._doc_terms:
                for term in term_counts:
                    doc_freq[term] += 1

            self._idf = {}
            for term, df in doc_freq.items():
                self._idf[term] = math.log(
                    1 + (self._num_docs - df + 0.5) / (df + 0.5)
                )

            self._ready = self._num_docs > 0
            if self._ready:
                logger.info("BM25 index loaded from disk: %d docs, %d terms", self._num_docs, len(self._idf))
        except Exception as e:
            logger.warning("BM25 load failed, starting fresh: %s", e)
            self._documents = []
            self._doc_terms = []
            self._idf = {}
            self._avg_doc_len = 0.0
            self._num_docs = 0
            self._ready = False

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return top-k results with BM25 scores."""
        if not self._ready or not self._num_docs:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i, term_counts in enumerate(self._doc_terms):
            doc_len = sum(term_counts.values())
            score = 0.0
            for term in set(query_tokens):
                if term not in self._idf:
                    continue
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self._avg_doc_len, 1)))
                score += self._idf[term] * (numerator / denominator)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            doc = dict(self._documents[idx])
            doc["score"] = round(score, 4)
            doc["source"] = "bm25"
            results.append(doc)
        return results

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def doc_count(self) -> int:
        return self._num_docs
