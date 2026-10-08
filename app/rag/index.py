"""Build a small, process-local index for one movie's collected reviews."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np

from app.rag.embedding import EmbeddingProvider
from app.rag.models import (
    IndexArtifact,
    RetrievalResult,
    ReviewChunk,
    ReviewDocument,
    ReviewPlatform,
)
from config import RagConfig


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_WHITESPACE = re.compile(r"\s+")
logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character unigrams/bigrams."""

    tokens: list[str] = []
    for value in _TOKEN_PATTERN.findall(text.casefold()):
        if value.isascii():
            tokens.append(value)
            continue
        characters = list(value)
        tokens.extend(characters)
        tokens.extend(
            characters[index] + characters[index + 1]
            for index in range(len(characters) - 1)
        )
    return tokens


class BM25Index:
    """Minimal BM25 implementation for a transient corpus of at most ~800 reviews."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.term_frequencies = [Counter(tokenize(text)) for text in texts]
        self.document_lengths = [sum(values.values()) for values in self.term_frequencies]
        self.document_frequency: Counter[str] = Counter()
        for values in self.term_frequencies:
            self.document_frequency.update(values.keys())
        self.document_count = len(texts)
        self.average_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

    def search(
        self,
        query: str,
        *,
        allowed_indices: Iterable[int] | None,
        k: int,
        k1: float,
        b: float,
    ) -> list[tuple[int, float]]:
        if k < 1 or not self.document_count:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        indices = list(
            range(self.document_count)
            if allowed_indices is None
            else allowed_indices
        )
        if not indices:
            return []
        if allowed_indices is None:
            document_count = self.document_count
            average_length = self.average_length
            document_frequency = self.document_frequency
        else:
            document_count = len(indices)
            average_length = sum(
                self.document_lengths[index] for index in indices
            ) / document_count
            document_frequency = Counter(
                term
                for index in indices
                for term in self.term_frequencies[index]
            )
        scored: list[tuple[int, float]] = []
        for index in indices:
            frequencies = self.term_frequencies[index]
            length = self.document_lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                term_document_frequency = document_frequency[term]
                inverse_frequency = math.log(
                    1
                    + (document_count - term_document_frequency + 0.5)
                    / (term_document_frequency + 0.5)
                )
                normalizer = frequency + k1 * (
                    1
                    - b
                    + b * length / (average_length or 1.0)
                )
                score += inverse_frequency * frequency * (k1 + 1) / normalizer
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]


@dataclass(slots=True)
class EphemeralOpinionIndex:
    """Dense and sparse data kept outside LangGraph checkpoint state."""

    index_id: str
    corpus_fingerprint: str
    documents: dict[str, ReviewDocument]
    chunks: list[ReviewChunk]
    dense_vectors: np.ndarray
    bm25: BM25Index
    embedding_model: str
    platform_indices: dict[ReviewPlatform, tuple[int, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.dense_vectors.ndim != 2:
            raise ValueError("dense_vectors must be a two-dimensional matrix")
        if self.dense_vectors.shape[0] != len(self.chunks):
            raise ValueError("dense vector count does not match chunk count")
        if not self.platform_indices:
            grouped: dict[ReviewPlatform, list[int]] = defaultdict(list)
            for index, chunk in enumerate(self.chunks):
                grouped[chunk.platform].append(index)
            self.platform_indices = {
                platform: tuple(indices)
                for platform, indices in grouped.items()
            }

class EphemeralRagStore:
    """Own transient indexes and retrieval results for the current process."""

    def __init__(self) -> None:
        self._indexes: dict[str, EphemeralOpinionIndex] = {}
        self._fingerprints: dict[str, str] = {}
        self._retrievals: dict[str, RetrievalResult] = {}

    def put_index(self, index: EphemeralOpinionIndex) -> None:
        self._indexes[index.index_id] = index
        self._fingerprints[index.corpus_fingerprint] = index.index_id

    def find_index(self, corpus_fingerprint: str) -> EphemeralOpinionIndex | None:
        index_id = self._fingerprints.get(corpus_fingerprint)
        return self._indexes.get(index_id) if index_id else None

    def get_index(self, index_id: str) -> EphemeralOpinionIndex:
        try:
            return self._indexes[index_id]
        except KeyError as exc:
            raise KeyError(f"Transient RAG index not found: {index_id}") from exc

    def put_retrieval(self, result: RetrievalResult) -> None:
        self._retrievals[result.retrieval_id] = result

    def get_retrieval(self, retrieval_id: str) -> RetrievalResult:
        try:
            return self._retrievals[retrieval_id]
        except KeyError as exc:
            raise KeyError(
                f"Transient retrieval result not found: {retrieval_id}"
            ) from exc

    def clear(self) -> None:
        self._indexes.clear()
        self._fingerprints.clear()
        self._retrievals.clear()


class OpinionIndexBuilder:
    """Normalize collected review files, chunk IMDb, and embed one movie."""

    def __init__(
        self,
        store: EphemeralRagStore,
        embedder: EmbeddingProvider,
        config: RagConfig,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config

    async def build(
        self,
        imdb_reviews_path: str,
        douban_reviews_path: str,
    ) -> IndexArtifact:
        paths = {
            "imdb": Path(imdb_reviews_path).expanduser(),
            "douban": Path(douban_reviews_path).expanduser(),
        }
        fingerprint = self._fingerprint(paths.values())
        cached = self.store.find_index(fingerprint)
        if cached is not None:
            return self._artifact(cached)

        documents: dict[str, ReviewDocument] = {}
        seen_text: set[tuple[str, str]] = set()
        for platform, path in paths.items():
            for position, raw in enumerate(_load_records(path)):
                document = _normalize_document(raw, platform, position)
                text_key = (
                    document.platform,
                    _WHITESPACE.sub(" ", document.text).strip().casefold(),
                )
                key = _document_key(document)
                if key in documents or text_key in seen_text:
                    continue
                documents[key] = document
                seen_text.add(text_key)

        chunks: list[ReviewChunk] = []
        for document in documents.values():
            document_key = _document_key(document)
            pieces = (
                [document.text]
                if document.platform == "douban"
                else _split_imdb_text(
                    document.text,
                    self.config.imdb_chunk_chars,
                    self.config.imdb_chunk_overlap,
                )
            )
            for position, piece in enumerate(pieces):
                embed_source = (
                    f"{document.title}\n\n{piece}"
                    if document.title
                    else piece
                )
                chunks.append(
                    ReviewChunk(
                        chunk_id=f"{document_key}:chunk-{position + 1}",
                        review_id=document_key,
                        platform=document.platform,
                        text=piece,
                        embed_text=embed_source[: self.config.embed_text_max_chars],
                    )
                )

        logger.info("已整理 %d 条独立评论，切分为 %d 个片段", len(documents), len(chunks))
        vectors = await self.embedder.embed_documents(
            [chunk.embed_text for chunk in chunks]
        )
        matrix = _normalized_matrix(vectors, expected_rows=len(chunks))
        index = EphemeralOpinionIndex(
            index_id=f"movie-index-{uuid4().hex}",
            corpus_fingerprint=fingerprint,
            documents=documents,
            chunks=chunks,
            dense_vectors=matrix,
            bm25=BM25Index([chunk.text for chunk in chunks]),
            embedding_model=self.embedder.name,
        )
        self.store.put_index(index)
        return self._artifact(index)

    def _fingerprint(self, paths: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for path in paths:
            try:
                content = path.read_bytes()
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Review file not found: {path}") from exc
            digest.update(str(path.resolve()).encode())
            digest.update(content)
        digest.update(
            json.dumps(
                {
                    "embedding_model": self.embedder.name,
                    "imdb_chunk_chars": self.config.imdb_chunk_chars,
                    "imdb_chunk_overlap": self.config.imdb_chunk_overlap,
                    "embed_text_max_chars": self.config.embed_text_max_chars,
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()

    @staticmethod
    def _artifact(index: EphemeralOpinionIndex) -> IndexArtifact:
        counts = Counter(document.platform for document in index.documents.values())
        return IndexArtifact(
            index_id=index.index_id,
            corpus_fingerprint=index.corpus_fingerprint,
            review_count=len(index.documents),
            chunk_count=len(index.chunks),
            platform_counts={
                platform: counts.get(platform, 0)
                for platform in ("imdb", "douban")
            },
            embedding_model=index.embedding_model,
        )


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Review file is not valid JSON: {path}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Review file must contain a JSON array: {path}")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"Every review must be a JSON object: {path}")
    return raw


def _normalize_document(
    raw: dict[str, Any],
    platform: ReviewPlatform,
    position: int,
) -> ReviewDocument:
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{platform} review {position + 1} has no text")
    source_id = raw.get("review_id" if platform == "imdb" else "comment_id")
    if not isinstance(source_id, str) or not source_id.strip():
        source_id = hashlib.sha256(text.strip().encode()).hexdigest()[:20]
    known_fields = {
        "review_id",
        "comment_id",
        "platform",
        "text",
        "title",
        "author",
        "date",
        "rating",
        "helpful_votes",
        "total_votes",
        "votes",
    }
    return ReviewDocument(
        review_id=source_id.strip(),
        platform=platform,
        text=text.strip(),
        title=_optional_text(raw.get("title")),
        author=_optional_text(raw.get("author")),
        date=_optional_text(raw.get("date")),
        rating=_optional_number(raw.get("rating")),
        helpful_votes=_optional_integer(
            raw.get("helpful_votes")
            if platform == "imdb"
            else raw.get("votes")
        ),
        total_votes=_optional_integer(raw.get("total_votes")),
        metadata={
            key: value
            for key, value in raw.items()
            if key not in known_fields
        },
    )


def _document_key(document: ReviewDocument) -> str:
    return f"{document.platform}:{document.review_id}"


def _split_imdb_text(text: str, size: int, overlap: int) -> list[str]:
    """Split long reviews near whitespace while keeping bounded overlap."""

    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + size)
        end = hard_end
        if hard_end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, hard_end),
                text.rfind(" ", start + size // 2, hard_end),
            )
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start if next_start < end else end
    return chunks


def _normalized_matrix(
    vectors: Sequence[Sequence[float]],
    *,
    expected_rows: int,
) -> np.ndarray:
    if len(vectors) != expected_rows:
        raise ValueError("embedding count does not match chunk count")
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("embeddings must have one non-zero dimension")
    matrix = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def normalize_query_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_rows: int,
    expected_dimension: int,
) -> np.ndarray:
    matrix = _normalized_matrix(vectors, expected_rows=expected_rows)
    if expected_rows and matrix.shape[1] != expected_dimension:
        raise ValueError("query embedding dimension does not match the index")
    return matrix


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_integer(value: Any) -> int | None:
    number = _optional_number(value)
    return int(number) if number is not None else None


__all__ = [
    "BM25Index",
    "EphemeralOpinionIndex",
    "EphemeralRagStore",
    "OpinionIndexBuilder",
    "normalize_query_vectors",
    "tokenize",
]
