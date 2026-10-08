"""Embedding boundary used by the transient dense index and unit-test fakes."""

from typing import Protocol

from langchain_core.embeddings import Embeddings
from app.progress import progress


class EmbeddingProvider(Protocol):
    name: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class LangChainEmbeddingProvider:
    """Batch a LangChain Embeddings implementation behind a small contract."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        name: str,
        batch_size: int,
    ) -> None:
        if not name.strip():
            raise ValueError("embedding provider name cannot be empty")
        if batch_size < 1:
            raise ValueError("embedding batch_size must be positive")
        self._embeddings = embeddings
        self.name = name.strip()
        self.batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            end = min(start + self.batch_size, len(texts))
            with progress(f"向量化片段 {start + 1}–{end}/{len(texts)}"):
                vectors.extend(
                    await self._embeddings.aembed_documents(texts[start:end])
                )
        return vectors

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_documents(texts)
