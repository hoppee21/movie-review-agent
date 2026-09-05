"""Dependency owner for one process-local movie-opinion RAG capability."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings

from app.rag.aggregation import LLMOpinionAggregator, OpinionAggregator
from app.rag.embedding import LangChainEmbeddingProvider
from app.rag.index import EphemeralRagStore, OpinionIndexBuilder
from app.rag.retrieval import (
    LLMHydeGenerator,
    LLMRetrievalQueryPlanner,
    OpinionRetriever,
)
from app.rag.reranker import LLMOpinionReranker
from config import DEFAULT_RAG_CONFIG, RagConfig, load_openai_api_key


@dataclass(slots=True)
class RagRuntime:
    """Keep heavy transient data out of LangGraph state and action handlers."""

    config: RagConfig
    store: EphemeralRagStore
    index_builder: OpinionIndexBuilder
    retriever: OpinionRetriever
    aggregator: OpinionAggregator

    def clear(self) -> None:
        self.store.clear()


def build_openai_rag_runtime(
    llm: BaseChatModel,
    *,
    config: RagConfig = DEFAULT_RAG_CONFIG,
    api_key: str | None = None,
) -> RagRuntime:
    """Wire OpenAI embeddings and LLM stages without making an API call."""

    embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=api_key or load_openai_api_key(),
    )
    embedder = LangChainEmbeddingProvider(
        embeddings,
        name=config.embedding_model,
        batch_size=config.embedding_batch_size,
    )
    store = EphemeralRagStore()
    index_builder = OpinionIndexBuilder(store, embedder, config)
    retriever = OpinionRetriever(
        store,
        embedder,
        LLMRetrievalQueryPlanner(llm),
        LLMOpinionReranker(llm, batch_size=config.rerank_batch_size),
        LLMHydeGenerator(llm),
        config,
    )
    aggregator = LLMOpinionAggregator(
        llm,
        evidence_per_platform=config.final_evidence_per_platform,
    )
    return RagRuntime(
        config=config,
        store=store,
        index_builder=index_builder,
        retriever=retriever,
        aggregator=aggregator,
    )


__all__ = ["RagRuntime", "build_openai_rag_runtime"]
