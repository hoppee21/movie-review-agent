"""Dependency owner for one process-local movie-opinion RAG capability."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.rag.aggregation import LLMOpinionAggregator, OpinionAggregator
from app.rag.embedding import LangChainEmbeddingProvider
from app.rag.index import EphemeralRagStore, OpinionIndexBuilder
from app.rag.reranker import AliyunReranker
from app.rag.retrieval import (
    LLMRetrievalQueryPlanner,
    OpinionRetriever,
)
from config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    DEFAULT_RAG_CONFIG,
    RagConfig,
    load_openai_api_key,
)


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


def build_openai_chat_model(
    model: str, *, api_key: str, reasoning_effort: str | None = None, **kwargs
) -> ChatOpenAI:
    """Give the current mini model an explicit reasoning budget; callers can choose none."""

    if reasoning_effort is None and (
        model == "gpt-5.4-mini" or model.startswith("gpt-5.4-mini-")
    ):
        reasoning_effort = "medium"
    if "timeout" not in kwargs and "request_timeout" not in kwargs:
        kwargs["timeout"] = API_TIMEOUT_SECONDS
    kwargs.setdefault("max_retries", API_MAX_RETRIES)
    return ChatOpenAI(
        model=model, api_key=api_key, reasoning_effort=reasoning_effort, **kwargs
    )


def build_openai_rag_runtime(
    llm: BaseChatModel,
    *,
    config: RagConfig = DEFAULT_RAG_CONFIG,
    api_key: str | None = None,
    rerank_api_key: str | None = None,
) -> RagRuntime:
    """Wire OpenAI embeddings/generation and Aliyun reranking without API calls."""

    embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=api_key or load_openai_api_key(),
        request_timeout=API_TIMEOUT_SECONDS,
        max_retries=API_MAX_RETRIES,
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
        AliyunReranker(
            model=config.rerank_model,
            endpoint=config.rerank_endpoint,
            api_key=rerank_api_key,
        ),
        config,
    )
    aggregator = LLMOpinionAggregator(llm)
    return RagRuntime(
        config=config,
        store=store,
        index_builder=index_builder,
        retriever=retriever,
        aggregator=aggregator,
    )


__all__ = ["RagRuntime", "build_openai_chat_model", "build_openai_rag_runtime"]
