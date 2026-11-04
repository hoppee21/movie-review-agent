"""Score original review passages with Aliyun's dedicated reranking API."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Protocol

import httpx

from app.progress import progress
from app.rag.models import RankedPassage, RetrievalCandidate
from app.run_log import record_event
from config import DEFAULT_RAG_CONFIG, load_dashscope_api_key


class PassageReranker(Protocol):
    async def rerank(
        self,
        *,
        question: str,
        aspect: str | None,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RankedPassage]: ...


class AliyunReranker:
    """One request per platform: scores are comparable only within that request."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_RAG_CONFIG.rerank_model,
        endpoint: str = DEFAULT_RAG_CONFIG.rerank_endpoint,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout <= 0 or max_retries < 0:
            raise ValueError("rerank timeout must be positive and retries non-negative")
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key.strip() if api_key else None
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport

    async def rerank(
        self,
        *,
        question: str,
        aspect: str | None,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RankedPassage]:
        if not candidates:
            return []
        groups: dict[str, list[RetrievalCandidate]] = {}
        for item in candidates:
            groups.setdefault(item.platform, []).append(item)
        for batch in groups.values():
            if len(batch) > 500:
                raise ValueError(
                    "Aliyun rerank supports at most 500 candidates per platform request"
                )
            if len({item.review_id for item in batch}) != len(batch):
                raise ValueError(
                    "Rerank candidates must contain each platform review once"
                )
        query = question.strip()
        if aspect and aspect.strip():
            query += f"\nFocus: {aspect.strip()}"
        if not query:
            raise ValueError("rerank query cannot be empty")
        api_key = self._api_key or load_dashscope_api_key()
        async with httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as client:
            # Wait for every in-flight request before closing the shared client.
            results = await asyncio.gather(
                *(
                    self._rank_platform(client, query, platform, batch)
                    for platform, batch in groups.items()
                ),
                return_exceptions=True,
            )
        ranked = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            ranked.extend(result)
        return ranked

    async def _rank_platform(self, client, query, platform, candidates):
        documents = [
            "\n\n[Separate passage from the same review]\n\n".join(item.contexts)
            for item in candidates
        ]
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": len(documents), "return_documents": False},
        }
        record_event(
            "rag.rerank_input",
            platform=platform,
            model=self.model,
            candidate_ids=[item.chunk_id for item in candidates],
            input=payload,
        )
        with progress(f"阿里云重排 {platform}（{len(candidates)} 条评论）"):
            response = await self._request(client, payload)
            try:
                raw = response.json()
            except ValueError:
                raise ValueError("Aliyun reranker returned invalid JSON") from None
            scores = _validated_scores(raw, len(candidates))
        record_event(
            "rag.rerank_output",
            platform=platform,
            results=[
                {"chunk_id": item.chunk_id, "score": scores[index]}
                for index, item in enumerate(candidates)
            ],
        )
        ranked = [
            RankedPassage(
                **item.model_dump(exclude={"source_id", "rerank_score"}),
                source_id=f"source:{item.platform}:{item.review_id}",
                rerank_score=scores[index],
            )
            for index, item in enumerate(candidates)
        ]
        return sorted(
            ranked,
            key=lambda item: (-item.rerank_score, -item.fusion_score, item.chunk_id),
        )

    async def _request(self, client, payload):
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(self.endpoint, json=payload)
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise RuntimeError(
                        "Aliyun reranker connection failed or timed out"
                    ) from None
            else:
                if response.is_success:
                    return response
                retryable = (
                    response.status_code in {408, 429} or response.status_code >= 500
                )
                if not retryable or attempt == self._max_retries:
                    # Provider error bodies may contain input or credentials; do not echo them.
                    raise RuntimeError(
                        f"Aliyun reranker request failed (HTTP {response.status_code})"
                    )
            await asyncio.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")


def _validated_scores(raw, expected_count: int) -> dict[int, float]:
    output = raw.get("output") if isinstance(raw, dict) else None
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list) or len(results) != expected_count:
        raise ValueError(
            "Aliyun reranker must return each candidate index exactly once"
        )
    scores = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Aliyun reranker returned an invalid result")
        index, score = item.get("index"), item.get("relevance_score")
        if type(index) is not int or not 0 <= index < expected_count or index in scores:
            raise ValueError(
                "Aliyun reranker must return each candidate index exactly once"
            )
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError(
                "Aliyun reranker relevance_score must be finite and in [0, 1]"
            )
        scores[index] = float(score)
    return scores


__all__ = ["AliyunReranker", "PassageReranker"]
