"""Tool facade for aspect-aware retrieval over one movie's reviews."""

from typing import Any

from app.rag.models import QueryOpinionsInput
from app.rag.retrieval import OpinionRetriever
from app.tools.core.tool import Tool


class QueryMovieOpinions(Tool):
    name = "query_movie_opinions"
    description = (
        "Retrieve and rerank evidence about a question or aspect from a "
        "temporary single-movie review index. It searches IMDb and Douban "
        "separately and returns a lightweight retrieval identifier plus "
        "coverage metadata."
    )
    parameters = QueryOpinionsInput.model_json_schema()

    def __init__(self, retriever: OpinionRetriever) -> None:
        self.retriever = retriever

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        value = QueryOpinionsInput.model_validate(params)
        artifact = await self.retriever.retrieve(
            index_id=value.index_id,
            question=value.question,
            aspect=value.aspect,
            platforms=value.platforms,
        )
        return artifact.model_dump(mode="json")


__all__ = ["QueryMovieOpinions"]
