"""Tool facade for evidence-grounded movie-opinion synthesis."""

from typing import Any

from app.rag.aggregation import OpinionAggregator
from app.rag.index import EphemeralRagStore
from app.rag.models import AggregateOpinionsInput
from app.tools.core.tool import Tool


class AggregateMovieOpinions(Tool):
    name = "aggregate_movie_opinions"
    description = (
        "Aggregate a saved retrieval result into a structured, cited movie "
        "opinion answer with viewpoint clusters, platform comparison, "
        "coverage, and sampling limitations."
    )
    parameters = AggregateOpinionsInput.model_json_schema()

    def __init__(
        self,
        store: EphemeralRagStore,
        aggregator: OpinionAggregator,
    ) -> None:
        self.store = store
        self.aggregator = aggregator

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        value = AggregateOpinionsInput.model_validate(params)
        retrieval = self.store.get_retrieval(value.retrieval_id)
        answer = await self.aggregator.aggregate(retrieval)
        return answer.model_dump(mode="json")


__all__ = ["AggregateMovieOpinions"]
