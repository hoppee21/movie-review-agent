"""Tool facade for building one transient movie-review index."""

from typing import Any

from app.rag.index import OpinionIndexBuilder
from app.rag.models import BuildIndexInput
from app.tools.core.tool import Tool


class BuildOpinionIndex(Tool):
    name = "build_opinion_index"
    description = (
        "Build a temporary hybrid-search index from the IMDb and Douban "
        "review JSON files collected for one resolved movie. Returns a "
        "lightweight index identifier; the full index remains in memory."
    )
    parameters = BuildIndexInput.model_json_schema()

    def __init__(self, builder: OpinionIndexBuilder) -> None:
        self.builder = builder

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        value = BuildIndexInput.model_validate(params)
        artifact = await self.builder.build(
            value.imdb_reviews_path,
            value.douban_reviews_path,
        )
        return artifact.model_dump(mode="json")


__all__ = ["BuildOpinionIndex"]
