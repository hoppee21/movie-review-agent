"""Collect IMDb and Douban reviews from Wikidata-resolved URLs."""

from pathlib import Path
import logging
from typing import Any

from app.tools.core.get_douban import (
    collect_douban_reviews,
    parse_douban_id,
)
from app.tools.core.get_imdb import collect_imdb_reviews, parse_imdb_id
from app.tools.core.tool import Tool
from app.progress import progress


REVIEWS_PER_PLATFORM = 400
OUTPUT_DIR = Path("movie_reviews")
logger = logging.getLogger(__name__)


class CollectMovieReviews(Tool):
    name = "collect_movie_reviews"
    description = (
        "Given the imdb_url and douban_url returned by resolve_movie, "
        "collect up to 400 reviews from each platform and return their JSON "
        "file paths and counts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "imdb_url": {
                "type": "string",
                "description": "The IMDb title URL returned by Wikidata.",
            },
            "douban_url": {
                "type": "string",
                "description": "The Douban subject URL returned by Wikidata.",
            },
        },
        "required": ["imdb_url", "douban_url"],
        "additionalProperties": False,
    }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        return await collect_movie_reviews(
            imdb_url=str(params.get("imdb_url")),
            douban_url=str(params.get("douban_url")),
        )


async def collect_movie_reviews(
    imdb_url: str,
    douban_url: str,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run both collectors sequentially and return lightweight artifacts."""

    imdb_id = parse_imdb_id(imdb_url)
    douban_id = parse_douban_id(douban_url)
    output_dir.mkdir(parents=True, exist_ok=True)

    imdb_path = output_dir / f"imdb_{imdb_id}.json"
    douban_path = output_dir / f"douban_{douban_id}.json"

    with progress("采集 IMDb 评论"):
        imdb_reviews = await collect_imdb_reviews(imdb_url, imdb_path, REVIEWS_PER_PLATFORM)
    logger.info("IMDb 已保存 %d 条评论", len(imdb_reviews))
    with progress("采集豆瓣评论"):
        douban_reviews = await collect_douban_reviews(douban_url, douban_path, REVIEWS_PER_PLATFORM)
    logger.info("豆瓣已保存 %d 条评论；接下来进入评论分析", len(douban_reviews))

    return {
        "imdb": {
            "url": imdb_url,
            "output_path": str(imdb_path.resolve()),
            "review_count": len(imdb_reviews),
        },
        "douban": {
            "url": douban_url,
            "output_path": str(douban_path.resolve()),
            "review_count": len(douban_reviews),
        },
    }
