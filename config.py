"""Read local credentials relative to the project root, not the working directory."""

import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_CONFIG_PATH = PROJECT_ROOT / ".env.local"
COOKIE_CONFIG_PATH = PROJECT_ROOT / "config" / "source_cookies.json"


@dataclass(frozen=True, slots=True)
class RagConfig:
    """Bounded retrieval settings controlled by code rather than the LLM."""

    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    imdb_chunk_chars: int = 1800
    imdb_chunk_overlap: int = 200
    embed_text_max_chars: int = 2000
    max_query_rewrites: int = 6
    first_stage_min_k: int = 20
    first_stage_max_k: int = 50
    first_stage_ratio: float = 0.05
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    candidate_cap_per_platform: int = 80
    rerank_batch_size: int = 12
    rerank_top_n_per_platform: int = 40
    final_evidence_per_platform: int = 12
    min_relevant_per_platform: int = 6
    max_retrieval_rounds: int = 2
    hyde_variants: int = 3

    def __post_init__(self) -> None:
        positive_fields = (
            "embedding_batch_size",
            "imdb_chunk_chars",
            "embed_text_max_chars",
            "max_query_rewrites",
            "first_stage_min_k",
            "first_stage_max_k",
            "rrf_k",
            "candidate_cap_per_platform",
            "rerank_batch_size",
            "rerank_top_n_per_platform",
            "final_evidence_per_platform",
            "min_relevant_per_platform",
            "max_retrieval_rounds",
            "hyde_variants",
        )
        for field in positive_fields:
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if not 0 <= self.imdb_chunk_overlap < self.imdb_chunk_chars:
            raise ValueError(
                "imdb_chunk_overlap must be non-negative and smaller than "
                "imdb_chunk_chars"
            )
        if not 0 < self.first_stage_ratio <= 1:
            raise ValueError("first_stage_ratio must be in (0, 1]")
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be positive")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be in [0, 1]")
        if self.first_stage_min_k > self.first_stage_max_k:
            raise ValueError(
                "first_stage_min_k cannot exceed first_stage_max_k"
            )
        if self.rerank_top_n_per_platform > self.candidate_cap_per_platform:
            raise ValueError(
                "rerank_top_n_per_platform cannot exceed "
                "candidate_cap_per_platform"
            )
        if self.final_evidence_per_platform > self.rerank_top_n_per_platform:
            raise ValueError(
                "final_evidence_per_platform cannot exceed "
                "rerank_top_n_per_platform"
            )
        if self.min_relevant_per_platform > self.rerank_top_n_per_platform:
            raise ValueError(
                "min_relevant_per_platform cannot exceed "
                "rerank_top_n_per_platform"
            )
        if self.max_retrieval_rounds > 2:
            raise ValueError("max_retrieval_rounds cannot exceed 2")
        if self.hyde_variants > 3:
            raise ValueError("hyde_variants cannot exceed 3")


DEFAULT_RAG_CONFIG = RagConfig()


def load_openai_api_key() -> str:
    """Read the API key without exporting it to the process environment."""

    value = dotenv_values(ENV_CONFIG_PATH).get("OPENAI_API_KEY")
    if not value or not value.strip():
        raise RuntimeError(f"OPENAI_API_KEY is not configured in {ENV_CONFIG_PATH}")
    return value.strip()


def load_source_cookie(source: str) -> str:
    """Return one source's Cookie header, or an empty string if unset."""

    try:
        config = json.loads(COOKIE_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cookie config not found: {COOKIE_CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Cookie config: {COOKIE_CONFIG_PATH}") from exc

    value = config.get(source, "")
    if not isinstance(value, str):
        raise RuntimeError(f"Cookie config value must be a string: {source}")
    return value.strip()
