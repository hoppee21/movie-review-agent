"""Read local credentials relative to the project root, not the working directory."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_CONFIG_PATH = PROJECT_ROOT / ".env.local"
COOKIE_CONFIG_PATH = PROJECT_ROOT / "config" / "source_cookies.json"
API_TIMEOUT_SECONDS = 180.0
API_MAX_RETRIES = 1


@dataclass(frozen=True, slots=True)
class RagConfig:
    """Bounded retrieval settings controlled by code rather than the LLM."""

    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    chunk_chars: int = 1800
    chunk_overlap: int = 200
    reading_context_chars: int = 3600
    embed_text_max_chars: int = 2000
    max_query_rewrites: int = 2
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    candidate_cap_per_platform: int = 80
    rerank_model: str = "qwen3.7-text-rerank"
    rerank_endpoint: str = "https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
    final_evidence_per_platform: int = 12
    min_sources_per_platform: int = 6
    max_retrieval_rounds: int = 2

    def __post_init__(self) -> None:
        positive_fields = (
            "embedding_batch_size",
            "chunk_chars",
            "reading_context_chars",
            "embed_text_max_chars",
            "max_query_rewrites",
            "rrf_k",
            "candidate_cap_per_platform",
            "final_evidence_per_platform",
            "min_sources_per_platform",
            "max_retrieval_rounds",
        )
        for field in positive_fields:
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if not self.rerank_model.strip():
            raise ValueError("rerank_model cannot be empty")
        endpoint = urlparse(self.rerank_endpoint)
        if (
            endpoint.scheme != "https"
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
        ):
            raise ValueError("rerank_endpoint must be an HTTPS URL without credentials")
        if not 0 <= self.chunk_overlap < self.chunk_chars:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_chars"
            )
        if self.reading_context_chars < self.chunk_chars:
            raise ValueError("reading_context_chars must be at least chunk_chars")
        if self.max_query_rewrites > 2:
            raise ValueError("max_query_rewrites cannot exceed 2 per platform")
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be positive")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be in [0, 1]")
        if self.min_sources_per_platform > self.final_evidence_per_platform:
            raise ValueError(
                "min_sources_per_platform cannot exceed final_evidence_per_platform"
            )
        if self.candidate_cap_per_platform + self.final_evidence_per_platform > 500:
            raise ValueError(
                "rerank candidates plus retained sources cannot exceed 500 per platform"
            )
        if self.max_retrieval_rounds > 2:
            raise ValueError("max_retrieval_rounds cannot exceed 2")


DEFAULT_RAG_CONFIG = RagConfig()


def load_openai_api_key() -> str:
    """Read the API key without exporting it to the process environment."""

    value = dotenv_values(ENV_CONFIG_PATH).get("OPENAI_API_KEY")
    if not value or not value.strip():
        raise RuntimeError(f"OPENAI_API_KEY is not configured in {ENV_CONFIG_PATH}")
    return value.strip()


def load_dashscope_api_key() -> str:
    """Prefer an exported DashScope key, otherwise use the local backend env file."""

    value = os.getenv("DASHSCOPE_API_KEY") or dotenv_values(ENV_CONFIG_PATH).get(
        "DASHSCOPE_API_KEY"
    )
    if not value or not value.strip():
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not configured; set it in the environment or .env.local"
        )
    return value.strip()


def load_source_cookie(source: str) -> str:
    """Return one source's Cookie header, or an empty string if unset."""

    try:
        config = json.loads(COOKIE_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cookie config not found: {COOKIE_CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Cookie config: {COOKIE_CONFIG_PATH}") from exc

    if not isinstance(config, dict):
        raise RuntimeError(
            f"Cookie config must contain a JSON object: {COOKIE_CONFIG_PATH}"
        )
    value = config.get(source, "")
    if not isinstance(value, str):
        raise RuntimeError(f"Cookie config value must be a string: {source}")
    return value.strip()
