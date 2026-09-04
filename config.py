"""Read local credentials relative to the project root, not the working directory."""

import json
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_CONFIG_PATH = PROJECT_ROOT / ".env.local"
COOKIE_CONFIG_PATH = PROJECT_ROOT / "config" / "source_cookies.json"


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
