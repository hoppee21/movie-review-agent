"""Resolve a movie and its IMDb/Douban URLs through Wikidata."""

import asyncio
import json
import re
import time
import unicodedata
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.tools.core.tool import Tool


class WikidataQueryService(Tool):
    """Find a movie by name or identifier and link its platform pages."""

    name = "resolve_movie"
    description = (
        "Find a movie from an English title or alias, IMDb ID, Douban URL, or "
        "Wikidata ID. Return the matching IMDb and Douban URLs. The Agent must "
        "translate or rewrite non-English user input before calling this tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "movie_query": {
                "type": "string",
                "description": (
                    "An English movie title or alias, IMDb ID/URL, Douban URL, "
                    "or Wikidata ID. Examples: Home Alone 4, tt0329200."
                ),
            },
            "release_year": {
                "type": "integer",
                "description": "Optional release year used to disambiguate names.",
            },
        },
        "required": ["movie_query"],
        "additionalProperties": False,
    }

    endpoint = "https://query.wikidata.org/sparql"
    user_agent = "MovieReviewAgent/1.0 (Wikidata movie resolver)"
    timeout_seconds = 30
    max_attempts = 3

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Resolve a movie without blocking the Agent event loop."""

        movie_query = params.get("movie_query")
        if not isinstance(movie_query, str) or not movie_query.strip():
            raise ValueError("movie_query must be a non-empty string")

        release_year = params.get("release_year")
        if release_year is not None and (
            isinstance(release_year, bool)
            or not isinstance(release_year, int)
            or not 1888 <= release_year <= 2100
        ):
            raise ValueError("release_year must be an integer from 1888 to 2100")

        return await asyncio.to_thread(
            self._resolve_movie,
            movie_query.strip(),
            release_year,
        )

    def _resolve_movie(
        self,
        movie_query: str,
        release_year: int | None,
    ) -> dict[str, Any]:
        identifier = self._parse_identifier(movie_query)
        query = (
            self._build_identifier_query(*identifier)
            if identifier
            else self._build_search_query(movie_query, release_year)
        )
        candidates = self._parse_candidates(self._request_results(query))

        if not candidates:
            return {
                "status": "not_found",
                "query": movie_query,
                "candidates": [],
            }

        if identifier or len(candidates) == 1:
            return self._resolved_result(movie_query, candidates[0])

        exact_matches = [
            candidate
            for candidate in candidates
            if self._normalize_name(candidate["title"])
            == self._normalize_name(movie_query)
        ]
        if len(exact_matches) == 1:
            return self._resolved_result(movie_query, exact_matches[0])

        return {
            "status": "needs_clarification",
            "query": movie_query,
            "candidates": candidates,
        }

    @staticmethod
    def _resolved_result(query: str, movie: dict[str, Any]) -> dict[str, Any]:
        return {"status": "resolved", "query": query, **movie}

    @classmethod
    def _parse_candidates(
        cls,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row in rows:
            try:
                wikidata_id = row["movie"]["value"].rsplit("/", 1)[-1]
                imdb_id = row["imdbId"]["value"]
                douban_id = row["doubanId"]["value"]
            except (KeyError, TypeError):
                continue

            if wikidata_id in seen:
                continue
            seen.add(wikidata_id)

            label = row.get("movieLabel", {}).get("value", wikidata_id)
            description = row.get("movieDescription", {}).get("value")
            release_date = row.get("releaseDate", {}).get("value", "")
            year_match = re.match(r"(\d{4})", release_date)

            candidate = {
                "wikidata_id": wikidata_id,
                "title": label,
                "release_year": (
                    int(year_match.group(1)) if year_match else None
                ),
                "imdb_url": f"https://www.imdb.com/title/{imdb_id}/",
                "douban_url": (
                    f"https://movie.douban.com/subject/{douban_id}/"
                ),
            }
            if description:
                candidate["description"] = description
            candidates.append(candidate)

        return candidates

    def _request_results(self, query: str) -> list[dict[str, Any]]:
        url = f"{self.endpoint}?{urlencode({'query': query, 'format': 'json'})}"
        request = Request(
            url,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent,
            },
        )

        for attempt in range(self.max_attempts):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["results"]["bindings"]
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_attempts - 1:
                    raise RuntimeError(
                        f"Wikidata Query Service returned HTTP {exc.code}"
                    ) from exc
                time.sleep(self._retry_delay(exc, attempt))
            except (TimeoutError, URLError) as exc:
                if attempt == self.max_attempts - 1:
                    raise RuntimeError("Wikidata Query Service request failed") from exc
                time.sleep(0.5 * (2**attempt))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError("Wikidata Query Service returned invalid JSON") from exc

        raise RuntimeError("Wikidata Query Service request failed")

    @staticmethod
    def _retry_delay(error: HTTPError, attempt: int) -> float:
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return 0.5 * (2**attempt)

    @classmethod
    def _build_identifier_query(cls, property_id: str, value: str) -> str:
        if property_id == "qid":
            selector = f"VALUES ?movie {{ wd:{value} }}"
        else:
            escaped_value = cls._escape_sparql(value)
            selector = f'?movie wdt:{property_id} "{escaped_value}".'

        return f"""
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT
          ?movie ?movieLabel ?movieDescription
          ?releaseDate ?imdbId ?doubanId
        WHERE {{
          {selector}
          ?movie wdt:P345 ?imdbId;
                 wdt:P4529 ?doubanId.
          OPTIONAL {{ ?movie wdt:P577 ?releaseDate. }}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en".
          }}
        }}
        LIMIT 5
        """.strip()

    @classmethod
    def _build_search_query(
        cls,
        movie_query: str,
        release_year: int | None,
    ) -> str:
        search = cls._escape_sparql(movie_query)
        if release_year is None:
            release_clause = "OPTIONAL { ?movie wdt:P577 ?releaseDate. }"
        else:
            release_clause = (
                "?movie wdt:P577 ?releaseDate. "
                f"FILTER(YEAR(?releaseDate) = {release_year})"
            )

        return f"""
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>

        SELECT DISTINCT
          ?rank ?movie ?movieLabel ?movieDescription
          ?releaseDate ?imdbId ?doubanId
        WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search "{search}";
                            mwapi:language "en";
                            mwapi:limit "10".
            ?movie wikibase:apiOutputItem mwapi:item.
            ?rank wikibase:apiOrdinal true.
          }}
          ?movie wdt:P345 ?imdbId;
                 wdt:P4529 ?doubanId.
          {release_clause}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en".
          }}
        }}
        ORDER BY ?rank
        LIMIT 5
        """.strip()

    @staticmethod
    def _parse_identifier(value: str) -> tuple[str, str] | None:
        imdb_match = re.search(r"\b(tt\d{7,10})\b", value, re.IGNORECASE)
        if imdb_match:
            return "P345", imdb_match.group(1).lower()

        douban_match = re.search(
            r"(?:movie\.douban\.com/subject/|douban\s*[:#]\s*)(\d+)",
            value,
            re.IGNORECASE,
        )
        if douban_match:
            return "P4529", douban_match.group(1)

        wikidata_match = re.search(
            r"(?:wikidata\.org/(?:wiki/|entity/))?\b(Q\d+)\b",
            value,
            re.IGNORECASE,
        )
        if wikidata_match:
            return "qid", wikidata_match.group(1).upper()

        return None

    @staticmethod
    def _normalize_name(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(normalized.split())

    @staticmethod
    def _escape_sparql(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
