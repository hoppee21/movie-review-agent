"""Resolve a movie and its IMDb/Douban URLs through Wikidata."""

import asyncio
import json
import re
import time
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.tools.core.tool import Tool


class WikidataQueryService(Tool):
    """Find a movie by name or identifier and link its platform pages."""

    name = "resolve_movie"
    description = (
        "Verify movie identity from a Chinese/English name or platform ID. "
        "Return real version/series candidates for confirmation; platform links "
        "are optional and do not determine whether a movie exists."
    )
    parameters = {
        "type": "object",
        "properties": {
            "movie_query": {
                "type": "string",
                "description": (
                    "User-provided movie name, description, IMDb/Douban URL or Wikidata ID."
                ),
            },
            "release_year": {
                "type": "integer",
                "description": "Only a year explicitly provided or confirmed by the user.",
            },
            "installment": {
                "type": "integer",
                "description": "Only a sequel number explicitly provided or confirmed by the user.",
            },
            "search_hints": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "Unverified name hints used only when the original query finds no match.",
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

        installment = params.get("installment")
        if installment is not None and (
            isinstance(installment, bool)
            or not isinstance(installment, int)
            or not 1 <= installment <= 100
        ):
            raise ValueError("installment must be an integer from 1 to 100")
        hints = params.get("search_hints", [])
        if (
            not isinstance(hints, list)
            or len(hints) > 3
            or any(not isinstance(hint, str) for hint in hints)
        ):
            raise ValueError("search_hints must contain at most three strings")
        return await asyncio.to_thread(
            self._resolve_movie, movie_query.strip(), release_year, installment, hints
        )

    def _resolve_movie(
        self,
        movie_query: str,
        release_year: int | None,
        installment: int | None = None,
        hints: Sequence[str] = (),
    ) -> dict[str, Any]:
        identifier = self._parse_identifier(movie_query)
        rows = []
        if identifier:
            rows = self._request_results(self._build_identifier_query(*identifier))
        else:
            for name in dict.fromkeys([movie_query, *hints]):
                if not name.strip() or self._parse_identifier(name):
                    continue
                title = re.fullmatch(r"《([^《》]+)》", name.strip())
                if title:
                    name = title.group(1)
                rows = self._request_results(self._build_search_query(name))
                if rows:
                    break
        candidates = self._parse_candidates(rows)
        series = {
            item["wikidata_id"]: item["title"]
            for item in candidates if item["entity_type"] == "series"
        }
        for item in candidates:
            series.update(item["series"])
        if series and not identifier:
            rows.extend(self._request_results(self._build_series_query(list(series)[:5])))
            candidates = self._parse_candidates(rows)
        movies = [item for item in candidates if item["entity_type"] == "movie"]
        if identifier and candidates and not movies:
            return {
                "status": "unsupported",
                "query": movie_query,
                "reason": "当前仅支持单部电影，暂不支持整个系列。",
                "candidates": candidates,
            }

        if release_year is not None:
            movies = [
                item for item in movies
                if not item["release_years"] or release_year in item["release_years"]
            ]
        ordinal_matches = [
            item for item in movies if installment in item["installments"]
        ] if installment is not None else []
        if ordinal_matches:
            movies = ordinal_matches

        if len(movies) == 1:
            item = movies[0]
            year_matches = release_year is None or release_year in item["release_years"]
            installment_matches = installment is None or installment in item["installments"]
            has_constraint = (
                identifier is not None or release_year is not None or installment is not None
            )
            if year_matches and installment_matches and has_constraint:
                return self._resolved_result(movie_query, item)

        options = sorted(movies, key=lambda item: (item["release_year"] or 9999, item["title"]))
        if release_year is None and installment is None:
            options.extend(
                {"wikidata_id": qid, "title": title, "entity_type": "series"}
                for qid, title in series.items()
            )

        if not options:
            return {"status": "not_found", "query": movie_query, "candidates": []}

        return {
            "status": "needs_clarification",
            "query": movie_query,
            "candidates": options,
            "question": "请确认具体电影及版本；若指整个系列，当前暂不支持联合处理。",
        }

    @staticmethod
    def _resolved_result(query: str, movie: dict[str, Any]) -> dict[str, Any]:
        return {"status": "resolved", "query": query, **movie}

    @staticmethod
    def _parse_candidates(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}

        for row in rows:
            try:
                wikidata_id = row["movie"]["value"].rsplit("/", 1)[-1]
            except (KeyError, TypeError):
                continue

            if not re.fullmatch(r"Q[1-9]\d*", wikidata_id):
                continue

            label = row.get("movieLabel", {}).get("value", wikidata_id)
            description = row.get("movieDescription", {}).get("value")
            release_date = row.get("releaseDate", {}).get("value", "")
            year_match = re.match(r"(\d{4})", release_date)

            candidate = candidates.setdefault(wikidata_id, {
                "wikidata_id": wikidata_id,
                "title": label,
                "entity_type": "series" if row.get("isSeries", {}).get("value") in {"true", "1"} else "movie",
                "release_years": [], "installments": [], "series": {},
                "release_year": None, "imdb_url": None, "douban_url": None,
            })
            if year_match:
                candidate["release_years"] = sorted(set([*candidate["release_years"], int(year_match.group(1))]))
                candidate["release_year"] = candidate["release_years"][0]
            for field, template in (("imdbId", "https://www.imdb.com/title/{}/"), ("doubanId", "https://movie.douban.com/subject/{}/")):
                if value := row.get(field, {}).get("value"):
                    candidate["imdb_url" if field == "imdbId" else "douban_url"] = template.format(value)
            if series_id := row.get("series", {}).get("value", "").rsplit("/", 1)[-1]:
                if re.fullmatch(r"Q[1-9]\d*", series_id):
                    candidate["series"][series_id] = row.get("seriesLabel", {}).get("value", series_id)
            ordinal = row.get("ordinal", {}).get("value", "")
            if ordinal.isdigit():
                candidate["installments"] = sorted(set([*candidate["installments"], int(ordinal)]))
            if description:
                candidate["description"] = description
        return list(candidates.values())

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
                rows = payload["results"]["bindings"]
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError("invalid result rows")
                return rows
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_attempts - 1:
                    raise RuntimeError(
                        f"电影资料查询暂不可用（HTTP {exc.code}），请稍后重试。"
                    ) from exc
                time.sleep(self._retry_delay(exc, attempt))
            except (TimeoutError, URLError) as exc:
                if attempt == self.max_attempts - 1:
                    raise RuntimeError("电影资料查询暂不可用，无法核实作品，请稍后重试。") from exc
                time.sleep(0.5 * (2**attempt))
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError("电影资料服务返回了无效数据，请稍后重试。") from exc

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

        return cls._candidate_query(selector)

    @classmethod
    def _build_search_query(
        cls,
        movie_query: str,
    ) -> str:
        search = cls._escape_sparql(movie_query)
        language = "zh" if re.search(r"[\u3400-\u9fff]", movie_query) else "en"
        # Limit entities before metadata expansion, not rows of release dates.
        selector = f"""
        {{ SELECT ?movie (MIN(?searchOrder) AS ?rank) WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search "{search}";
                            mwapi:language "{language}";
                            mwapi:limit "20".
            ?movie wikibase:apiOutputItem mwapi:item.
            ?searchOrder wikibase:apiOrdinal true.
          }}
        }} GROUP BY ?movie ORDER BY ?rank LIMIT 20 }}
        """
        return cls._candidate_query(selector)

    @classmethod
    def _build_series_query(cls, ids: list[str]) -> str:
        values = " ".join(f"wd:{qid}" for qid in ids if re.fullmatch(r"Q[1-9]\d*", qid))
        return cls._candidate_query(f"""
        {{ SELECT DISTINCT ?movie WHERE {{
          VALUES ?group {{ {values} }}
          {{ ?movie wdt:P179 ?group. }} UNION {{ ?group wdt:P527 ?movie. }}
          ?movie wdt:P31/wdt:P279* wd:Q11424.
        }} LIMIT 100 }}
        """)

    @staticmethod
    def _candidate_query(selector: str) -> str:
        return f"""
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
        SELECT DISTINCT ?rank ?movie ?movieLabel ?movieDescription ?releaseDate
                        ?imdbId ?doubanId ?series ?seriesLabel ?ordinal ?isSeries
        WHERE {{
          {selector}
          VALUES ?kind {{ wd:Q11424 wd:Q24862 }}
          ?movie wdt:P31/wdt:P279* ?kind.
          BIND(?kind = wd:Q24862 AS ?isSeries)
          OPTIONAL {{ ?movie wdt:P345 ?imdbId. }}
          OPTIONAL {{ ?movie wdt:P4529 ?doubanId. }}
          OPTIONAL {{ ?movie wdt:P577 ?releaseDate. }}
          OPTIONAL {{
            ?movie p:P179 ?membership.
            ?membership ps:P179 ?series.
            OPTIONAL {{ ?membership pq:P1545 ?ordinal. }}
          }}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en,zh".
          }}
        }}
        ORDER BY ?rank ?releaseDate
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
    def _escape_sparql(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
