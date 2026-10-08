"""Collect helpful IMDb user reviews."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import Locator, Page, TimeoutError, async_playwright

from config import load_source_cookie
from app.tools.core.tool import Tool


IMDB_ID = "tt0133093"
TARGET_REVIEWS = 400
OUTPUT_FILE = "imdb_top400_reviews.json"

PROFILE_DIR = "imdb_profile"
LOAD_DELAY_SECONDS = 5
LOAD_RETRY_DELAY_MS = 2_000
LOAD_GROWTH_TIMEOUT_MS = 30_000
MAX_IDLE_LOAD_ROUNDS = 3
REVIEW_SELECTOR = "[data-testid='review-card-parent']"
LOAD_BUTTON_SELECTOR = (
    "button[data-testid='load-more-trigger'], "
    "button.ipc-see-more__button"
)


class GetImdbReviews(Tool):
    name = "get_imdb_reviews"
    description = (
        "Collect up to 400 helpful IMDb user reviews for an IMDb title URL, "
        "save their full text to JSON, and return the path and review count."
    )
    parameters = {
        "type": "object",
        "properties": {
            "imdb_url": {
                "type": "string",
                "description": "An IMDb title URL such as "
                "https://www.imdb.com/title/tt0133093/.",
            }
        },
        "required": ["imdb_url"],
        "additionalProperties": False,
    }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        imdb_url = params.get("imdb_url")
        imdb_id = parse_imdb_id(imdb_url)
        output_path = Path(f"imdb_reviews_{imdb_id}.json")
        reviews = await collect_imdb_reviews(str(imdb_url), output_path)
        return {
            "output_path": str(output_path.resolve()),
            "review_count": len(reviews),
        }


def parse_imdb_id(imdb_url: Any) -> str:
    if not isinstance(imdb_url, str):
        raise ValueError("imdb_url must be a string")

    match = re.fullmatch(
        r"https?://(?:www\.)?imdb\.com/title/(tt\d+)/?(?:\?.*)?",
        imdb_url.strip(),
    )
    if not match:
        raise ValueError("Invalid IMDb title URL")
    return match.group(1)


async def collect_imdb_reviews(
    imdb_url: str,
    output_path: Path,
    limit: int = TARGET_REVIEWS,
) -> list[dict[str, Any]]:
    """Start from the first review and save the complete result to JSON."""

    imdb_id = parse_imdb_id(imdb_url)
    if not 1 <= limit <= TARGET_REVIEWS:
        raise ValueError(f"limit must be between 1 and {TARGET_REVIEWS}")

    save_reviews(output_path, [])

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context(locale="en-US")

        cookie = load_source_cookie("imdb")
        if cookie:
            await context.add_cookies(parse_cookie_header(cookie))

        page = await context.new_page()
        try:
            response = await page.goto(
                reviews_url(imdb_url),
                wait_until="domcontentloaded",
                timeout=60_000,
            )
            await page.wait_for_timeout(5_000)

            if response and response.status >= 400:
                raise RuntimeError(f"IMDb returned HTTP {response.status}")
            if not await page.locator(REVIEW_SELECTOR).count():
                raise RuntimeError("IMDb returned no reviews")

            await load_reviews(page, limit)
            reviews = await parse_reviews(page, imdb_id, limit)
        finally:
            await browser.close()

    save_reviews(output_path, reviews)
    return reviews


def reviews_url(imdb_url: str) -> str:
    return (
        f"{imdb_url.rstrip('/')}/reviews/"
        "?sort=totalVotes&dir=desc&ratingFilter=0"
    )


async def load_reviews(page: Page, limit: int) -> int:
    """Load reviews until the target or three consecutive idle rounds."""

    idle_rounds = 0
    while True:
        previous = await page.locator(REVIEW_SELECTOR).count()
        if previous >= limit:
            return previous

        button = await find_load_button(page)
        if button is not None:
            try:
                await button.scroll_into_view_if_needed()
                await button.click(timeout=15_000)
                await page.wait_for_function(
                    f"document.querySelectorAll({REVIEW_SELECTOR!r}).length "
                    f"> {previous}",
                    timeout=LOAD_GROWTH_TIMEOUT_MS,
                )
            except TimeoutError:
                pass

        current = await page.locator(REVIEW_SELECTOR).count()
        if current > previous:
            idle_rounds = 0
            await asyncio.sleep(LOAD_DELAY_SECONDS)
            continue

        idle_rounds += 1
        if idle_rounds >= MAX_IDLE_LOAD_ROUNDS:
            return current
        await wait_before_load_retry(page)


async def find_load_button(page: Page) -> Optional[Locator]:
    buttons = page.locator(LOAD_BUTTON_SELECTOR)
    for index in range(await buttons.count() - 1, -1, -1):
        button = buttons.nth(index)
        if not await button.is_visible():
            continue
        if await button.get_attribute("data-testid") == "load-more-trigger":
            return button
        label = " ".join(
            value
            for value in (
                await button.inner_text(),
                await button.get_attribute("aria-label"),
            )
            if value
        ).strip().lower()
        if (
            "load more" in label
            or "see all" in label
            or "more reviews" in label
            or re.search(r"\d+\s+more", label)
        ):
            return button
    return None


async def wait_before_load_retry(page: Page) -> None:
    """Scroll to the last review so a delayed load button can render."""

    cards = page.locator(REVIEW_SELECTOR)
    if await cards.count():
        await cards.last.scroll_into_view_if_needed()
    await page.wait_for_timeout(LOAD_RETRY_DELAY_MS)


async def parse_reviews(
    page: Page,
    imdb_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    reviews = []
    seen = set()
    cards = page.locator(REVIEW_SELECTOR)

    for index in range(min(await cards.count(), limit)):
        card = cards.nth(index)
        review = await parse_review(card, imdb_id)
        key = review["review_id"] or review["text"]
        if key in seen or not review["text"]:
            continue
        seen.add(key)
        reviews.append(review)

    return reviews


async def parse_review(card: Locator, imdb_id: str) -> dict[str, Any]:
    raw = await card.inner_text()
    helpful = re.search(
        r"([\d,]+)\s+out\s+of\s+([\d,]+)\s+found\s+this\s+helpful",
        raw,
        re.I,
    )
    rating_text = await first_text(
        card,
        "[data-testid='review-rating']",
        ".ipc-rating-star--rating",
    )
    rating = re.search(r"\b(10|[1-9])(?:\.0)?\b", rating_text or "")

    return {
        "review_id": await review_id(card),
        "platform": "imdb",
        "imdb_id": imdb_id,
        "title": await first_text(card, "[data-testid='review-title']"),
        "author": await first_text(card, "[data-testid='author-link']"),
        "date": await first_text(card, "[data-testid='review-date']"),
        "rating": int(rating.group(1)) if rating else None,
        "helpful_votes": to_int(helpful.group(1)) if helpful else None,
        "total_votes": to_int(helpful.group(2)) if helpful else None,
        "text": await first_text(
            card,
            "[data-testid='review-overflow']",
            ".ipc-html-content-inner-div",
            ".ipc-html-content--base",
        ),
    }


async def review_id(card: Locator) -> Optional[str]:
    value = await card.get_attribute("data-review-id")
    if value:
        return value
    link = card.locator("a[href*='/review/rw']").first
    if not await link.count():
        return None
    href = await link.get_attribute("href")
    match = re.search(r"/review/(rw\d+)", href or "")
    return match.group(1) if match else None


async def first_text(card: Locator, *selectors: str) -> Optional[str]:
    for selector in selectors:
        locator = card.locator(selector).first
        if await locator.count():
            value = (await locator.inner_text()).strip()
            if value:
                return value
    return None


def to_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_cookie_header(header: str) -> list[dict[str, str]]:
    values = {}
    for part in header.split(";"):
        if "=" in part:
            name, value = part.strip().split("=", 1)
            values[name] = value.strip('"')
    return [
        {"name": name, "value": value, "domain": ".imdb.com", "path": "/"}
        for name, value in values.items()
    ]


def save_reviews(path: Path, reviews: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main() -> None:
    reviews = await collect_imdb_reviews(
        f"https://www.imdb.com/title/{IMDB_ID}/",
        Path(OUTPUT_FILE),
        TARGET_REVIEWS,
    )
    print(f"Saved {len(reviews)} reviews to {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
