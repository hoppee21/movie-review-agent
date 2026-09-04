"""Collect popular short comments from Douban."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import Locator, Page, async_playwright

from config import load_source_cookie
from app.tools.core.tool import Tool


DOUBAN_ID = "1295644"
TARGET_REVIEWS = 400
OUTPUT_FILE = "douban_top400_reviews.json"

PAGE_SIZE = 20
PAGE_DELAY_SECONDS = 10


class GetDoubanReviews(Tool):
    name = "get_douban_reviews"
    description = (
        "Collect up to 400 popular short reviews for a Douban movie URL, save "
        "their full text to JSON, and return the path and review count."
    )
    parameters = {
        "type": "object",
        "properties": {
            "douban_url": {
                "type": "string",
                "description": "A Douban movie URL such as "
                "https://movie.douban.com/subject/1295644/.",
            }
        },
        "required": ["douban_url"],
        "additionalProperties": False,
    }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        douban_url = str(params.get("douban_url"))
        subject_id = parse_douban_id(douban_url)
        output_path = Path(f"douban_reviews_{subject_id}.json")
        reviews = await collect_douban_reviews(douban_url, output_path)
        return {
            "output_path": str(output_path.resolve()),
            "review_count": len(reviews),
        }


def parse_douban_id(douban_url: Any) -> str:
    if not isinstance(douban_url, str):
        raise ValueError("douban_url must be a string")

    match = re.fullmatch(
        r"https?://movie\.douban\.com/subject/([1-9]\d*)/?(?:\?.*)?",
        douban_url.strip(),
    )
    if not match:
        raise ValueError("Invalid Douban movie URL")
    return match.group(1)


async def collect_douban_reviews(
    douban_url: str,
    output_path: Path,
    limit: int = TARGET_REVIEWS,
) -> list[dict[str, Any]]:
    """Start at the first popular-comments page and save after each page."""

    subject_id = parse_douban_id(douban_url)
    if not 1 <= limit <= TARGET_REVIEWS:
        raise ValueError(f"limit must be between 1 and {TARGET_REVIEWS}")

    comments: list[dict[str, Any]] = []
    seen: set[str] = set()
    save_comments(output_path, comments)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context(locale="zh-CN")

        cookie = load_source_cookie("douban")
        if cookie:
            await context.add_cookies(parse_cookie_header(cookie))

        page = context.pages[0] if context.pages else await context.new_page()

        try:
            for start in range(0, limit, PAGE_SIZE):
                response = await page.goto(
                    comments_url(subject_id, start),
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )
                await page.wait_for_timeout(5_000)
                check_access(page)

                if response and response.status >= 400:
                    raise RuntimeError(f"Douban returned HTTP {response.status}")

                items = page.locator(".comment-item")
                count = await items.count()
                if count == 0:
                    raise RuntimeError(f"No comments found at start={start}")

                for index in range(count):
                    comment = await parse_comment(items.nth(index), subject_id)
                    key = comment["comment_id"] or comment["text"]
                    if key in seen:
                        continue
                    seen.add(key)
                    comments.append(comment)
                    if len(comments) == limit:
                        break

                save_comments(output_path, comments)
                print(f"Collected {len(comments)}/{limit}")

                if len(comments) == limit:
                    break
                await asyncio.sleep(PAGE_DELAY_SECONDS)
        finally:
            await browser.close()

    return comments


def comments_url(subject_id: str, start: int) -> str:
    return (
        f"https://movie.douban.com/subject/{subject_id}/comments"
        f"?start={start}&limit={PAGE_SIZE}&status=P&sort=new_score"
    )


async def parse_comment(item: Locator, subject_id: str) -> dict[str, Any]:
    rating_class = await attribute(item.locator(".rating").first, "class")
    rating_match = re.search(r"allstar(\d+)", rating_class or "")
    votes_text = await text(item.locator(".vote-count").first)

    return {
        "comment_id": await item.get_attribute("data-cid"),
        "platform": "douban",
        "subject_id": subject_id,
        "author": await text(item.locator(".comment-info a").first),
        "date": await attribute(item.locator(".comment-time").first, "title"),
        "rating": int(rating_match.group(1)) // 10 if rating_match else None,
        "votes": int(votes_text) if votes_text and votes_text.isdigit() else None,
        "text": (await item.locator(".short").inner_text()).strip(),
    }


async def text(locator: Locator) -> Optional[str]:
    if not await locator.count():
        return None
    value = (await locator.inner_text()).strip()
    return value or None


async def attribute(locator: Locator, name: str) -> Optional[str]:
    if not await locator.count():
        return None
    return await locator.get_attribute(name)


def check_access(page: Page) -> None:
    if "sec.douban.com" in page.url or "accounts.douban.com" in page.url:
        raise RuntimeError(f"Douban redirected to {page.url}")


def parse_cookie_header(header: str) -> list[dict[str, str]]:
    values = {}
    for part in header.split(";"):
        if "=" in part:
            name, value = part.strip().split("=", 1)
            values[name] = value.strip('"')

    return [
        {"name": name, "value": value, "domain": ".douban.com", "path": "/"}
        for name, value in values.items()
    ]


def save_comments(path: Path, comments: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(comments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main() -> None:
    reviews = await collect_douban_reviews(
        f"https://movie.douban.com/subject/{DOUBAN_ID}/",
        Path(OUTPUT_FILE),
        TARGET_REVIEWS,
    )
    print(f"Saved {len(reviews)} reviews to {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
