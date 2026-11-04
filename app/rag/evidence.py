"""Select independently sourced passages; source coverage is not semantic proof."""

from collections.abc import Sequence

from app.rag.models import CoverageReport, RankedPassage, ReviewPlatform


def unique_evidence(evidence: Sequence[RankedPassage]) -> list[RankedPassage]:
    """Keep one ranked reading context per review and suppress copied reviews."""
    reviews = set()
    texts = set()
    result = []
    for item in sorted(
        evidence,
        key=lambda item: (-item.rerank_score, -item.fusion_score, item.source_id),
    ):
        review_key = (item.platform, item.review_id)
        text_key = (item.platform, " ".join(item.full_text.casefold().split()))
        if review_key in reviews or text_key in texts or not text_key[1]:
            continue
        reviews.add(review_key)
        texts.add(text_key)
        result.append(item)
    return result


def select_evidence(
    evidence: Sequence[RankedPassage],
    *,
    platforms: Sequence[ReviewPlatform],
    limit_per_platform: int,
) -> list[RankedPassage]:
    """Rank within each platform, using independent review budgets."""
    unique = unique_evidence(evidence)
    return [
        item
        for platform in dict.fromkeys(platforms)
        for item in [e for e in unique if e.platform == platform][:limit_per_platform]
    ]


def coverage_report(
    evidence: Sequence[RankedPassage],
    *,
    platforms: Sequence[ReviewPlatform],
    minimum: int,
) -> CoverageReport:
    """Count selected sources, without treating a rerank score as a factual verdict."""
    unique = unique_evidence(evidence)
    counts = {p: sum(item.platform == p for item in unique) for p in platforms}
    missing = [p for p in platforms if counts[p] < minimum]
    return CoverageReport(
        minimum_reviews=minimum,
        sufficient=not missing,
        source_counts=counts,
        missing_platforms=missing,
        reason="；".join(f"{p}：独立评论不足（{counts[p]}/{minimum}）" for p in missing)
        or None,
    )
