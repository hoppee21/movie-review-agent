"""Shared opinion selection and coverage rules, independent of retrieval algorithms."""

import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from app.rag.models import (
    CoverageReport,
    OpinionStance,
    RankedEvidence,
    ReviewPlatform,
)


def opinion_scope(item: RankedEvidence) -> tuple[str, str]:
    # Ignore typographic name variants; semantic aliases still belong to extraction.
    return re.sub(r"[\s\-·.'’]", "", item.target.casefold()), item.subaspect


def unique_evidence(evidence: Sequence[RankedEvidence]) -> list[RankedEvidence]:
    """Deduplicate repeated opinions, while retaining distinct opinions from a review."""

    seen_spans = defaultdict(list)
    quote_sources = {}
    result = []
    for item in sorted(
        evidence,
        key=lambda x: (
            -x.relevance,
            -x.evidence_quality,
            -x.fusion_score,
            x.evidence_id,
        ),
    ):
        span = (item.evidence_span or "").strip()
        if item.relevance < 2 or not span or not item.contains_quote(span):
            continue
        quote = " ".join(span.casefold().split())
        group = (item.platform, *opinion_scope(item), item.stance)
        key = (*group, item.review_id, " ".join(item.opinion.casefold().split()))
        if any(quote in previous or previous in quote for previous in seen_spans[key]):
            continue
        quote_key = (*group, quote)
        if (
            len(quote) >= 64
            and quote_sources.get(quote_key, item.review_id) != item.review_id
        ):
            continue
        seen_spans[key].append(quote)
        quote_sources[quote_key] = item.review_id
        result.append(item)
    return result


def select_evidence(
    evidence: Sequence[RankedEvidence],
    *,
    platforms: Sequence[ReviewPlatform],
    limit_per_platform: int,
    minimum_reviews: int = 1,
) -> list[RankedEvidence]:
    """Favor independent sources and grounded explanations, without topic quotas."""
    relevant = unique_evidence(evidence)
    result = []
    for platform in platforms:
        remaining = [item for item in relevant if item.platform == platform]
        reviews = set()
        stances = set()
        platform_stances = set()
        for _ in range(limit_per_platform):
            if not remaining:
                break
            item = max(
                remaining,
                key=lambda x: (
                    len(reviews) < minimum_reviews and x.review_id not in reviews,
                    x.stance != OpinionStance.UNCLEAR
                    and x.stance not in platform_stances,
                    x.evidence_quality,
                    x.relevance,
                    (*opinion_scope(x), x.stance) not in stances,
                    x.review_id not in reviews,
                    x.fusion_score,
                ),
            )
            result.append(item)
            remaining.remove(item)
            reviews.add(item.review_id)
            platform_stances.add(item.stance)
            stances.add((*opinion_scope(item), item.stance))
    return result


def coverage_report(
    evidence: Sequence[RankedEvidence],
    *,
    platforms: Sequence[ReviewPlatform],
    minimum: int,
) -> CoverageReport:
    """Report source counts; semantic answerability is assessed against the question."""

    relevant = [
        item for item in unique_evidence(evidence) if item.platform in platforms
    ]
    review_stances = defaultdict(set)
    for item in relevant:
        review_stances[item.platform, item.review_id].add(item.stance)
    counts = {platform: Counter() for platform in platforms}
    for (platform, _), stances in review_stances.items():
        explicit = stances - {OpinionStance.UNCLEAR}
        stance = (
            OpinionStance.MIXED
            if len(explicit) > 1
            else next(iter(explicit), OpinionStance.UNCLEAR)
        )
        counts[platform][stance.value] += 1
    by_platform = {p: sum(counts[p].values()) for p in platforms}
    missing = [p for p in platforms if by_platform[p] < minimum]
    return CoverageReport(
        minimum_reviews=minimum,
        sufficient=not missing,
        relevant_by_platform=by_platform,
        stance_counts={
            p: {s.value: counts[p][s.value] for s in OpinionStance} for p in platforms
        },
        missing_platforms=missing,
        reason="；".join(
            f"{p}：独立评论不足（{by_platform[p]}/{minimum}）" for p in missing
        )
        or None,
    )
