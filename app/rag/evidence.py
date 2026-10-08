"""Shared opinion selection and coverage rules, independent of retrieval algorithms."""

from collections import Counter, defaultdict
from collections.abc import Sequence
import re

from app.rag.models import (
    CoverageGap,
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
    for item in sorted(evidence, key=lambda x: (
        -x.relevance, -x.evidence_quality, -x.fusion_score, x.evidence_id,
    )):
        span = (item.evidence_span or "").strip()
        if item.relevance < 2 or not span or span not in item.chunk_text or span not in item.full_text:
            continue
        quote = " ".join(span.casefold().split())
        group = (item.platform, *opinion_scope(item), item.stance)
        key = (*group, item.review_id, " ".join(item.opinion.casefold().split()))
        if any(quote in previous or previous in quote for previous in seen_spans[key]):
            continue
        quote_key = (*group, quote)
        if len(quote) >= 64 and quote_sources.get(quote_key, item.review_id) != item.review_id:
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
    """Spend the final budget on dimensions, comparable targets and distinct opinions."""

    relevant = unique_evidence(evidence)
    groups = [{opinion_scope(item) for item in relevant if item.platform == p} for p in platforms]
    common = set.intersection(*groups) if groups else set()
    result = []
    for platform in platforms:
        remaining = [item for item in relevant if item.platform == platform]
        dimensions, targets, explained, stances, reviews = set(), set(), set(), set(), set()
        platform_stances = set()
        for _ in range(limit_per_platform):
            if not remaining:
                break
            item = max(remaining, key=lambda x: (
                x.subaspect not in dimensions,
                len(reviews) < minimum_reviews and x.review_id not in reviews,
                x.stance != OpinionStance.UNCLEAR and x.stance not in platform_stances,
                opinion_scope(x) in common and opinion_scope(x) not in targets,
                opinion_scope(x) not in targets,
                x.evidence_quality == 2 and opinion_scope(x) not in explained,
                (*opinion_scope(x), x.stance) not in stances,
                x.review_id not in reviews,
                x.evidence_quality, x.relevance, x.fusion_score,
            ))
            result.append(item)
            remaining.remove(item)
            dimensions.add(item.subaspect)
            targets.add(opinion_scope(item))
            if item.evidence_quality == 2:
                explained.add(opinion_scope(item))
            stances.add((*opinion_scope(item), item.stance))
            platform_stances.add(item.stance)
            reviews.add(item.review_id)
    return result


def coverage_report(
    evidence: Sequence[RankedEvidence],
    *,
    platforms: Sequence[ReviewPlatform],
    subaspects: Sequence[str],
    minimum: int,
) -> CoverageReport:
    """Check the actual writing evidence; emit the same gaps used by follow-up retrieval."""

    relevant = [item for item in unique_evidence(evidence) if item.platform in platforms]
    review_stances = defaultdict(set)
    for item in relevant:
        review_stances[item.platform, item.review_id].add(item.stance)
    counts = {platform: Counter() for platform in platforms}
    for (platform, _), stances in review_stances.items():
        explicit = stances - {OpinionStance.UNCLEAR}
        stance = (
            OpinionStance.MIXED if len(explicit) > 1
            else next(iter(explicit), OpinionStance.UNCLEAR)
        )
        counts[platform][stance.value] += 1
    by_platform = {p: sum(counts[p].values()) for p in platforms}
    groups = defaultdict(list)
    for item in relevant:
        groups[item.platform, *opinion_scope(item)].append(item)

    gaps = []
    for platform in platforms:
        if by_platform[platform] < minimum:
            gaps.append(CoverageGap(platform=platform, kind="reviews"))
        observed = {item.subaspect for item in relevant if item.platform == platform}
        for subaspect in subaspects:
            if subaspect not in observed:
                gaps.append(CoverageGap(platform=platform, kind="subaspect", subaspect=subaspect))

    for (platform, target, subaspect), items in groups.items():
        if not any(item.evidence_quality == 2 for item in items):
            gaps.append(CoverageGap(
                platform=platform, kind="reason", target=items[0].target, subaspect=subaspect,
            ))
        for other in platforms:
            if other != platform and (other, target, subaspect) not in groups:
                gaps.append(CoverageGap(
                    platform=other, kind="comparison", target=items[0].target, subaspect=subaspect,
                ))

    missing = {p: [g.subaspect for g in gaps if g.platform == p and g.kind == "subaspect"]
               for p in platforms}
    missing_subaspects = [label for label in subaspects if any(label in v for v in missing.values())]
    return CoverageReport(
        minimum_reviews=minimum,
        sufficient=not gaps,
        relevant_by_platform=by_platform,
        stance_counts={p: {s.value: counts[p][s.value] for s in OpinionStance} for p in platforms},
        covered_subaspects=[s for s in subaspects if s not in missing_subaspects],
        missing_subaspects=missing_subaspects,
        missing_platforms=[g.platform for g in gaps if g.kind == "reviews"],
        missing_subaspects_by_platform=missing,
        reason="；".join(describe_gap(gap) for gap in gaps) or None,
        gaps=gaps,
    )


def describe_gap(gap: CoverageGap) -> str:
    need = {
        "reviews": "独立评论不足",
        "subaspect": "缺少直接观点",
        "reason": "仅有判断，缺少原文明示的理由或条件",
        "comparison": "缺少同一对象和方面的可比观点",
    }[gap.kind]
    scope = " / ".join(value for value in (gap.platform, gap.target, gap.subaspect) if value)
    return f"{scope}：{need}"
