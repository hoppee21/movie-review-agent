"""Pure conversion between LLM movie analysis and executable movie state."""

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.schemas import (
    Action,
    AgentState,
    ClarificationIssue,
    ClarificationOption,
    ClarificationRecord,
    MovieAnalysis,
    MovieCandidate,
    MovieTarget,
    PatchAuthority,
    RunResult,
    RunStatus,
    StatePatch,
)


def movie_updates(state: AgentState, analysis: MovieAnalysis) -> AgentState:
    """Convert analysis into one target, one issue, or an unsupported result."""

    if len(analysis.mentions) > 1:
        return {
            "movie_analysis": analysis,
            "movie_targets": [],
            "pending_issues": [],
            "final_result": RunResult(
                status=RunStatus.UNSUPPORTED,
                reason="Multiple movie targets are reserved but not executed in this skeleton.",
                data={"movie_queries": [item.query for item in analysis.mentions]},
            ),
        }

    if not analysis.mentions:
        issue = ClarificationIssue(
            issue_id=_analysis_issue_id(state),
            kind="missing_movie",
            question=(analysis.clarification_question or "").strip()
            or "请补充电影名称、年份或剧情线索。",
            suggested_actions=[Action.ASK_USER],
        )
        return {
            "movie_analysis": analysis,
            "movie_targets": [],
            "pending_issues": [issue],
        }

    mention = analysis.mentions[0]
    needs_question = bool((analysis.clarification_question or "").strip())
    if len(mention.candidates) == 1 and not needs_question:
        candidate = mention.candidates[0]
        return {
            "movie_analysis": analysis,
            "movie_targets": [_candidate_target(mention.query, candidate)],
            "pending_issues": [],
        }
    if not mention.candidates and is_identifier(mention.query) and not needs_question:
        return {
            "movie_analysis": analysis,
            "movie_targets": [
                MovieTarget(target_id="movie-1", query=mention.query)
            ],
            "pending_issues": [],
        }

    options = [
        _candidate_option(mention.query, candidate, index)
        for index, candidate in enumerate(mention.candidates, start=1)
    ]
    issue = ClarificationIssue(
        issue_id=_analysis_issue_id(state),
        kind="ambiguous_movie" if options else "missing_movie",
        question=(analysis.clarification_question or "").strip()
        or "请从候选中选择电影，或补充英文片名、年份或平台 ID。",
        target_id="movie-1",
        options=options,
        suggested_actions=[Action.ASK_USER],
    )
    return {
        "movie_analysis": analysis,
        "movie_targets": [],
        "pending_issues": [issue],
    }


def movie_context(targets: list[MovieTarget]) -> str:
    if not targets:
        raise RuntimeError("Movie context requires at least one target")
    normalized = [MovieTarget.model_validate(target) for target in targets]
    return "\n".join(
        f"- {target_label(target)}"
        + (f" [{target.wikidata_id}]" if target.wikidata_id else "")
        for target in normalized
    )


def target_label(target: MovieTarget) -> str:
    title = target.english_title or target.query
    return f"{title} ({target.release_year})" if target.release_year else title


def clarification_messages(
    records: list[ClarificationRecord],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for raw_record in records:
        record = ClarificationRecord.model_validate(raw_record)
        messages.append(
            AIMessage(
                content=json.dumps(
                    {
                        "issue_id": record.issue_id,
                        "question": record.question,
                    },
                    ensure_ascii=False,
                )
            )
        )
        answer = record.selected_label or ""
        if record.text:
            answer = f"{answer}；{record.text}" if answer else record.text
        messages.append(HumanMessage(content=answer))
    return messages


def is_identifier(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:tt\d{7,10}|Q\d+|douban\s*[:#]\s*\d+"
            r"|https?://(?:www\.)?(?:imdb\.com/title/tt\d{7,10}"
            r"|movie\.douban\.com/subject/\d+|wikidata\.org/(?:wiki|entity)/Q\d+)"
            r"(?:[/?#]\S*)?)",
            value.strip(),
            re.IGNORECASE,
        )
    )


def _candidate_target(query: str, candidate: MovieCandidate) -> MovieTarget:
    return MovieTarget(
        target_id="movie-1",
        query=query,
        english_title=candidate.english_title,
        release_year=candidate.release_year,
    )


def _candidate_option(
    query: str,
    candidate: MovieCandidate,
    index: int,
) -> ClarificationOption:
    target = _candidate_target(query, candidate)
    return ClarificationOption(
        option_id=f"candidate-{index}",
        label=target_label(target),
        details={"release_year": candidate.release_year},
        internal_patch=StatePatch(
            set_values={"movie_targets": [target]},
            invalidates=["movie"],
            authority=PatchAuthority.USER_CONFIRMED,
        ),
    )


def _analysis_issue_id(state: AgentState) -> str:
    return f"movie-analysis-{len(state.get('clarification_history', [])) + 1}"
