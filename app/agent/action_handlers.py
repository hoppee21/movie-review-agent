"""Adapters from registered Actions to existing Tool contracts and interrupts."""

import re
from collections.abc import Callable, Iterable
from typing import Any

from langgraph.types import interrupt

from app.agent.movie_state import get_movie_target, numbers_in_text, target_label
from app.agent.schemas import (
    Action,
    ActionInvocation,
    ActionResult,
    ActionStatus,
    AgentState,
    ClarificationIssue,
    ClarificationOption,
    ClarificationRecord,
    ClarificationReply,
    ExecutionMode,
    MovieTarget,
    MovieTargetStatus,
    PatchAuthority,
    RunResult,
    RunStatus,
    StatePatch,
)
from app.tools.core.collect_movie_reviews import CollectMovieReviews
from app.tools.core.tool import Tool
from app.tools.core.wikidata_query_service import WikidataQueryService
from app.tools.registry import ActionHandler, ActionRegistry

RegistryExtension = Callable[[ActionRegistry], None]


def build_default_registry(
    *,
    resolver: Tool | None = None,
    collector: Tool | None = None,
    extensions: Iterable[RegistryExtension] = (),
    max_clarification_rounds: int = 3,
) -> ActionRegistry:
    """Register core actions, then install optional capability groups."""

    if max_clarification_rounds < 1:
        raise ValueError("max_clarification_rounds must be positive")
    resolver = resolver or WikidataQueryService()
    collector = collector or CollectMovieReviews()
    registry = ActionRegistry()
    registry.register(
        Action.ASK_USER,
        modes={ExecutionMode.RESOLUTION},
        handler=_ask_user_handler(max_clarification_rounds),
    )
    registry.register(
        Action.RESOLVE_MOVIE,
        modes={ExecutionMode.RESOLUTION, ExecutionMode.EXECUTION},
        handler=_tool_resolver_handler(resolver),
    )
    registry.register(
        Action.COLLECT_REVIEWS,
        modes={ExecutionMode.EXECUTION},
        handler=_tool_collector_handler(collector),
    )
    for extension in extensions:
        extension(registry)
    return registry


def _replace_target(
    state: AgentState,
    replacement: MovieTarget,
) -> list[MovieTarget]:
    targets = [
        MovieTarget.model_validate(target) for target in state.get("movie_targets", [])
    ]
    for index, target in enumerate(targets):
        if target.target_id == replacement.target_id:
            targets[index] = replacement
            return targets
    raise RuntimeError(f"Movie target not found: {replacement.target_id}")


def _resolved_target(target: MovieTarget, value: dict[str, Any]) -> MovieTarget:
    imdb_url = value.get("imdb_url")
    douban_url = value.get("douban_url")
    if not isinstance(value.get("wikidata_id"), str):
        raise RuntimeError("Resolved movie is missing its verified Wikidata identity")
    return target.model_copy(
        update={
            "english_title": str(
                value.get("title") or target.english_title or target.query
            ),
            "release_year": value.get("release_year"),
            "status": MovieTargetStatus.RESOLVED,
            "wikidata_id": value.get("wikidata_id"),
            "imdb_url": imdb_url,
            "douban_url": douban_url,
        }
    )


def _tool_resolver_handler(tool: Tool) -> ActionHandler:
    async def handle(
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        target = get_movie_target(state, invocation.step.target_id)
        if target.status == MovieTargetStatus.RESOLVED:
            return ActionResult(
                status=ActionStatus.COMPLETED,
                output={"target": target.model_dump()},
            )

        params: dict[str, Any] = {
            "movie_query": target.wikidata_id or target.query,
        }
        if target.release_year is not None:
            params["release_year"] = target.release_year
        if target.installment is not None:
            params["installment"] = target.installment
        if target.search_hints:
            params["search_hints"] = target.search_hints
        result = await tool.execute(params)
        status = result.get("status")

        if status == "resolved":
            resolved = _resolved_target(target, result)
            return ActionResult(
                status=ActionStatus.COMPLETED,
                patch=StatePatch(
                    set_values={"movie_targets": _replace_target(state, resolved)},
                    invalidates=["request", "plan"],
                    authority=PatchAuthority.TOOL_VERIFIED,
                ),
                output=result,
            )

        if status == "needs_clarification":
            options = [
                _resolver_option(state, target, candidate, index)
                for index, candidate in enumerate(
                    result.get("candidates", []),
                    start=1,
                )
            ]
            return ActionResult(
                status=ActionStatus.NEEDS_CLARIFICATION,
                issue=ClarificationIssue(
                    issue_id=f"{invocation.step.step_id}:movie",
                    kind="movie_identity",
                    question=result.get("question") or "请确认你指的具体电影及版本。",
                    target_id=target.target_id,
                    options=options,
                    suggested_actions=[Action.ASK_USER],
                ),
                output=result,
            )

        if status == "not_found":
            return ActionResult(
                status=ActionStatus.NEEDS_CLARIFICATION,
                issue=ClarificationIssue(
                    issue_id=f"{invocation.step.step_id}:not-found",
                    kind="movie_not_found",
                    question=(
                        "没有找到匹配作品，请检查片名，或补充年份、演员等线索或平台 ID。"
                    ),
                    target_id=target.target_id,
                    suggested_actions=[Action.ASK_USER],
                ),
                output=result,
            )
        if status == "unsupported":
            return ActionResult(status=ActionStatus.UNSUPPORTED, output=result)
        raise RuntimeError(f"Unexpected resolve_movie status: {status!r}")

    return handle


def _resolver_option(
    state: AgentState,
    target: MovieTarget,
    candidate: dict[str, Any],
    index: int,
) -> ClarificationOption:
    if candidate.get("entity_type") == "series":
        return ClarificationOption(
            option_id=f"candidate-{index}",
            label=f"{candidate['title']}（整个系列）",
            details={"wikidata_id": candidate["wikidata_id"], "entity_type": "series"},
            internal_patch=StatePatch(
                set_values={
                    "final_result": RunResult(
                        status=RunStatus.UNSUPPORTED,
                        reason="当前仅支持单部电影，暂不支持整个系列的联合处理。",
                        data={"requested_series": candidate},
                    )
                },
                authority=PatchAuthority.USER_CONFIRMED,
            ),
        )
    resolved = _resolved_target(target, candidate)
    return ClarificationOption(
        option_id=f"candidate-{index}",
        label=target_label(resolved),
        details={
            "release_year": resolved.release_year,
            "wikidata_id": resolved.wikidata_id,
            "entity_type": "movie",
            "installments": candidate.get("installments", []),
            "imdb_url": resolved.imdb_url,
            "douban_url": resolved.douban_url,
        },
        internal_patch=StatePatch(
            set_values={"movie_targets": _replace_target(state, resolved)},
            invalidates=["movie"],
            authority=PatchAuthority.USER_CONFIRMED,
        ),
    )


def _tool_collector_handler(tool: Tool) -> ActionHandler:
    async def handle(
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        target = get_movie_target(state, invocation.step.target_id)
        if not target.imdb_url or not target.douban_url:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                output={
                    "reason": "已核实电影身份，但缺少 IMDb 或豆瓣链接，当前无法采集双平台评论。",
                    "movie": target.model_dump(mode="json"),
                },
            )
        result = await tool.execute(
            {"imdb_url": target.imdb_url, "douban_url": target.douban_url}
        )
        artifacts = dict(state.get("artifacts", {}))
        target_artifacts = dict(artifacts.get(target.target_id, {}))
        for stale_name in ("rag_index", "retrieval", "answer"):
            target_artifacts.pop(stale_name, None)
        target_artifacts["reviews"] = result
        artifacts[target.target_id] = target_artifacts
        return ActionResult(
            status=ActionStatus.COMPLETED,
            patch=StatePatch(
                set_values={"artifacts": artifacts},
                authority=PatchAuthority.TOOL_VERIFIED,
            ),
            output=result,
        )

    return handle


def _ask_user_handler(max_rounds: int) -> ActionHandler:
    async def handle(
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        issues = [
            ClarificationIssue.model_validate(issue)
            for issue in state.get("pending_issues", [])
        ]
        if not issues or issues[0].issue_id != invocation.issue_id:
            raise RuntimeError("Active clarification issue does not match invocation")
        if state.get("clarification_rounds", 0) >= max_rounds:
            return ActionResult(
                status=ActionStatus.CANCELLED,
                output={"reason": "clarification_budget_exhausted"},
            )

        issue = issues[0]
        raw_reply = interrupt(issue.public_payload())
        reply = ClarificationReply.model_validate(raw_reply)
        if reply.issue_id != issue.issue_id:
            raise ValueError("reply issue_id does not match pending clarification")
        option = _selected_option(issue, reply)

        if reply.cancelled:
            return ActionResult(
                status=ActionStatus.CANCELLED,
                output={"reason": "user_cancelled"},
            )

        record = ClarificationRecord(
            issue_id=issue.issue_id,
            question=issue.question,
            options=[item.public_payload() for item in issue.options],
            option_id=option.option_id if option else None,
            selected_label=option.label if option else None,
            text=reply.text,
        )
        selected_patch = option.internal_patch if option else None
        set_values = dict(selected_patch.set_values) if selected_patch else {}
        set_values.update(
            pending_issues=issues[1:],
            clarification_rounds=state.get("clarification_rounds", 0) + 1,
        )
        append_values = dict(selected_patch.append_values) if selected_patch else {}
        append_values.setdefault("clarification_history", []).append(record)
        invalidates = list(selected_patch.invalidates) if selected_patch else []
        if reply.text:
            invalidates.extend(["request", "plan"] if option else ["movie"])

        return ActionResult(
            status=ActionStatus.COMPLETED,
            patch=StatePatch(
                set_values=set_values,
                append_values=append_values,
                invalidates=list(dict.fromkeys(invalidates)),
                authority=PatchAuthority.USER_CONFIRMED,
            ),
        )

    return handle


def _selected_option(
    issue: ClarificationIssue,
    reply: ClarificationReply,
) -> ClarificationOption | None:
    if reply.option_id is not None:
        option = next(
            (item for item in issue.options if item.option_id == reply.option_id),
            None,
        )
        if option is None:
            raise ValueError("reply option_id is not in pending clarification")
        return option

    # Interpret only a complete, explicit choice against the shown list.
    text = (reply.text or "").strip()
    match = re.fullmatch(
        r"(?:我选|选择|选)?第?([一二两三四五六七八九十\d]+)(个|项|部|号|版)?[。.!！]?",
        text,
    )
    if not issue.options or not match:
        return None
    numbers = numbers_in_text(text)
    if len(numbers) != 1:
        return None
    number = numbers.pop()
    if match.group(2) == "部":
        matches = [
            item
            for item in issue.options
            if number in item.details.get("installments", [])
        ]
    elif 1 <= number <= len(issue.options):
        return issue.options[number - 1]
    else:
        matches = [
            item for item in issue.options if item.details.get("release_year") == number
        ]
    return matches[0] if len(matches) == 1 else None
