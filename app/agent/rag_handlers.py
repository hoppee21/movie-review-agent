"""Thin Action adapters between LangGraph state and the transient RAG runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from app.agent.schemas import (
    Action,
    ActionInvocation,
    ActionResult,
    ActionStatus,
    AgentState,
    ExecutionMode,
    ExecutionPlan,
    PatchAuthority,
    RequestAnalysis,
    StatePatch,
)
from app.tools.core.aggregate_movie_opinions import AggregateMovieOpinions
from app.tools.core.build_opinion_index import BuildOpinionIndex
from app.tools.core.query_movie_opinions import QueryMovieOpinions
from app.tools.core.tool import Tool
from app.tools.registry import ActionRegistry

if TYPE_CHECKING:
    from app.rag.runtime import RagRuntime


ActionHandler = Callable[
    [ActionInvocation, AgentState],
    Awaitable[ActionResult],
]


def register_rag_actions(registry: ActionRegistry, runtime: RagRuntime) -> None:
    """Install the three public RAG actions; retrieval internals stay private."""

    build_tool = BuildOpinionIndex(runtime.index_builder)
    query_tool = QueryMovieOpinions(runtime.retriever)
    aggregate_tool = AggregateMovieOpinions(runtime.store, runtime.aggregator)
    registry.register(
        Action.BUILD_INDEX,
        modes={ExecutionMode.EXECUTION},
        handler=_build_handler(build_tool),
    )
    registry.register(
        Action.QUERY_OPINIONS,
        modes={ExecutionMode.EXECUTION},
        handler=_query_handler(query_tool),
    )
    registry.register(
        Action.AGGREGATE_OPINIONS,
        modes={ExecutionMode.EXECUTION},
        handler=_aggregate_handler(aggregate_tool),
    )


def _build_handler(tool: Tool) -> ActionHandler:
    async def handle(
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        target_id, target_artifacts = _target_artifacts(invocation, state)
        reviews = target_artifacts.get("reviews")
        if not isinstance(reviews, dict):
            raise RuntimeError("build_opinion_index requires collected reviews")
        paths: dict[str, str] = {}
        for platform in ("imdb", "douban"):
            source = reviews.get(platform)
            path = source.get("output_path") if isinstance(source, dict) else None
            if not isinstance(path, str) or not path.strip():
                raise RuntimeError(
                    f"build_opinion_index is missing {platform} review path"
                )
            paths[f"{platform}_reviews_path"] = path
        result = await tool.execute(paths)
        target_artifacts["rag_index"] = result
        target_artifacts.pop("retrieval", None)
        target_artifacts.pop("answer", None)
        return _artifact_result(state, target_id, target_artifacts, result)

    return handle


def _query_handler(tool: Tool) -> ActionHandler:
    async def handle(
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        target_id, target_artifacts = _target_artifacts(invocation, state)
        index = target_artifacts.get("rag_index")
        index_id = index.get("index_id") if isinstance(index, dict) else None
        if not isinstance(index_id, str):
            raise RuntimeError("query_movie_opinions requires a RAG index")
        request = RequestAnalysis.model_validate(state.get("request_analysis"))
        plan = ExecutionPlan.model_validate(state.get("execution_plan"))
        result = await tool.execute(
            {
                "index_id": index_id,
                "question": request.question,
                "aspect": plan.intent.aspect,
                "platforms": plan.intent.platforms,
            }
        )
        target_artifacts["retrieval"] = result
        target_artifacts.pop("answer", None)
        return _artifact_result(state, target_id, target_artifacts, result)

    return handle


def _aggregate_handler(tool: Tool) -> ActionHandler:
    async def handle(
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        target_id, target_artifacts = _target_artifacts(invocation, state)
        retrieval = target_artifacts.get("retrieval")
        retrieval_id = (
            retrieval.get("retrieval_id")
            if isinstance(retrieval, dict)
            else None
        )
        if not isinstance(retrieval_id, str):
            raise RuntimeError(
                "aggregate_movie_opinions requires a retrieval result"
            )
        result = await tool.execute({"retrieval_id": retrieval_id})
        target_artifacts["answer"] = result
        return _artifact_result(state, target_id, target_artifacts, result)

    return handle


def _target_artifacts(
    invocation: ActionInvocation,
    state: AgentState,
) -> tuple[str, dict[str, Any]]:
    target_id = invocation.step.target_id
    if target_id is None:
        raise RuntimeError("RAG action requires a target_id")
    if not any(
        getattr(target, "target_id", None) == target_id
        or (
            isinstance(target, dict)
            and target.get("target_id") == target_id
        )
        for target in state.get("movie_targets", [])
    ):
        raise RuntimeError(f"Movie target not found: {target_id}")
    artifacts = state.get("artifacts", {})
    raw = artifacts.get(target_id, {})
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid artifact namespace: {target_id}")
    return target_id, dict(raw)


def _artifact_result(
    state: AgentState,
    target_id: str,
    target_artifacts: dict[str, Any],
    output: dict[str, Any],
) -> ActionResult:
    artifacts = dict(state.get("artifacts", {}))
    artifacts[target_id] = target_artifacts
    return ActionResult(
        status=ActionStatus.COMPLETED,
        patch=StatePatch(
            set_values={"artifacts": artifacts},
            authority=PatchAuthority.TOOL_VERIFIED,
        ),
        output=output,
    )


__all__ = ["register_rag_actions"]
