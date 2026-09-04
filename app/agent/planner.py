"""LLM intent analysis followed by deterministic whitelist compilation."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.agent.schemas import (
    Action,
    ExecutionPlan,
    MovieTarget,
    PlanIntent,
    PlanStep,
    TaskType,
)


SYSTEM_PROMPT = """
你只负责判断电影 Agent 应执行哪类任务，不直接选择或调用工具。
- resolve_urls：查找电影对应的 IMDb/豆瓣链接。
- collect_reviews：采集电影评论，但不分析观点。
- opinion_qa：询问某电影、某平台用户在某方面的看法。
- platform_comparison：比较同一电影在 IMDb 与豆瓣上的观点。
- unsupported：与电影链接、评论采集或评论观点无关，或需要当前 Agent 不支持的多电影执行。
- platforms 只包含用户要求的平台；未指定平台时使用 imdb 和 douban。
- aspect 只保留用户明确询问的评价方面；没有则为 null。
- unsupported_reason 只在 task=unsupported 时填写，否则为 null。
""".strip()


ACTION_PLANS: dict[TaskType, tuple[Action, ...]] = {
    TaskType.RESOLVE_URLS: (Action.RESOLVE_MOVIE,),
    TaskType.COLLECT_REVIEWS: (
        Action.RESOLVE_MOVIE,
        Action.COLLECT_REVIEWS,
    ),
    TaskType.OPINION_QA: (
        Action.RESOLVE_MOVIE,
        Action.COLLECT_REVIEWS,
        Action.BUILD_INDEX,
        Action.QUERY_OPINIONS,
        Action.AGGREGATE_OPINIONS,
    ),
    TaskType.PLATFORM_COMPARISON: (
        Action.RESOLVE_MOVIE,
        Action.COLLECT_REVIEWS,
        Action.BUILD_INDEX,
        Action.QUERY_OPINIONS,
        Action.AGGREGATE_OPINIONS,
    ),
    TaskType.UNSUPPORTED: (),
}


class PlanBuilder:
    """Own the LLM chain; action selection remains deterministic."""

    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "电影上下文：\n{movie_context}\n\n用户需求：\n{question}",
                ),
            ]
        )
        output = llm.with_structured_output(
            PlanIntent,
            method="json_schema",
            strict=True,
        )
        self._chain = prompt | output

    async def analyze(
        self,
        question: str,
        movie_context: str,
    ) -> dict[str, Any] | BaseModel:
        return await self._chain.ainvoke(
            {"question": question, "movie_context": movie_context}
        )


def compile_plan(intent: PlanIntent, targets: list[MovieTarget]) -> ExecutionPlan:
    """Compile intent into stable target-bound steps without LLM tool choice."""

    if len(targets) != 1:
        intent = PlanIntent(
            task=TaskType.UNSUPPORTED,
            platforms=intent.platforms,
            aspect=intent.aspect,
            unsupported_reason="The first skeleton executes exactly one movie target.",
        )
        return ExecutionPlan(intent=intent, steps=[])

    target_id = MovieTarget.model_validate(targets[0]).target_id
    actions = ACTION_PLANS[intent.task]
    return ExecutionPlan(
        intent=intent,
        steps=[
            PlanStep(
                step_id=f"step-{index}",
                action=action,
                target_id=target_id,
            )
            for index, action in enumerate(actions, start=1)
        ],
    )
