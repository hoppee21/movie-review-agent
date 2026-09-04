"""结合上游电影上下文与用户补充理解需求，不重新识别电影。"""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.agent.schemas.request import RequestAnalysis


SYSTEM_PROMPT = """
你只负责结合原始问题、澄清记录和电影上下文，理解用户想了解什么或想做什么。
- 使用电影上下文中的片名和年份，不重新猜测电影；只有 ID/URL 时保留标识符，不编造片名。
- 以用户最新明确补充为准，保留平台、评价对象、方面、比较关系和时间等限制。
- question 要明确说明用户想了解什么或想做什么，不能只重复电影名。
- 用户只提供片名、未说明需求时，如实记录这一点，不替用户编造问题。
- 不回答问题，不制定计划，不调用工具，不添加用户未指定的条件。
""".strip()


class RequestAnalyzer:
    """使用注入的聊天模型输出精简的 RequestAnalysis。"""

    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{user_query}"),
                MessagesPlaceholder("clarifications"),
                ("human", "电影上下文：\n{movie_context}"),
            ]
        )

        output = llm.with_structured_output(
            RequestAnalysis,
            method="json_schema",
            strict=True,
        )

        self._chain = prompt | output

    async def analyze(
        self,
        user_query: str,
        movie_context: str,
        clarifications: Sequence[BaseMessage] = (),
    ) -> dict[str, Any] | BaseModel:
        """保留原问题和澄清记录，用上游提供的电影名或标识符重述需求。"""

        return await self._chain.ainvoke({
            "user_query": user_query,
            "movie_context": movie_context,
            "clarifications": list(clarifications),
        })
