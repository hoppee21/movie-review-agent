"""Extract movie mentions without interpreting or answering the request."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.agent.schemas.movie import MovieAnalysis


SYSTEM_PROMPT = """
你只负责识别用户提到了哪些电影，不理解评价需求、不回答问题、不制定计划。
- 每一部不同电影必须成为一个独立 mention；不要把两个明确电影误当成同一电影的候选。
- mention.query 保留用户用于指代该电影的原始名称、ID、URL或描述。
- 对单个中文名、英文名、别名或剧情描述，提供最多三个可能的标准英文候选，不虚构。
- 用户提供 IMDb ID/URL、豆瓣 URL 或 Wikidata ID 时，保留原值且 candidates 为空。
- candidates 和年份都未经外部核实；不确定年份时使用 null。
- 完全没有电影线索时 mentions 为空，并提出一个简短 clarification_question。
- 只有一个目标但无法区分候选时提出 clarification_question。
- 用户明确提出多个不同电影时分别输出 mentions，clarification_question 为 null。
- 单个目标的线索足够用于外部解析时 clarification_question 为 null。
- 澄清记录包含此前展示的候选；用户说“第二部”等时结合候选顺序理解。
""".strip()


class MovieAnalyzer:
    """Return strict MovieAnalysis using the injected chat model."""

    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{user_query}"),
                MessagesPlaceholder("clarifications"),
            ]
        )
        output = llm.with_structured_output(
            MovieAnalysis,
            method="json_schema",
            strict=True,
        )
        self._chain = prompt | output

    async def analyze(
        self,
        user_query: str,
        clarifications: Sequence[BaseMessage] = (),
    ) -> dict[str, Any] | BaseModel:
        return await self._chain.ainvoke(
            {"user_query": user_query, "clarifications": list(clarifications)}
        )
