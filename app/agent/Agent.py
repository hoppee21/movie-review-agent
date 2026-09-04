"""Extensible movie Plan-and-Execute runtime with bounded clarification."""

from typing import Any, Literal, Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agent.controller import (
    apply_action_result,
    finalize_state,
    next_node,
)
from app.agent.executor import Executor, build_default_registry
from app.agent.movie_analyzer import MovieAnalyzer
from app.agent.movie_state import (
    clarification_messages,
    movie_context,
    movie_updates,
)
from app.agent.planner import PlanBuilder, compile_plan
from app.agent.request_analyzer import RequestAnalyzer
from app.agent.resolution import next_resolution_invocation
from app.agent.schemas import (
    ActionInvocation,
    AgentState,
    CHECKPOINT_MODEL_TYPES,
    ClarificationIssue,
    ClarificationReply,
    ExecutionMode,
    ExecutionPlan,
    MovieAnalysis,
    PlanIntent,
    RequestAnalysis,
    TaskType,
)
from app.tools.registry import ActionRegistry


ControllerDestination = Literal[
    "analyze_movie",
    "analyze_request",
    "resolve_next",
    "build_plan",
    "execute_action",
    "finalize",
]


class MovieAgent:
    """Own dependencies and the compiled graph; nodes remain narrowly scoped."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        action_registry: ActionRegistry | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        max_clarification_rounds: int = 3,
    ) -> None:
        self.movie_analyzer = MovieAnalyzer(llm)
        self.request_analyzer = RequestAnalyzer(llm)
        self.plan_builder = PlanBuilder(llm)
        self.executor = Executor(
            action_registry
            or build_default_registry(
                max_clarification_rounds=max_clarification_rounds
            )
        )
        self.graph = self._build_graph(checkpointer)

    def _build_graph(
        self,
        checkpointer: BaseCheckpointSaver | None,
    ):
        builder = StateGraph(AgentState)
        builder.add_node("analyze_movie", self._analyze_movie)
        builder.add_node("analyze_request", self._analyze_request)
        builder.add_node("controller", self._controller)
        builder.add_node("resolve_next", self._resolve_next)
        builder.add_node("build_plan", self._build_plan)
        builder.add_node("execute_action", self._execute_action)
        builder.add_node("apply_result", self._apply_result)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "analyze_movie")
        builder.add_edge("analyze_movie", "controller")
        builder.add_edge("analyze_request", "controller")
        builder.add_edge("resolve_next", "execute_action")
        builder.add_edge("build_plan", "controller")
        builder.add_edge("execute_action", "apply_result")
        builder.add_edge("apply_result", "controller")
        builder.add_edge("finalize", END)

        if checkpointer is None:
            checkpointer = InMemorySaver(
                serde=JsonPlusSerializer(
                    allowed_msgpack_modules=[
                        (value.__module__, value.__name__)
                        for value in CHECKPOINT_MODEL_TYPES
                    ]
                )
            )
        return builder.compile(checkpointer=checkpointer)

    async def run(self, user_query: str, *, thread_id: str) -> dict[str, Any]:
        """Run a new request or retry the failed node of the same request."""

        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        config = self._thread_config(thread_id)
        snapshot = await self.graph.aget_state(config)
        if snapshot.next:
            if (
                not snapshot.interrupts
                and snapshot.values.get("user_query") == user_query
                and any(task.error is not None for task in snapshot.tasks)
            ):
                return await self.graph.ainvoke(None, config=config)
            raise ValueError(
                "This thread is unfinished; resume it or use a new thread_id"
            )
        return await self.graph.ainvoke(
            self._initial_state(user_query),
            config=config,
        )

    async def analyze(self, user_query: str, *, thread_id: str) -> dict[str, Any]:
        """Backward-compatible alias for run()."""

        return await self.run(user_query, thread_id=thread_id)

    async def resume(
        self,
        reply: str | Mapping[str, Any] | ClarificationReply,
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        """Validate and submit one reply to the currently interrupted issue."""

        config = self._thread_config(thread_id)
        snapshot = await self.graph.aget_state(config)
        if not snapshot.interrupts:
            raise ValueError("This thread has no pending clarification")
        payload = snapshot.interrupts[0].value
        if not isinstance(payload, dict) or not isinstance(
            payload.get("issue_id"),
            str,
        ):
            raise RuntimeError("Pending clarification payload is invalid")

        parsed = self._parse_reply(reply, payload["issue_id"])
        if parsed.issue_id != payload["issue_id"]:
            raise ValueError("reply issue_id does not match pending clarification")
        option_ids = {
            option["option_id"]
            for option in payload.get("options", [])
            if isinstance(option, dict)
            and isinstance(option.get("option_id"), str)
        }
        if parsed.option_id is not None and parsed.option_id not in option_ids:
            raise ValueError("reply option_id is not in pending clarification")
        return await self.graph.ainvoke(
            Command(resume=parsed.model_dump(mode="json")),
            config=config,
        )

    async def _analyze_movie(self, state: AgentState) -> AgentState:
        result = await self.movie_analyzer.analyze(
            state["user_query"],
            clarification_messages(state.get("clarification_history", [])),
        )
        return movie_updates(state, MovieAnalysis.model_validate(result))

    async def _analyze_request(self, state: AgentState) -> AgentState:
        result = await self.request_analyzer.analyze(
            state["user_query"],
            movie_context(state.get("movie_targets", [])),
            clarification_messages(state.get("clarification_history", [])),
        )
        return {"request_analysis": RequestAnalysis.model_validate(result)}

    async def _build_plan(self, state: AgentState) -> AgentState:
        request = state.get("request_analysis")
        if request is None:
            raise RuntimeError("Cannot plan before request analysis")
        request = RequestAnalysis.model_validate(request)
        result = await self.plan_builder.analyze(
            request.question,
            movie_context(state.get("movie_targets", [])),
        )
        intent = PlanIntent.model_validate(result)
        if not intent.platforms and intent.task != TaskType.UNSUPPORTED:
            intent = intent.model_copy(update={"platforms": ["imdb", "douban"]})
        return {
            "execution_plan": compile_plan(
                intent,
                state.get("movie_targets", []),
            ),
            "plan_cursor": 0,
        }

    def _controller(
        self,
        state: AgentState,
    ) -> Command[ControllerDestination]:
        destination = next_node(state)
        update: dict[str, Any] | None = None
        if destination == "execute_action":
            plan = ExecutionPlan.model_validate(state["execution_plan"])
            step = plan.steps[state.get("plan_cursor", 0)]
            update = {
                "active_action": ActionInvocation(
                    mode=ExecutionMode.EXECUTION,
                    step=step,
                )
            }
        return Command(update=update, goto=destination)

    @staticmethod
    def _resolve_next(state: AgentState) -> AgentState:
        issues = state.get("pending_issues", [])
        if not issues:
            raise RuntimeError("resolve_next requires a pending issue")
        return {
            "active_action": next_resolution_invocation(
                ClarificationIssue.model_validate(issues[0])
            )
        }

    async def _execute_action(self, state: AgentState) -> AgentState:
        invocation = state.get("active_action")
        if invocation is None:
            raise RuntimeError("execute_action requires an active action")
        return {
            "action_result": await self.executor.execute(invocation, state)
        }

    @staticmethod
    def _apply_result(state: AgentState) -> AgentState:
        return apply_action_result(state)

    @staticmethod
    def _finalize(state: AgentState) -> AgentState:
        return finalize_state(state)

    @staticmethod
    def _parse_reply(
        reply: str | Mapping[str, Any] | ClarificationReply,
        issue_id: str,
    ) -> ClarificationReply:
        if isinstance(reply, str):
            if not reply.strip():
                raise ValueError("reply must be a non-empty string")
            return ClarificationReply(issue_id=issue_id, text=reply)
        if isinstance(reply, ClarificationReply):
            return reply
        return ClarificationReply.model_validate(dict(reply))

    @staticmethod
    def _thread_config(thread_id: str) -> RunnableConfig:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _initial_state(user_query: str) -> AgentState:
        return {
            "user_query": user_query,
            "movie_analysis": None,
            "movie_targets": [],
            "request_analysis": None,
            "clarification_history": [],
            "clarification_rounds": 0,
            "pending_issues": [],
            "execution_plan": None,
            "plan_cursor": 0,
            "active_action": None,
            "action_result": None,
            "artifacts": {},
            "final_result": None,
        }
