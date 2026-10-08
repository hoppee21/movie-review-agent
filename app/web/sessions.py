"""Small in-memory session host: async jobs, idempotent replies and SSE snapshots."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from app.progress import observe_progress
from app.web.adapter import AgentDriver, stage_label
from app.web.contracts import CreateRun, Message, Outcome, Reply, Run, Session, Stage


logger = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status, self.message = status, message


@dataclass
class HostedSession:
    view: Session
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    driver: AgentDriver | None = None
    task: asyncio.Task | None = None
    requests: dict[str, tuple[str, str]] = field(default_factory=dict)
    last_reply: Reply | None = None
    closed: bool = False


class SessionService:
    def __init__(self, driver_factory: Callable[[str], AgentDriver], *, default_model: str, max_sessions: int = 50):
        self.driver_factory, self.default_model = driver_factory, default_model
        self.sessions: dict[str, HostedSession] = {}
        self.max_sessions = max_sessions
        # The existing collectors write per-movie files. Serialize their jobs.
        self.worker = asyncio.Semaphore(1)

    def create(self, model: str | None = None) -> Session:
        if len(self.sessions) >= self.max_sessions:
            raise SessionError(409, "会话数量已达上限，请先删除不用的会话。")
        stamp = now()
        view = Session(id=uuid4().hex, model=(model or self.default_model).strip() or self.default_model,
                       created_at=stamp, updated_at=stamp)
        self.sessions[view.id] = HostedSession(view)
        return view

    def get(self, session_id: str) -> HostedSession:
        if session_id not in self.sessions:
            raise SessionError(404, "会话已不存在，服务重启后请开始新的分析。")
        return self.sessions[session_id]

    def start(self, session_id: str, request: CreateRun) -> Session:
        hosted = self.get(session_id)
        if self._duplicate(hosted, request.request_id, "query:" + request.query):
            return hosted.view
        self._idle(hosted)
        if hosted.driver:
            hosted.driver.close()
            hosted.driver = None
        run = Run(id=uuid4().hex, query=request.query, status="queued", started_at=now())
        hosted.view.run = run
        hosted.last_reply = None
        if not hosted.view.messages:
            hosted.view.title = request.query[:36]
        self._message(hosted, "user", request.query)
        hosted.requests[request.request_id] = ("query:" + request.query, run.id)
        self._publish(hosted)
        hosted.task = asyncio.create_task(self._execute(hosted, run))
        return hosted.view

    def reply(self, session_id: str, run_id: str, reply: Reply) -> Session:
        hosted = self.get(session_id)
        fingerprint = "reply:" + run_id + ":" + reply.model_dump_json(exclude={"request_id"})
        if self._duplicate(hosted, reply.request_id, fingerprint):
            return hosted.view
        run = self._run(hosted, run_id)
        issue = run.clarification
        if run.status != "waiting_for_input" or not issue or issue.id != reply.issue_id:
            raise SessionError(409, "这个澄清问题已经更新，请刷新后重试。")
        choices = {option.id: option.label for option in issue.choices}
        if reply.option_id and reply.option_id not in choices:
            raise SessionError(422, "请选择当前问题中的一个选项。")
        if not reply.cancelled and not reply.option_id and not (reply.text or "").strip():
            raise SessionError(422, "请选择一项，或补充一些信息。")
        text = "取消本次分析" if reply.cancelled else "\n".join(v for v in (choices.get(reply.option_id or "", ""), (reply.text or "").strip()) if v)
        self._message(hosted, "user", text)
        run.status, run.clarification = "queued", None
        hosted.requests[reply.request_id] = (fingerprint, run.id)
        hosted.last_reply = reply
        self._publish(hosted)
        hosted.task = asyncio.create_task(self._execute(hosted, run, reply=reply))
        return hosted.view

    def retry(self, session_id: str, run_id: str) -> Session:
        hosted = self.get(session_id)
        run = self._run(hosted, run_id)
        if run.status != "failed":
            raise SessionError(409, "只有失败的分析可以重试。")
        run.status, run.error, run.finished_at = "queued", None, None
        self._publish(hosted)
        hosted.task = asyncio.create_task(self._execute(hosted, run, reply=hosted.last_reply))
        return hosted.view

    async def cancel(self, session_id: str, run_id: str) -> Session:
        hosted = self.get(session_id)
        run = self._run(hosted, run_id)
        if run.status == "waiting_for_input" and run.clarification:
            return self.reply(session_id, run_id, Reply(request_id=uuid4().hex, issue_id=run.clarification.id, cancelled=True))
        if run.status not in {"running", "queued"}:
            return hosted.view
        if hosted.task and not hosted.task.done():
            hosted.task.cancel()
            await asyncio.gather(hosted.task, return_exceptions=True)
        if run.status == "cancelled":
            return hosted.view
        if hosted.driver:
            hosted.driver.close()
            hosted.driver = None
        run.status, run.finished_at = "cancelled", now()
        self._message(hosted, "assistant", "本次分析已停止。你可以开始一个新问题。")
        self._publish(hosted)
        return hosted.view

    async def delete(self, session_id: str) -> None:
        hosted = self.get(session_id)
        hosted.closed = True
        if hosted.task and not hosted.task.done():
            hosted.task.cancel()
            await asyncio.gather(hosted.task, return_exceptions=True)
        if hosted.driver:
            hosted.driver.close()
        self.sessions.pop(session_id, None)
        async with hosted.condition:
            hosted.condition.notify_all()

    async def close(self) -> None:
        for session_id in list(self.sessions):
            await self.delete(session_id)

    @staticmethod
    def _duplicate(hosted: HostedSession, request_id: str, fingerprint: str) -> bool:
        previous = hosted.requests.get(request_id)
        if previous and previous[0] != fingerprint:
            raise SessionError(409, "这个请求编号已经用于其他内容。")
        return previous is not None

    @staticmethod
    def _idle(hosted: HostedSession) -> None:
        if hosted.view.run and hosted.view.run.status in {"running", "queued", "waiting_for_input"}:
            raise SessionError(409, "请先完成或停止当前分析。")

    @staticmethod
    def _run(hosted: HostedSession, run_id: str) -> Run:
        if not hosted.view.run or hosted.view.run.id != run_id:
            raise SessionError(409, "这次分析已经更新，请刷新后重试。")
        return hosted.view.run

    @staticmethod
    def _message(hosted: HostedSession, role: str, text: str, **kwargs) -> None:
        hosted.view.messages.append(Message(id=uuid4().hex, role=role, text=text, created_at=now(), **kwargs))

    @staticmethod
    def _publish(hosted: HostedSession) -> None:
        hosted.view.revision += 1
        hosted.view.updated_at = now()

        async def notify():
            async with hosted.condition:
                hosted.condition.notify_all()

        asyncio.create_task(notify())

    async def _execute(self, hosted: HostedSession, run: Run, *, reply: Reply | None = None) -> None:
        loop = asyncio.get_running_loop()
        driver = None
        task = asyncio.current_task()

        def update_stage(event: dict[str, Any]):
            if hosted.closed or hosted.task is not task or hosted.view.run is not run or run.status != "running":
                return
            label, phase = stage_label(event["label"]), event["phase"]
            if phase == "started":
                if len(run.stages) == 1 and run.stages[0].label == "理解问题与规划任务":
                    run.stages[0].status = "completed"
                run.stages.append(Stage(id=uuid4().hex, label=label, status="running"))
                run.stages[:] = run.stages[-80:]
            else:
                stage = next((s for s in reversed(run.stages) if s.label == label and s.status == "running"), None)
                if stage:
                    stage.status = "failed" if phase == "failed" else "completed"
                    stage.elapsed_seconds = event.get("elapsed_seconds")
            self._publish(hosted)

        try:
            async with self.worker:
                if hosted.closed or hosted.view.run is not run or run.status == "cancelled":
                    return
                run.status = "running"
                run.stages.append(Stage(id=uuid4().hex, label="理解问题与规划任务", status="running"))
                self._publish(hosted)
                if hosted.driver is None:
                    hosted.driver = self.driver_factory(hosted.view.model)
                driver = hosted.driver
                with observe_progress(lambda event: loop.call_soon_threadsafe(update_stage, event)):
                    thread_id = f"web:{hosted.view.id}:{run.id}"
                    outcome = await (driver.reply(reply.model_dump(exclude={"request_id"}), thread_id=thread_id)
                                     if reply else driver.run(run.query, thread_id=thread_id))
                if not hosted.closed and hosted.view.run is run:
                    self._finish(hosted, outcome)
        except asyncio.CancelledError:
            if driver:
                driver.close()
                if hosted.driver is driver:
                    hosted.driver = None
            raise
        except Exception as exc:
            logger.exception("Web run failed (%s)", run.id)
            run.status, run.finished_at = "failed", now()
            # Do not return provider messages or credentials to the browser.
            run.error = f"分析暂时中断（{type(exc).__name__}）。请检查服务日志，或重试本次分析。"
            for stage in run.stages:
                if stage.status == "running":
                    stage.status = "failed"
            self._publish(hosted)

    def _finish(self, hosted: HostedSession, outcome: Outcome) -> None:
        run = hosted.view.run
        for stage in run.stages:
            if stage.status == "running" or (outcome.clarification and stage.status == "failed"):
                stage.status = "completed"
        if outcome.clarification:
            run.status, run.clarification = "waiting_for_input", outcome.clarification
            self._message(hosted, "assistant", outcome.clarification.question, clarification=outcome.clarification)
        else:
            run.status, run.finished_at = "cancelled" if outcome.cancelled else "completed", now()
            self._message(hosted, "assistant", outcome.text, report=outcome.report)
            if hosted.driver:
                hosted.driver.close()
                hosted.driver = None
        self._publish(hosted)
