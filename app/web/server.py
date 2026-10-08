"""FastAPI host for the stable Agent UI protocol and optional built frontend."""

from contextlib import asynccontextmanager
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.web.adapter import MovieAgentDriver, report_from_answer
from app.web.contracts import CreateRun, CreateSession, PROTOCOL_VERSION, Reply
from app.web.sessions import SessionError, SessionService


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = ROOT / "reports" / "matrix_claim_audit_replay_current.json"


def create_app(*, driver_factory=MovieAgentDriver, default_model: str | None = None) -> FastAPI:
    service = SessionService(driver_factory, default_model=default_model or os.getenv("AGENT_MODEL", "gpt-5.4-mini"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await service.close()

    app = FastAPI(title="Movie Evidence Agent", version=PROTOCOL_VERSION, lifespan=lifespan)
    app.state.sessions = service
    origins = os.getenv("AGENT_WEB_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "Last-Event-ID"])

    @app.exception_handler(SessionError)
    async def session_error(request, exc):
        return JSONResponse({"error": {"message": exc.message}}, status_code=exc.status)

    @app.get("/api/v1/capabilities")
    async def capabilities():
        return {"protocol_version": PROTOCOL_VERSION, "name": "Movie Evidence Agent", "default_model": service.default_model,
                "features": ["progress", "clarification", "reports", "evidence", "cancellation", "retry"],
                "example_available": EXAMPLE_PATH.exists(), "max_query_chars": 12000,
                "storage": "process_memory"}

    @app.get("/api/v1/sessions")
    async def sessions():
        return {"items": [{"id": h.view.id, "title": h.view.title, "updated_at": h.view.updated_at,
                           "status": h.view.run.status if h.view.run else "idle"}
                          for h in sorted(service.sessions.values(), key=lambda s: s.view.updated_at, reverse=True)]}

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(body: CreateSession):
        return service.create(body.model)

    @app.get("/api/v1/sessions/{session_id}")
    async def session(session_id: str):
        return service.get(session_id).view

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        await service.delete(session_id)

    @app.post("/api/v1/sessions/{session_id}/runs", status_code=202)
    async def start_run(session_id: str, body: CreateRun):
        return service.start(session_id, body)

    @app.post("/api/v1/sessions/{session_id}/runs/{run_id}/reply", status_code=202)
    async def reply(session_id: str, run_id: str, body: Reply):
        return service.reply(session_id, run_id, body)

    @app.post("/api/v1/sessions/{session_id}/runs/{run_id}/cancel")
    async def cancel(session_id: str, run_id: str):
        return await service.cancel(session_id, run_id)

    @app.post("/api/v1/sessions/{session_id}/runs/{run_id}/retry", status_code=202)
    async def retry(session_id: str, run_id: str):
        return service.retry(session_id, run_id)

    @app.get("/api/v1/sessions/{session_id}/events")
    async def events(session_id: str, request: Request, after: int = -1, last_event_id: str | None = Header(default=None)):
        hosted = service.get(session_id)

        async def stream():
            try:
                seen = int(last_event_id) if last_event_id else after
            except ValueError:
                seen = -1
            while not hosted.closed and not await request.is_disconnected():
                if hosted.view.revision > seen:
                    seen = hosted.view.revision
                    data = {"protocol_version": PROTOCOL_VERSION, "type": "session.updated", "session": hosted.view.model_dump(mode="json")}
                    yield f"id: {seen}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                try:
                    async with hosted.condition:
                        await asyncio.wait_for(hosted.condition.wait_for(lambda: hosted.closed or hosted.view.revision > seen), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/v1/example-report")
    async def example():
        if not EXAMPLE_PATH.exists():
            raise SessionError(404, "本地还没有保存的示例报告。")
        saved = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        return {"query": saved.get("question", "《黑客帝国》的演员表演"),
                "report": report_from_answer(saved["answer"], title="The Matrix · 1999"), "kind": "saved_example"}

    # API routes precede the static mount, preserving one-origin production use.
    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
