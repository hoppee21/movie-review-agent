"""Small stage logs; library callers choose whether to display INFO messages."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import Any, Callable


logger = logging.getLogger(__name__)
_observer: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("progress_observer", default=None)


@contextmanager
def observe_progress(callback: Callable[[dict[str, Any]], None]):
    """Scope progress to one request; also propagates into asyncio.to_thread."""
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def _emit(label: str, phase: str, started: float):
    observer = _observer.get()
    if observer:
        try:
            observer({"label": label, "phase": phase, "elapsed_seconds": monotonic() - started})
        except Exception:
            logger.exception("Progress observer failed")


@contextmanager
def progress(label: str):
    started = monotonic()
    logger.info("%s：开始", label)
    _emit(label, "started", started)
    try:
        yield
    except Exception as exc:
        logger.info("%s：中断（%s），耗时 %.1f 秒", label, type(exc).__name__, monotonic() - started)
        _emit(label, "failed", started)
        raise
    else:
        logger.info("%s：完成，耗时 %.1f 秒", label, monotonic() - started)
        _emit(label, "completed", started)
