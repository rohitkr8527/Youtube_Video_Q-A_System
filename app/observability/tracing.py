from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterator

from app.config import get_settings


_lock = Lock()


class Trace:
    def __init__(self, operation: str, video_id: str | None = None) -> None:
        self.trace_id = uuid.uuid4().hex
        self.operation = operation
        self.video_id = video_id
        self.events: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"name": name, "at_ms": round((time.perf_counter() - self.started) * 1000, 2), **data})

    @contextmanager
    def span(self, name: str, **data: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.event(name, duration_ms=round((time.perf_counter() - start) * 1000, 2), **data)

    def finish(self, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": self.trace_id,
            "operation": self.operation,
            "video_id": self.video_id,
            "total_ms": round((time.perf_counter() - self.started) * 1000, 2),
            "events": self.events,
            **data,
        }
        path = get_settings().trace_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
