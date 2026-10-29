"""Per-request tracing: spans with timings + attributes, written to logs/traces-YYYY-MM-DD.jsonl and SQLite."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.store import Store


class Trace:
    def __init__(self, session_id: str | None = None, source: str = "api", **attrs):
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.source = source
        self.started = time.time()
        self._t0 = time.perf_counter()
        self.spans: list[dict] = []
        self.attrs: dict = dict(attrs)

    @contextmanager
    def span(self, name: str, **attrs):
        start = time.perf_counter()
        data = dict(attrs)
        try:
            yield data
        except Exception as e:
            data["error"] = repr(e)
            raise
        finally:
            self.spans.append({
                "name": name,
                "start_ms": round((start - self._t0) * 1000, 1),
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                **data,
            })

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def to_record(self) -> dict:
        return {"trace_id": self.trace_id, "session_id": self.session_id, "source": self.source,
                "created_at": self.started, "latency_ms": self.elapsed_ms, **self.attrs, "spans": self.spans}


class TraceWriter:
    def __init__(self, store: Store, logs_dir: Path):
        self.store = store
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def write(self, trace: Trace) -> dict:
        rec = trace.to_record()
        day = datetime.fromtimestamp(trace.started).strftime("%Y-%m-%d")
        with open(self.logs_dir / f"traces-{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.store.insert_trace({
            "trace_id": trace.trace_id, "session_id": trace.session_id, "source": trace.source,
            "created_at": trace.started, "question": rec.get("question"),
            "standalone_question": rec.get("standalone_question"), "status": rec.get("status"),
            "answer": rec.get("answer"), "latency_ms": rec["latency_ms"],
            "prompt_version": rec.get("prompt_version"), "model": rec.get("model"),
            "has_citation": int(bool(rec.get("sources"))), "data": rec,
        })
        return rec
