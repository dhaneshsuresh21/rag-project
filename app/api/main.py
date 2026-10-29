"""FastAPI backend. Run: uvicorn app.api.main:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.components import get_components
from app.ingestion.indexer import ingest
from app.observability.metrics import compute_metrics
from app.observability.retention import purge
from app.rag.service import RagService
from app.utils import configure_logging

configure_logging()
log = logging.getLogger("api")
state: dict = {}


async def _retention_loop():
    while True:
        try:
            c = get_components()
            log.info("Retention purge: %s", purge(c.settings, c.store))
        except Exception:
            log.exception("Retention purge failed")
        await asyncio.sleep(24 * 3600)


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["service"] = RagService()
    asyncio.get_running_loop().run_in_executor(None, state["service"].warmup)  # non-blocking
    task = asyncio.create_task(_retention_loop())
    yield
    task.cancel()


app = FastAPI(title="Solution Docs Assistant API", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int = Field(description="1 = thumbs up, -1 = thumbs down")
    comment: str | None = Field(default=None, max_length=2000)


class IngestRequest(BaseModel):
    force: bool = False


def _service() -> RagService:
    return state["service"]


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Admin endpoints need X-Admin-Token when ADMIN_TOKEN is set (always set it in production)."""
    expected = os.getenv("ADMIN_TOKEN")
    if expected and not (x_admin_token and secrets.compare_digest(x_admin_token, expected)):
        raise HTTPException(401, "invalid or missing X-Admin-Token")


@app.get("/health")
def health():
    c = get_components()
    ollama = {"reachable": False, "models": []}
    try:
        r = httpx.get(f"{c.settings.ollama.base_url}/api/tags", timeout=3)
        ollama = {"reachable": r.status_code == 200, "models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as e:
        ollama["error"] = str(e)
    required = [c.settings.llm.model, c.settings.embeddings.model]
    missing = [m for m in required if not any(x == m or x.startswith(m + ":") for x in ollama["models"])]
    return {
        "status": "ok" if ollama["reachable"] and not missing else "degraded",
        "ollama": ollama, "missing_models": missing,
        "documents": len(c.store.list_documents()), "chunks": c.vectors.count(),
        "llm": c.settings.llm.model, "embeddings": c.settings.embeddings.model,
        "reranker": {"model": c.settings.reranker.model, "enabled": c.settings.reranker.enabled,
                     "error": c.reranker.error},
        "show_sources": c.settings.app.show_sources,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    return _service().ask(req.message, session_id=req.session_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def events():
        for ev in _service().ask_stream(req.message, session_id=req.session_id):
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    get_components().store.add_feedback(req.trace_id, req.rating, req.comment)
    return {"ok": True}


@app.post("/admin/ingest", dependencies=[Depends(require_admin)])
def admin_ingest(req: IngestRequest):
    c = get_components()
    return ingest(c.settings, c.store, c.vectors, c.embedder, force=req.force)


@app.get("/documents")
def documents():
    return [{k: d[k] for k in ("doc_id", "title", "feature_id", "client", "doc_type", "chunk_count", "warnings")}
            for d in get_components().store.list_documents()]


@app.get("/metrics")
def metrics(days: int = 7):
    return compute_metrics(get_components().store, days=days)
