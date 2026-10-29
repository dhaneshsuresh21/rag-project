"""RagService: runs the LangGraph pipeline with tracing + chat history. Shared by the API and the eval."""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

from app.components import Components, get_components
from app.observability.tracing import Trace, TraceWriter
from app.prompts import prompt_ref
from app.rag.graph import build_graph

log = logging.getLogger(__name__)


class RagService:
    def __init__(self, components: Components | None = None):
        self.c = components or get_components()
        self.graph = build_graph(self.c)
        self.writer = TraceWriter(self.c.store, self.c.settings.path("logs_dir"))

    def warmup(self) -> dict:
        """Load embedder, reranker and LLM into memory so the first user request isn't slow."""
        from app.components import make_chat_llm

        result = {}
        steps = {
            "embeddings": lambda: self.c.embedder.embed_query("warmup"),
            "reranker": lambda: self.c.reranker.rerank("warmup", [{"section_path": "w", "text": "warmup"}], 1),
            "llm": lambda: make_chat_llm(self.c.settings.llm.model_copy(update={"num_predict": 1}),
                                         self.c.settings.ollama.base_url).invoke("Hi"),
        }
        for name, fn in steps.items():
            try:
                fn()
                result[name] = "ok"
            except Exception as e:
                result[name] = f"failed: {e}"
        log.info("Warm-up: %s", result)
        return result

    def _start(self, question: str, session_id: str | None, source: str, use_history: bool):
        session_id = session_id or uuid.uuid4().hex
        history = self.c.store.get_history(session_id, 2 * self.c.settings.app.history_turns) if use_history else []
        trace = Trace(session_id=session_id, source=source, question=question,
                      prompt_version=prompt_ref("answer"), model=self.c.settings.llm.model)
        inputs = {"question": question, "history": history}
        return session_id, trace, inputs, {"configurable": {"trace": trace}}

    def _finish(self, trace: Trace, state: dict, session_id: str, persist: bool, include_context: bool) -> dict:
        status = state.get("status", "ERROR")
        trace.attrs.update(
            standalone_question=state.get("standalone_question"), status=status, answer=state.get("answer"),
            raw_answer=state.get("raw_answer"), sources=state.get("sources", []), usage=state.get("usage", {}),
            citation=state.get("citation", {}), top_score=state.get("top_score"), error=state.get("error"))
        self.writer.write(trace)
        if persist:
            self.c.store.add_message(session_id, "user", state.get("question", ""))
            self.c.store.add_message(session_id, "assistant", state.get("answer", ""))
        result = {
            "trace_id": trace.trace_id, "session_id": session_id, "status": status,
            "answer": state.get("answer", ""), "sources": state.get("sources", []),
            "latency_ms": trace.elapsed_ms, "standalone_question": state.get("standalone_question"),
        }
        if include_context:
            result["context"] = [{"chunk_id": c["chunk_id"], "cid": c.get("cid"), "text": c["text"],
                                  "section_path": c["section_path"]} for c in state.get("context", [])]
            result["raw_answer"] = state.get("raw_answer")
        return result

    def ask(self, question: str, session_id: str | None = None, source: str = "api", persist: bool = True,
            include_context: bool = False) -> dict:
        session_id, trace, inputs, config = self._start(question, session_id, source, use_history=persist)
        try:
            state = self.graph.invoke(inputs, config)
        except Exception as e:
            log.exception("Pipeline failed")
            state = {**inputs, "status": "ERROR", "answer": self.c.settings.app.error_message, "error": repr(e)}
        return self._finish(trace, {**inputs, **state}, session_id, persist, include_context)

    def ask_stream(self, question: str, session_id: str | None = None) -> Iterator[dict]:
        """Yields {'event': 'token'|'step'|'final', 'data': ...}. Tokens come from the generate node only."""
        session_id, trace, inputs, config = self._start(question, session_id, "api", use_history=True)
        state: dict = dict(inputs)
        yield {"event": "start", "data": {"trace_id": trace.trace_id, "session_id": session_id}}
        try:
            for mode, chunk in self.graph.stream(inputs, config, stream_mode=["messages", "updates"]):
                if mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") == "generate" and isinstance(msg.content, str) and msg.content:
                        yield {"event": "token", "data": {"text": msg.content}}
                else:
                    for node, update in chunk.items():
                        if update:
                            state.update(update)
                        yield {"event": "step", "data": {"node": node}}
        except Exception as e:
            log.exception("Pipeline failed")
            state.update(status="ERROR", answer=self.c.settings.app.error_message, error=repr(e))
        yield {"event": "final", "data": self._finish(trace, state, session_id, True, False)}
