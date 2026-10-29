"""LangGraph RAG pipeline: rewrite -> retrieve -> rerank -> gate -> generate -> validate (or refuse)."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.components import Components
from app.prompts import load_prompt
from app.rag.citations import strip_citations, validate_citations

log = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]
    standalone_question: str
    candidates: list[dict]
    context: list[dict]
    top_score: float | None
    rerank_used: bool
    raw_answer: str
    attempt: int
    repair_reason: str
    status: str  # ANSWERED | NOT_IN_DOCS | ERROR (RETRY is internal only)
    answer: str
    sources: list[dict]
    citation: dict
    usage: dict
    error: str


class _NullTrace:
    def span(self, name, **attrs):
        return nullcontext({})


def _trace(config: RunnableConfig):
    return (config or {}).get("configurable", {}).get("trace") or _NullTrace()


def format_context(context: list[dict]) -> str:
    return "\n\n".join(f"[{c['cid']}] {c['metadata'].get('doc_title', '')} > {c['section_path']}\n{c['text']}"
                       for c in context)


def build_graph(comp: Components):
    s = comp.settings
    answer_prompt, rewrite_prompt = load_prompt("answer"), load_prompt("rewrite")

    def rewrite(state: RAGState, config: RunnableConfig) -> RAGState:
        q, history = state["question"].strip(), state.get("history") or []
        with _trace(config).span("rewrite") as sp:
            if not history:
                sp["skipped"] = True
                return {"standalone_question": q}
            convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-2 * s.app.history_turns:])
            try:
                out = comp.rewrite_llm.invoke([
                    SystemMessage(rewrite_prompt["system"]),
                    HumanMessage(rewrite_prompt["user"].format(history=convo, question=q)),
                ], config=config).content.strip().strip('"').splitlines()
                rewritten = out[0].strip() if out else ""
            except Exception as e:  # degrade: use the original question
                sp["error"], rewritten = repr(e), ""
            standalone = rewritten if 3 < len(rewritten) < 500 else q
            sp["output"] = standalone
            return {"standalone_question": standalone}

    def retrieve(state: RAGState, config: RunnableConfig) -> RAGState:
        with _trace(config).span("retrieve") as sp:
            cands = comp.retriever.retrieve(state["standalone_question"])
            sp["candidates"] = [{k: c.get(k) for k in ("chunk_id", "bm25_rank", "vector_rank", "rrf_score")}
                                for c in cands]
            return {"candidates": cands}

    def rerank(state: RAGState, config: RunnableConfig) -> RAGState:
        with _trace(config).span("rerank") as sp:
            ranked = comp.reranker.rerank(state["standalone_question"], state.get("candidates", []),
                                          s.retrieval.rerank_top_n)
            used = comp.reranker.available
            context = comp.retriever.expand_and_budget(ranked)
            for i, c in enumerate(context, start=1):
                c["cid"] = f"C{i}"
            top = ranked[0].get("rerank_score") if ranked and used else None
            sp.update(rerank_used=used, top_score=top,
                      ranked=[{"chunk_id": c["chunk_id"], "score": c.get("rerank_score")} for c in ranked],
                      context=[c["chunk_id"] for c in context])
            if comp.reranker.error:
                sp["degraded"] = comp.reranker.error
            return {"context": context, "top_score": top, "rerank_used": used}

    def gate(state: RAGState) -> str:
        if not state.get("context"):
            return "refuse"
        if state.get("rerank_used") and (state.get("top_score") or 0) < s.retrieval.min_rerank_score:
            return "refuse"
        return "generate"

    def refuse(state: RAGState, config: RunnableConfig) -> RAGState:
        with _trace(config).span("refuse", reason="low_relevance_or_no_context"):
            return {"status": "NOT_IN_DOCS", "answer": s.app.not_in_docs_message, "sources": [],
                    "citation": {"reason": "low_relevance_or_no_context"}}

    def generate(state: RAGState, config: RunnableConfig) -> RAGState:
        attempt = state.get("attempt", 0)
        with _trace(config).span("generate", model=s.llm.model, attempt=attempt) as sp:
            messages = [
                SystemMessage(answer_prompt["system"]),
                HumanMessage(answer_prompt["user"].format(context=format_context(state["context"]),
                                                          question=state["standalone_question"])),
            ]
            if attempt:  # repair turn: show the model its rejected answer and why
                messages += [AIMessage(state.get("raw_answer", "")),
                             HumanMessage(answer_prompt["repair"].format(reason=state.get("repair_reason", "")))]
            try:
                msg = comp.llm.invoke(messages, config=config)
            except Exception as e:  # graceful degradation: point to the most relevant sections
                sp["error"] = repr(e)
                sections = list(dict.fromkeys(
                    f"{c['metadata'].get('doc_title', '')} > {c['section_path']}" for c in state["context"][:3]))
                return {"status": "ERROR", "error": repr(e), "sources": [],
                        "answer": s.app.error_message + "\nRelevant sections: " + "; ".join(sections)}
            usage = dict(getattr(msg, "usage_metadata", None) or {})
            sp["usage"] = usage
            return {"raw_answer": msg.content, "usage": usage}

    def validate(state: RAGState, config: RunnableConfig) -> RAGState:
        if state.get("status") == "ERROR":
            return {}
        with _trace(config).span("validate", attempt=state.get("attempt", 0)) as sp:
            ctx = {c["cid"]: c for c in state["context"]}
            res = validate_citations(state.get("raw_answer", ""), set(ctx), s.app.require_citations,
                                     context_text=format_context(state["context"]))
            citation = {"used": res.used_ids, "invalid": res.invalid_ids, "unknown_identifiers":
                        res.unknown_identifiers, "removed": res.removed_sentences, "uncited": res.uncited_sentences,
                        "reason": res.reason, "attempt": state.get("attempt", 0)}
            sp.update(status=res.status, **citation)
            if res.status != "ANSWERED":
                if res.reason != "model_refused" and state.get("attempt", 0) < s.app.max_generation_retries:
                    sp["action"] = "retry"
                    return {"status": "RETRY", "attempt": state.get("attempt", 0) + 1, "repair_reason": res.reason,
                            "citation": citation}
                return {"status": "NOT_IN_DOCS", "answer": s.app.not_in_docs_message, "sources": [],
                        "citation": citation}
            sources = [{"cid": cid, "chunk_id": ctx[cid]["chunk_id"], "doc_title": ctx[cid]["metadata"]["doc_title"],
                        "feature_id": ctx[cid]["metadata"].get("feature_id", ""),
                        "section_path": ctx[cid]["section_path"]} for cid in res.used_ids]
            answer = res.answer if s.app.show_sources else strip_citations(res.answer)
            return {"status": "ANSWERED", "answer": answer, "sources": sources, "citation": citation}

    g = StateGraph(RAGState)
    for name, fn in [("rewrite", rewrite), ("retrieve", retrieve), ("rerank", rerank), ("refuse", refuse),
                     ("generate", generate), ("validate", validate)]:
        g.add_node(name, fn)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", gate, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", lambda st: "generate" if st.get("status") == "RETRY" else END,
                            {"generate": "generate", END: END})
    g.add_edge("refuse", END)
    return g.compile()
