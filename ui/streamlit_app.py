"""Standalone web chat (MVP). Run: streamlit run ui/streamlit_app.py  (needs the API running)."""
from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st

API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Solution Docs Assistant", page_icon=":books:", layout="wide")


def api_get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, timeout: float = 600):
    headers = {"X-Admin-Token": os.environ["ADMIN_TOKEN"]} if os.getenv("ADMIN_TOKEN") else {}
    r = httpx.post(f"{API}{path}", json=payload, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()


def stream_chat(message: str, session_id: str):
    """Yields (event, data) tuples from the SSE endpoint."""
    with httpx.stream("POST", f"{API}/chat/stream", json={"message": message, "session_id": session_id},
                      timeout=httpx.Timeout(600, connect=10)) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                yield event, json.loads(line[5:].strip())
                event = None


def render_sources(sources: list[dict]):
    if sources:
        lines = [f"- **[{s['cid']}]** {s['doc_title']} › {s['section_path']}" for s in sources]
        with st.expander("Based on", expanded=False):
            st.markdown("\n".join(lines))


def send_feedback(trace_id: str, key: str):
    val = st.session_state.get(key)
    if val is not None:
        try:
            api_post("/feedback", {"trace_id": trace_id, "rating": 1 if val == 1 else -1}, timeout=10)
        except Exception as e:
            st.toast(f"Feedback not saved: {e}")


# ---------------- sidebar ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []

with st.sidebar:
    st.title("Solution Docs Assistant")
    page = st.radio("Page", ["Chat", "Documents", "Metrics"], label_visibility="collapsed")
    if st.button("New chat", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()
    try:
        h = api_get("/health")
        st.caption(f"Status: **{h['status']}** · LLM `{h['llm']}` · {h['documents']} docs / {h['chunks']} chunks")
        if h.get("missing_models"):
            st.warning("Missing Ollama models: " + ", ".join(h["missing_models"]) +
                       "\n\nRun `ollama pull <model>`.")
        show_sources = h.get("show_sources", True)
    except Exception as e:
        st.error(f"API not reachable at {API}: {e}")
        show_sources = True

# ---------------- pages ----------------
if page == "Chat":
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if show_sources:
                    render_sources(m.get("sources", []))
                if m.get("trace_id"):
                    key = f"fb_{m['trace_id']}"
                    st.feedback("thumbs", key=key, on_change=send_feedback, args=(m["trace_id"], key))

    if prompt := st.chat_input("Ask about a feature, gap or configuration..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder, buf, final = st.empty(), "", None
            placeholder.markdown("_Searching the solution docs..._")
            try:
                for event, data in stream_chat(prompt, st.session_state.session_id):
                    if event == "token":
                        buf += data["text"]
                        placeholder.markdown(buf + " ▌")
                    elif event == "step" and data.get("node") == "validate":
                        buf = ""  # draft is checked; a retry may regenerate it
                        placeholder.markdown("_Checking the answer against the documents..._")
                    elif event == "final":
                        final = data
            except Exception as e:
                final = {"answer": f"Error contacting the assistant: {e}", "sources": [], "status": "ERROR"}
            final = final or {"answer": buf, "sources": [], "status": "ERROR"}
            placeholder.markdown(final["answer"])  # replaced by the validated answer
            st.session_state.messages.append({"role": "assistant", "content": final["answer"],
                                              "sources": final.get("sources", []),
                                              "trace_id": final.get("trace_id")})
        st.rerun()

elif page == "Documents":
    st.header("Indexed documents")
    c1, c2 = st.columns(2)
    if c1.button("Re-index changed docs"):
        with st.spinner("Indexing..."):
            st.json(api_post("/admin/ingest", {"force": False}))
    if c2.button("Force re-index all"):
        with st.spinner("Indexing..."):
            st.json(api_post("/admin/ingest", {"force": True}))
    try:
        st.dataframe(api_get("/documents"), use_container_width=True)
    except Exception as e:
        st.error(str(e))

else:
    st.header("Quality & usage metrics")
    days = st.slider("Window (days)", 1, 90, 7)
    try:
        m = api_get("/metrics", days=days)
        cols = st.columns(4)
        cols[0].metric("Requests", m["requests"])
        cols[1].metric("P50 latency (s)", f"{(m['latency_ms_p50'] or 0) / 1000:.1f}")
        cols[2].metric("P95 latency (s)", f"{(m['latency_ms_p95'] or 0) / 1000:.1f}")
        cols[3].metric("Thumbs-up ratio", m["thumbs_up_ratio"] if m["thumbs_up_ratio"] is not None else "-")
        cols = st.columns(3)
        cols[0].metric("Citation coverage", m["citation_coverage"] if m["citation_coverage"] is not None else "-")
        cols[1].metric("Not-in-docs rate", m["not_in_docs_rate"] if m["not_in_docs_rate"] is not None else "-")
        cols[2].metric("Failure rate", m["failure_rate"] if m["failure_rate"] is not None else "-")
        st.subheader("Status counts")
        st.json(m["status_counts"])
        st.subheader("Top unanswered questions (content gaps)")
        st.write(m["top_unanswered"] or "None")
        st.subheader("Thumbs-down questions")
        st.write(m["thumbs_down"] or "None")
    except Exception as e:
        st.error(str(e))
