"""SQLite store: documents, chunks (for BM25 + neighbour expansion), chat history, traces, feedback."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, source_file TEXT, title TEXT, feature_id TEXT, client TEXT,
    doc_type TEXT, content_hash TEXT, chunk_count INTEGER, indexed_at REAL, warnings TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT, ord INTEGER, section_path TEXT, text TEXT,
    tokens INTEGER, part INTEGER, parts INTEGER, metadata TEXT
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id, ord);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY, session_id TEXT, source TEXT, created_at REAL, question TEXT,
    standalone_question TEXT, status TEXT, answer TEXT, latency_ms REAL, prompt_version TEXT,
    model TEXT, has_citation INTEGER, data TEXT
);
CREATE INDEX IF NOT EXISTS ix_traces_created ON traces(created_at);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, rating INTEGER, comment TEXT, created_at REAL
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---------- meta ----------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, value))

    def bump_index_version(self) -> str:
        v = str(time.time_ns())
        self.set_meta("index_version", v)
        return v

    # ---------- documents / chunks ----------
    def get_document(self, doc_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY title")]

    def replace_document(self, doc: dict, chunks: list[dict]) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc["doc_id"],))
            c.execute(
                "INSERT OR REPLACE INTO documents VALUES (:doc_id,:source_file,:title,:feature_id,:client,"
                ":doc_type,:content_hash,:chunk_count,:indexed_at,:warnings)", doc)
            c.executemany(
                "INSERT INTO chunks VALUES (:chunk_id,:doc_id,:ord,:section_path,:text,:tokens,:part,:parts,:metadata)",
                [{**ch, "metadata": json.dumps(ch["metadata"])} for ch in chunks])

    def delete_document(self, doc_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    @staticmethod
    def _chunk_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"] or "{}")
        return d

    def all_chunks(self) -> list[dict]:
        with self.conn() as c:
            return [self._chunk_row(r) for r in c.execute("SELECT * FROM chunks ORDER BY doc_id, ord")]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})"
        with self.conn() as c:
            return {r["chunk_id"]: self._chunk_row(r) for r in c.execute(q, chunk_ids)}

    def section_siblings(self, doc_id: str, section_path: str) -> list[dict]:
        with self.conn() as c:
            return [self._chunk_row(r) for r in c.execute(
                "SELECT * FROM chunks WHERE doc_id=? AND section_path=? ORDER BY ord", (doc_id, section_path))]

    # ---------- chat history ----------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                      (session_id, role, content, time.time()))

    def get_history(self, session_id: str, limit: int) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                             (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ---------- traces / feedback ----------
    def insert_trace(self, row: dict) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO traces VALUES (:trace_id,:session_id,:source,:created_at,:question,"
                ":standalone_question,:status,:answer,:latency_ms,:prompt_version,:model,:has_citation,:data)",
                {**row, "data": json.dumps(row.get("data", {}), default=str)})

    def add_feedback(self, trace_id: str, rating: int, comment: str | None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO feedback(trace_id,rating,comment,created_at) VALUES(?,?,?,?)",
                      (trace_id, rating, comment, time.time()))

    def traces_since(self, since_ts: float, source: str | None = "api") -> list[dict]:
        q = ("SELECT t.trace_id, t.created_at, t.question, t.status, t.latency_ms, t.has_citation, t.model, "
             "t.prompt_version, (SELECT rating FROM feedback f WHERE f.trace_id=t.trace_id "
             "ORDER BY f.id DESC LIMIT 1) AS rating FROM traces t WHERE t.created_at>=?")
        args: list = [since_ts]
        if source:
            q += " AND t.source=?"
            args.append(source)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY t.created_at DESC", args)]

    def purge_older_than(self, cutoff_ts: float) -> dict[str, int]:
        with self.conn() as c:
            counts = {}
            for table in ("messages", "traces", "feedback"):
                counts[table] = c.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff_ts,)).rowcount
        return counts
