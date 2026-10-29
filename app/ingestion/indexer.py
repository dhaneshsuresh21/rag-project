"""Incremental ingestion: parse changed .docx files, chunk, embed, and replace their chunks in Chroma + SQLite."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.config import Settings
from app.ingestion.chunker import chunk_document, embed_text
from app.ingestion.docx_parser import parse_docx
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore
from app.store import Store
from app.utils import sha256_file, slugify

log = logging.getLogger(__name__)


def _chroma_meta(chunk: dict) -> dict:
    md = chunk["metadata"]
    return {
        "doc_id": chunk["doc_id"], "ord": chunk["ord"], "section_path": chunk["section_path"],
        "part": chunk["part"], "parts": chunk["parts"], "doc_title": md["doc_title"],
        "feature_id": md["feature_id"], "client": md["client"], "doc_type": md["doc_type"],
    }


def ingest(settings: Settings, store: Store, vectors: VectorStore, embedder: Embedder,
           docs_dir: Path | None = None, force: bool = False) -> dict:
    docs_dir = Path(docs_dir or settings.path("docs_dir"))
    files = sorted(p for p in docs_dir.glob("**/*.docx") if not p.name.startswith("~$"))
    summary = {"indexed": [], "skipped": [], "removed": [], "failed": [], "warnings": {}}
    seen: set[str] = set()

    for path in files:
        doc_id = slugify(path.stem)
        seen.add(doc_id)
        existing = store.get_document(doc_id)
        if existing and not force and existing["content_hash"] == sha256_file(path):
            summary["skipped"].append(path.name)
            continue
        try:
            t0 = time.perf_counter()
            parsed = parse_docx(path, settings.parsing)
            chunks = chunk_document(parsed, settings.chunking)
            texts = [embed_text(c) for c in chunks]
            embeddings = embedder.embed_documents(texts) if chunks else []
            vectors.delete_doc(doc_id)
            vectors.upsert([c["chunk_id"] for c in chunks], embeddings, texts, [_chroma_meta(c) for c in chunks])
            store.replace_document({
                "doc_id": doc_id, "source_file": str(path), "title": parsed.title,
                "feature_id": parsed.feature_id, "client": parsed.client, "doc_type": parsed.doc_type,
                "content_hash": parsed.content_hash, "chunk_count": len(chunks), "indexed_at": time.time(),
                "warnings": json.dumps(parsed.warnings),
            }, chunks)
            summary["indexed"].append({"file": path.name, "doc_id": doc_id, "title": parsed.title,
                                       "chunks": len(chunks), "seconds": round(time.perf_counter() - t0, 2)})
            if parsed.warnings:
                summary["warnings"][path.name] = parsed.warnings
        except Exception as e:
            log.exception("Failed to ingest %s", path)
            summary["failed"].append({"file": path.name, "error": str(e)})

    for doc in store.list_documents():  # docs removed from the folder
        if doc["doc_id"] not in seen:
            vectors.delete_doc(doc["doc_id"])
            store.delete_document(doc["doc_id"])
            summary["removed"].append(doc["doc_id"])

    if summary["indexed"] or summary["removed"]:
        store.bump_index_version()
    summary["total_documents"] = len(store.list_documents())
    summary["total_chunks"] = vectors.count()
    return summary
