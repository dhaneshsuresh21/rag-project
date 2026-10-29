from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402


class VectorStore:
    """Embedded, persistent Chroma collection. Embeddings are always supplied by us (no default EF)."""

    def __init__(self, path: Path, collection: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str],
               metadatas: list[dict]) -> None:
        if ids:
            self.col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.col.count()

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        n = min(k, self.count())
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[embedding], n_results=n, include=["distances"])
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]
