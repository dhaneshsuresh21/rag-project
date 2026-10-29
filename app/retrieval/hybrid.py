from __future__ import annotations

import threading

from app.config import RetrievalCfg
from app.ingestion.chunker import embed_text
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.fusion import rrf_fuse
from app.retrieval.vector_store import VectorStore
from app.store import Store


class HybridRetriever:
    """BM25 (keyword) + vector (semantic) retrieval fused with RRF. Rebuilds BM25 when the index changes."""

    def __init__(self, cfg: RetrievalCfg, store: Store, vectors: VectorStore, embedder: Embedder):
        self.cfg, self.store, self.vectors, self.embedder = cfg, store, vectors, embedder
        self._bm25: BM25Index | None = None
        self._bm25_version: str | None = None
        self._lock = threading.Lock()

    def _ensure_bm25(self) -> BM25Index:
        version = self.store.get_meta("index_version", "0")
        with self._lock:
            if self._bm25 is None or version != self._bm25_version:
                chunks = self.store.all_chunks()
                self._bm25 = BM25Index([c["chunk_id"] for c in chunks], [embed_text(c) for c in chunks])
                self._bm25_version = version
            return self._bm25

    def retrieve(self, query: str) -> list[dict]:
        bm25_hits = self._ensure_bm25().search(query, self.cfg.bm25_top_k)
        vector_hits = self.vectors.query(self.embedder.embed_query(query), self.cfg.vector_top_k)
        fused = rrf_fuse({"bm25": bm25_hits, "vector": vector_hits}, k=self.cfg.rrf_k)[: self.cfg.fused_top_k]
        rows = self.store.get_chunks([f["chunk_id"] for f in fused])
        return [{**rows[f["chunk_id"]], **f} for f in fused if f["chunk_id"] in rows]

    def expand_and_budget(self, ranked: list[dict]) -> list[dict]:
        """Add sibling parts of split sections (neighbour expansion), then trim to the context token budget."""
        out, seen = [], set()
        for c in ranked:
            group = [c]
            if self.cfg.neighbour_expansion and c.get("parts", 1) > 1:
                group = [s if s["chunk_id"] != c["chunk_id"] else c
                         for s in self.store.section_siblings(c["doc_id"], c["section_path"])]
            for g in group:
                if g["chunk_id"] not in seen:
                    seen.add(g["chunk_id"])
                    out.append(g)
        budget, final = self.cfg.max_context_tokens, []
        for c in out:
            if final and budget - c["tokens"] < 0:
                continue
            budget -= c["tokens"]
            final.append(c)
        return final
