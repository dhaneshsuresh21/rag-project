from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from app.config import EmbeddingCfg


class Embedder:
    """Local Ollama embeddings with the task prefixes nomic-embed-text expects."""

    def __init__(self, cfg: EmbeddingCfg, base_url: str):
        self.cfg = cfg
        self.model = OllamaEmbeddings(model=cfg.model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = [self.cfg.doc_prefix + t for t in texts[i:i + self.cfg.batch_size]]
            out.extend(self.model.embed_documents(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(self.cfg.query_prefix + text)

    def as_langchain(self) -> OllamaEmbeddings:
        return self.model
