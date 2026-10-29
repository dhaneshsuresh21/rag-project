"""Builds shared components once (store, vector store, embedder, retriever, reranker, LLMs)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_ollama import ChatOllama

from app.config import LLMCfg, Settings, get_settings
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore
from app.store import Store


def make_chat_llm(cfg: LLMCfg, base_url: str, json_mode: bool = False) -> ChatOllama:
    kwargs = dict(model=cfg.model, base_url=base_url, temperature=cfg.temperature, num_ctx=cfg.num_ctx,
                  num_predict=cfg.num_predict, keep_alive=cfg.keep_alive,
                  client_kwargs={"timeout": cfg.timeout_s})
    if cfg.num_thread:
        kwargs["num_thread"] = cfg.num_thread
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


@dataclass
class Components:
    settings: Settings
    store: Store
    vectors: VectorStore
    embedder: Embedder
    retriever: HybridRetriever
    reranker: Reranker
    llm: ChatOllama
    rewrite_llm: ChatOllama


@lru_cache
def get_components() -> Components:
    s = get_settings()
    store = Store(s.path("sqlite_path"))
    vectors = VectorStore(s.path("chroma_dir"), s.vector_store.collection)
    embedder = Embedder(s.embeddings, s.ollama.base_url)
    return Components(
        settings=s,
        store=store,
        vectors=vectors,
        embedder=embedder,
        retriever=HybridRetriever(s.retrieval, store, vectors, embedder),
        reranker=Reranker(s.reranker),
        llm=make_chat_llm(s.llm, s.ollama.base_url),
        rewrite_llm=make_chat_llm(s.rewrite_llm, s.ollama.base_url),
    )
