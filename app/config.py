from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class AppCfg(BaseModel):
    name: str = "Solution Docs Assistant"
    show_sources: bool = True
    history_turns: int = 6
    require_citations: bool = True
    max_generation_retries: int = 1
    retention_days: int = 90
    not_in_docs_message: str = "I couldn't find this in the solution documents."
    error_message: str = "The assistant is temporarily unavailable."


class PathsCfg(BaseModel):
    docs_dir: str = "data/docs"
    storage_dir: str = "storage"
    chroma_dir: str = "storage/chroma"
    sqlite_path: str = "storage/app.db"
    logs_dir: str = "logs"


class VectorStoreCfg(BaseModel):
    collection: str = "solution_docs"


class ApiCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class OllamaCfg(BaseModel):
    base_url: str = "http://localhost:11434"


class LLMCfg(BaseModel):
    model: str = "phi4-mini"
    temperature: float = 0.0
    num_ctx: int = 8192
    num_predict: int = 400
    num_thread: int | None = None
    keep_alive: str = "30m"
    timeout_s: int = 180


class EmbeddingCfg(BaseModel):
    model: str = "nomic-embed-text"
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    batch_size: int = 16


class RerankerCfg(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    revision: str | None = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    local_files_only: bool = True
    device: str = "cpu"
    max_length: int = 512


class ParsingCfg(BaseModel):
    feature_id_pattern: str = r"(?<![A-Za-z0-9])(?:PRAC|GAP|FEAT|FEATURE|PRJ|PROJ)[-_ ]?\d+(?!\d)"
    client_label_pattern: str = r"^\s*(?:client|customer)\s*[:\-]\s*(.+)$"
    pseudo_heading_max_chars: int = 120
    title_styles: list[str] = ["Subtitle", "Title"]
    generic_titles: list[str] = ["Final Solution Template", "PRA Final Solution Template"]
    title_prefix_pattern: str = r"^\s*(?:solution|title)\s*[:\-]\s*"
    ignore_headings: list[str] = ["PRA Final Solution Template", "Contents"]
    front_matter_styles: list[str] = ["Version", "Date Month Year", "Title Date", "Subtitle", "Title"]
    skip_style_prefixes: list[str] = ["toc"]
    bold_subheadings: bool = True


class ChunkingCfg(BaseModel):
    max_tokens: int = 800
    target_tokens: int = 650
    overlap_tokens: int = 100
    min_tokens: int = 60
    empty_markers: list[str] = ["na", "n/a", "n.a.", "none", "nil", "not applicable", "-", "tbd", "no impact",
                                "not required", "no changes"]


class RetrievalCfg(BaseModel):
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    fused_top_k: int = 10
    rerank_top_n: int = 5
    min_rerank_score: float = 0.02
    neighbour_expansion: bool = True
    max_context_tokens: int = 2500


class RagasCfg(BaseModel):
    enabled: bool = True
    timeout_s: int = 900
    max_workers: int = 1


class TestsetCfg(BaseModel):
    questions_per_doc: int = 8
    unanswerable_per_doc: int = 2


class EvalCfg(BaseModel):
    thresholds: dict[str, float] = {}
    regression_tolerance: float = 0.02
    ragas: RagasCfg = RagasCfg()
    testset: TestsetCfg = TestsetCfg()


class Settings(BaseModel):
    app: AppCfg = AppCfg()
    paths: PathsCfg = PathsCfg()
    vector_store: VectorStoreCfg = VectorStoreCfg()
    api: ApiCfg = ApiCfg()
    ollama: OllamaCfg = OllamaCfg()
    llm: LLMCfg = LLMCfg()
    rewrite_llm: LLMCfg = LLMCfg(num_ctx=4096, num_predict=64)
    judge_llm: LLMCfg = LLMCfg(num_predict=1024)
    embeddings: EmbeddingCfg = EmbeddingCfg()
    reranker: RerankerCfg = RerankerCfg()
    parsing: ParsingCfg = ParsingCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    eval: EvalCfg = EvalCfg()

    def path(self, name: str) -> Path:
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else ROOT / p


def load_settings(config_dir: Path | None = None) -> Settings:
    config_dir = config_dir or ROOT / "config"
    data: dict = {}
    for f in ("app.yaml", "models.yaml", "retrieval.yaml"):
        fp = config_dir / f
        if fp.exists():
            data.update(yaml.safe_load(fp.read_text(encoding="utf-8")) or {})
    eval_fp = config_dir / "eval.yaml"
    if eval_fp.exists():
        data["eval"] = yaml.safe_load(eval_fp.read_text(encoding="utf-8")) or {}

    s = Settings.model_validate(data)
    if url := os.getenv("OLLAMA_BASE_URL"):
        s.ollama.base_url = url
    if model := os.getenv("RAG_LLM_MODEL"):
        s.llm.model = model
        s.rewrite_llm.model = model
    if judge := os.getenv("RAG_JUDGE_MODEL"):
        s.judge_llm.model = judge
    return s


@lru_cache
def get_settings() -> Settings:
    return load_settings()
