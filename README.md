# rag-solution-docs

On-prem RAG assistant over Word **solution documents**. 100% open source, self-hosted LLM, no external APIs.
Design: see `../02-HLD.md` (v0.5), `../05-model-selection.md` and the walkthrough `../06-how-it-works.md`.

```
.docx ─► parse (python-docx, template-aware) ─► structure-aware chunks ─► nomic-embed-text ─► Chroma + BM25 + SQLite
question ─► rewrite (follow-ups) ─► BM25 + vector ─► RRF ─► MiniLM cross-encoder ─► gate (τ) ─► phi4-mini ─► citation validator
```

## Prerequisites
- Python 3.11+ (tested with 3.12)
- [Ollama](https://ollama.com) running locally, with:
  ```powershell
  ollama pull phi4-mini
  ollama pull nomic-embed-text
  ```
- The reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB) at the pinned revision `233902d2…` (`config/models.yaml`) must be in the local Hugging Face cache. The app loads it **offline only** (`HF_HUB_OFFLINE=1`, `local_files_only: true`). On a fresh machine, download it once:
  ```powershell
  $env:HF_HUB_OFFLINE="0"; .\.venv\Scripts\python.exe -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', revision='233902d25c440f23af6f7d6e94d2946bac0bee0a')"
  ```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "pip==26.2.1"
.\.venv\Scripts\python.exe -m pip install --uploaded-prior-to P7D -r requirements.txt
copy .env.example .env      # sets HF_HUB_OFFLINE=1 + telemetry off; set ADMIN_TOKEN for shared/prod use
```

## Run
```powershell
# 1. Put .docx solution docs in data/docs/, then index (only changed docs are re-embedded)
.\.venv\Scripts\python.exe -m app.ingestion            # --force to re-index everything

# 2. API (terminal 1)
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 3. Web UI (terminal 2) -> http://localhost:8501
.\.venv\Scripts\python.exe -m streamlit run ui/streamlit_app.py
```

API endpoints: `GET /health`, `POST /chat`, `POST /chat/stream` (SSE), `POST /feedback`, `POST /admin/ingest`
(needs header `X-Admin-Token` when `ADMIN_TOKEN` is set), `GET /documents`, `GET /metrics?days=7`. OpenAPI docs: http://127.0.0.1:8000/docs

On startup the API warms up the embeddings, reranker and LLM in the background (log line `Warm-up: {...ok}`).
HTTP client logs (httpx, huggingface_hub, …) are shown only at WARNING. What the startup log means: `../../log_explained.md`.

## Production (Ubuntu + Docker)
`.venv` is for local development only and is never shipped. Production runs the **Docker image** built from `Dockerfile`:
pinned dependencies, CPU-only PyTorch, reranker baked in, non-root user, offline at runtime.
```bash
cp .env.example .env                 # set ADMIN_TOKEN
docker compose up -d --build         # ollama + model pull + api (127.0.0.1:8000) + ui (:8501)
curl -X POST http://localhost:8000/admin/ingest -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Content-Type: application/json" -d '{}'
```
Docs are mounted read-only from `./data/docs`. The index and traces persist in the `rag-storage` and `rag-logs` volumes.
For a GPU server, uncomment the `deploy` block of the `ollama` service. Details: `../06-how-it-works.md`.

## Evaluate (RAGAS + custom metrics)
```powershell
.\.venv\Scripts\python.exe -m eval.generate_testset     # LLM-generated Q&A -> eval/golden/generated-*.jsonl
#   (already done: generated-20261020-182237.jsonl, 24 items, 0 verified)
# review the file, set "verified": true on correct items
.\.venv\Scripts\python.exe -m eval.run --save-baseline  # first run: store baseline
.\.venv\Scripts\python.exe -m eval.run                  # later runs: gate vs thresholds + baseline (exit 1 on fail)
.\.venv\Scripts\python.exe -m eval.run --no-ragas       # fast: custom metrics only (no judge LLM)
```
Thresholds: `config/eval.yaml`. Reports: `eval/reports/`.

## Tools
```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py                  # end-to-end questions through the pipeline
.\.venv\Scripts\python.exe scripts\preview_chunks.py [--show 3]   # dry-run parse + chunk (no Ollama)
.\.venv\Scripts\python.exe scripts\inspect_docs.py data\docs      # doc audit: styles, headings, tables
.\.venv\Scripts\python.exe -m app.observability.retention         # purge logs/traces older than 90 days
.\.venv\Scripts\python.exe -m pytest -q                           # unit tests (no Ollama needed)
```

## Layout
| Path | Purpose |
|---|---|
| `app/ingestion/` | `docx_parser.py` (template-aware), `chunker.py` (sections, split/overlap, sibling merge), `indexer.py` (incremental), CLI |
| `app/retrieval/` | `embeddings.py`, `vector_store.py` (Chroma), `bm25.py`, `fusion.py` (RRF), `reranker.py`, `hybrid.py` |
| `app/rag/` | `graph.py` (LangGraph pipeline + repair retry), `citations.py` (citation, empty-answer, identifier checks), `service.py` (tracing, history, warm-up) |
| `app/utils.py` | Token estimate, hashing, percentiles, `configure_logging()` (quiet HTTP logs) |
| `app/api/main.py` | FastAPI backend |
| `app/observability/` | JSONL + SQLite tracing, metrics, 90-day retention |
| `ui/streamlit_app.py` | Standalone web chat (Chat / Documents / Metrics pages) |
| `eval/` | Test-set generation, RAGAS + custom metrics, CI-style gate |
| `config/` | `app.yaml`, `models.yaml`, `retrieval.yaml`, `eval.yaml` (all behaviour is config) |
| `prompts/` | Versioned prompts (`answer`, `rewrite`, `testset_gen`) |
| `data/docs/` | Solution docs (.docx) |
| `storage/`, `logs/` | Chroma, SQLite, traces (git-ignored) |

## Notes
- `langchain-community` is pinned to 0.4.1 because ragas 0.4.3 imports a module removed in 0.4.2.
- The `ResourceTracker ... _recursion_count` message on exit comes from the `multiprocess` package on Python 3.12.0. It is harmless.
- Telemetry is disabled for Chroma, RAGAS, Hugging Face, LangChain (`.env.example`) and Streamlit (`.streamlit/config.toml`).
- The reranker revision is pinned in two places that must match: `config/models.yaml` (`reranker.revision`) and the `Dockerfile` (`RERANKER_REVISION`).
- `scripts/make_sample_docs.py` creates synthetic sample docs. It's unused now that real docs are in `data/docs/`.
