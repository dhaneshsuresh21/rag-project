# AGENTS.md

## Commands (Windows / PowerShell, run from this folder)
- Python: `.\.venv\Scripts\python.exe`
- Install deps: `.\.venv\Scripts\python.exe -m pip install --uploaded-prior-to P7D -r requirements.txt`
- Tests (no Ollama needed): `.\.venv\Scripts\python.exe -m pytest -q`
- Dry-run chunking: `.\.venv\Scripts\python.exe scripts\preview_chunks.py`
- Index docs: `.\.venv\Scripts\python.exe -m app.ingestion [--force]`
- API: `.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000`
- UI: `.\.venv\Scripts\python.exe -m streamlit run ui/streamlit_app.py`
- Eval: `.\.venv\Scripts\python.exe -m eval.run [--no-ragas]`
- Smoke test (needs Ollama + index): `.\.venv\Scripts\python.exe scripts\smoke_test.py`
- Production: `docker compose up -d --build` on the Ubuntu server (never ship `.venv`)

## Conventions
- Everything must stay open source and local (Ollama). No external LLM/embedding APIs.
- Behaviour lives in `config/*.yaml` and versioned `prompts/*.yaml`. Bump the prompt `version` on any prompt change.
- Set `PYTHONIOENCODING=utf-8` when printing doc content to the Windows console.
- Solution docs follow the "PRA Final Solution Template": the real title is in the Subtitle style, there is a TOC, and empty sections contain "NA".
- Run `eval.run` before merging prompt, config, chunking or model changes.
- Inspect traces in `logs/traces-*.jsonl` (`raw_answer`, `citation`, span timings) when an answer looks wrong.
- Always set `ADMIN_TOKEN` in production (it protects `/admin/ingest`).
- No Hugging Face calls at runtime: keep `HF_HUB_OFFLINE=1` in `.env`. The reranker revision is pinned in `config/models.yaml` and must match `RERANKER_REVISION` in the `Dockerfile`.
- Entry points call `app.utils.configure_logging()` (HTTP client loggers at WARNING). Use it for new CLIs.
- Golden set: `eval/golden/*.jsonl`. Set `"verified": true` only after a human has checked an item.
