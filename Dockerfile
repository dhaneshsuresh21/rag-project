# Production image for the RAG API and the web UI (same image, different command).
# Build:  docker build -t rag-solution-docs:0.1.0 .
FROM python:3.12-slim

ARG PIP_VERSION=26.2.1
# Resolve dependencies as of a fixed date -> reproducible builds + only packages published >= 7 days before it
ARG PIP_UPLOADED_PRIOR_TO=2026-10-13T00:00:00Z
ARG TORCH_VERSION=2.14.1
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Must match config/models.yaml -> reranker.revision
ARG RERANKER_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    ANONYMIZED_TELEMETRY=False \
    RAGAS_DO_NOT_TRACK=true \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LANGCHAIN_TRACING_V2=false

WORKDIR /app

# CPU-only PyTorch (avoids ~2 GB of CUDA libraries); then the pinned app dependencies
COPY requirements.txt .
RUN python -m pip install "pip==${PIP_VERSION}" \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && pip install --uploaded-prior-to "${PIP_UPLOADED_PRIOR_TO}" -r requirements.txt

# Bake the reranker into the image so the container runs fully offline
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}', revision='${RERANKER_REVISION}')"
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY ui ./ui
COPY eval ./eval
COPY scripts ./scripts
COPY config ./config
COPY prompts ./prompts
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 rag \
 && mkdir -p /app/storage /app/logs /app/data/docs /app/eval/golden /app/eval/reports \
 && chown -R rag:rag /app /opt/hf-cache
USER rag

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
