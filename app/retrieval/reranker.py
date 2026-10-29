from __future__ import annotations

import inspect
import logging
import math
import os
import threading

from app.config import RerankerCfg

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
log = logging.getLogger(__name__)


class Reranker:
    """Local cross-encoder. Scores are returned as sigmoid probabilities (0..1).
    If the model cannot be loaded, `available` becomes False and callers fall back to RRF order."""

    def __init__(self, cfg: RerankerCfg):
        self.cfg = cfg
        self._model = None
        self._predict_kwargs: dict = {}
        self.available = cfg.enabled
        self.error: str | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None or not self.available:
            return
        with self._lock:  # warm-up thread and a request may race here; load only once
            if self._model is not None or not self.available:
                return
            try:
                import torch
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.cfg.model, device=self.cfg.device, max_length=self.cfg.max_length,
                                     revision=self.cfg.revision, local_files_only=self.cfg.local_files_only)
                params = inspect.signature(model.predict).parameters
                for name in ("activation_fn", "activation_fct"):  # name differs across sentence-transformers versions
                    if name in params:
                        self._predict_kwargs = {name: torch.nn.Identity()}
                        break
                self._model = model
                log.info("Reranker loaded: %s@%s (local_files_only=%s)", self.cfg.model,
                         self.cfg.revision or "main", self.cfg.local_files_only)
            except Exception as e:  # graceful degradation
                self.available, self.error = False, str(e)
                log.warning("Reranker unavailable, falling back to RRF order: %s", e)

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        self._load()
        if not self.available or not candidates:
            return candidates[:top_n]
        pairs = [(query, f"{c['section_path']}\n{c['text']}") for c in candidates]
        logits = self._model.predict(pairs, show_progress_bar=False, **self._predict_kwargs)
        for c, logit in zip(candidates, logits):
            c["rerank_score"] = round(1.0 / (1.0 + math.exp(-float(logit))), 4)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
