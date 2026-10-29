from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3", "chromadb.telemetry")


def configure_logging(level: int = logging.INFO) -> None:
    """App logs at INFO; per-request HTTP client chatter (Ollama/HF calls) only at WARNING and above."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def estimate_tokens(text: str) -> int:
    """Cheap, offline token estimate (~4 chars/token for English). Good enough for chunk sizing."""
    return max(1, (len(text) + 3) // 4)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "doc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
