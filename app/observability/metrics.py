from __future__ import annotations

import time
from collections import Counter

from app.store import Store
from app.utils import percentile


def compute_metrics(store: Store, days: int = 7, source: str | None = "api") -> dict:
    rows = store.traces_since(time.time() - days * 86400, source=source)
    n = len(rows)
    statuses = Counter(r["status"] for r in rows)
    answered = [r for r in rows if r["status"] == "ANSWERED"]
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    rated = [r["rating"] for r in rows if r["rating"] is not None]

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    return {
        "window_days": days,
        "requests": n,
        "status_counts": dict(statuses),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "citation_coverage": ratio(sum(r["has_citation"] for r in answered), len(answered)),
        "not_in_docs_rate": ratio(statuses.get("NOT_IN_DOCS", 0), n),
        "failure_rate": ratio(statuses.get("ERROR", 0), n),
        "feedback_count": len(rated),
        "thumbs_up_ratio": ratio(sum(1 for r in rated if r > 0), len(rated)),
        "top_unanswered": [r["question"] for r in rows if r["status"] == "NOT_IN_DOCS"][:20],
        "thumbs_down": [r["question"] for r in rows if r["rating"] is not None and r["rating"] < 0][:20],
    }
