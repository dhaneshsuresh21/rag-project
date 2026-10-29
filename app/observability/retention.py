"""Delete chat logs, traces and feedback older than the retention period (default 90 days).
CLI: python -m app.observability.retention"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from app.config import Settings, get_settings
from app.store import Store


def purge(settings: Settings, store: Store) -> dict:
    days = settings.app.retention_days
    cutoff = time.time() - days * 86400
    result = {"retention_days": days, "rows_deleted": store.purge_older_than(cutoff), "files_deleted": []}
    cutoff_day = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    for f in Path(settings.path("logs_dir")).glob("traces-*.jsonl"):
        if f.stem.removeprefix("traces-") < cutoff_day:
            f.unlink()
            result["files_deleted"].append(f.name)
    return result


if __name__ == "__main__":
    s = get_settings()
    print(json.dumps(purge(s, Store(s.path("sqlite_path"))), indent=2))
