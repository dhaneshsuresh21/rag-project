"""CLI: python -m app.ingestion [--force] [--docs-dir PATH]"""
from __future__ import annotations

import argparse
import json

from app.components import get_components
from app.ingestion.indexer import ingest
from app.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Word solution docs into the RAG store.")
    parser.add_argument("--force", action="store_true", help="Re-index all docs even if unchanged")
    parser.add_argument("--docs-dir", default=None, help="Override docs folder (default: config paths.docs_dir)")
    args = parser.parse_args()
    configure_logging()

    c = get_components()
    summary = ingest(c.settings, c.store, c.vectors, c.embedder, docs_dir=args.docs_dir, force=args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
