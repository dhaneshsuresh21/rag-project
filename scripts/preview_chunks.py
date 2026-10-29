"""Dry-run parsing + chunking (no Ollama needed).
Usage: python scripts/preview_chunks.py [docs_dir] [--show N]   (--show prints the first N chunk texts per doc)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.ingestion.chunker import chunk_document, embed_text  # noqa: E402
from app.ingestion.docx_parser import parse_docx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs_dir", nargs="?", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    s = get_settings()
    for p in sorted(Path(args.docs_dir or s.path("docs_dir")).glob("*.docx")):
        d = parse_docx(p, s.parsing)
        chunks = chunk_document(d, s.chunking)
        toks = [c["tokens"] for c in chunks] or [0]
        print("=" * 100)
        print(f"{p.name}\n  title={d.title!r}  id={d.feature_id!r}  type={d.doc_type}  client={d.client}  "
              f"version={d.version!r}\n  warnings={d.warnings}")
        print(f"  chunks={len(chunks)}  tokens min/avg/max = {min(toks)}/{sum(toks) // len(toks)}/{max(toks)}")
        for c in chunks:
            print(f"   {c['ord']:>3} {c['tokens']:>4}t  part {c['part']}/{c['parts']}  {c['section_path'][:90]}")
        for c in chunks[: args.show]:
            print("-" * 100)
            print(embed_text(c)[:1500])


if __name__ == "__main__":
    main()
