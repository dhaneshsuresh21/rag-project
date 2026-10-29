"""End-to-end smoke test through the full pipeline (needs Ollama + indexed docs).
Usage: python scripts/smoke_test.py ["question 1" "question 2" ...]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.service import RagService  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is the job ID of the manual fire job for the sales data feed extraction?",
    "Which parameter has to be set up for the SPG SSA reconciliation round-off?",
    "Can users edit the passenger name on the FIM capture screen?",
    "How do I configure SAML single sign-on for the PRA portal?",  # should be refused (not in docs)
]


def main() -> int:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    svc = RagService()
    failures = 0
    for q in questions:
        r = svc.ask(q, source="smoke", persist=False, include_context=True)
        failures += r["status"] == "ERROR"
        print("=" * 100)
        print(f"Q: {q}\nstatus={r['status']}  latency={r['latency_ms'] / 1000:.1f}s  "
              f"context={[c['chunk_id'].split('::')[-1] for c in r.get('context', [])]}")
        print(f"A: {r['answer']}")
        for s in r["sources"]:
            print(f"   [{s['cid']}] {s['doc_title']} > {s['section_path']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
