"""Summarize an eval run: metrics vs thresholds, retrieval misses, weakest RAGAS items, comparison with the baseline.
Usage: python scripts/eval_summary.py [report-YYYYMMDD-HHMMSS.json]   (default: latest report)"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval" / "reports"


def main() -> None:
    reports = sorted(glob.glob(str(REPORTS / "report-*.json")))
    if not reports:
        print("No reports found. Run: python -m eval.run")
        return
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(reports[-1])
    if not report_path.is_absolute():
        report_path = REPORTS / report_path.name
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    stamp = rep["timestamp"]
    baseline_path = REPORTS / "baseline.json"
    base = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None

    print(f"Report   : {report_path.name}  (llm={rep['llm']}, judge={rep['judge']}, prompt={rep['prompt']})")
    print(f"Baseline : {base['timestamp'] if base else '-'}")
    print(f"Gate     : {'PASSED' if rep['passed'] else 'FAILED'}")
    for f in rep["failures"]:
        print(f"  - {f}")

    print("\nMetric                         value      baseline   threshold")
    from_cfg = {}
    try:
        import yaml
        from_cfg = yaml.safe_load((ROOT / "config" / "eval.yaml").read_text(encoding="utf-8"))["thresholds"]
    except Exception:
        pass
    for k, v in rep["metrics"].items():
        b = base["metrics"].get(k) if base else None
        fmt = lambda x: "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))  # noqa: E731
        print(f"  {k:<28} {fmt(v):<10} {fmt(b):<10} {fmt(from_cfg.get(k))}")

    rows_path = REPORTS / f"rows-{stamp}.jsonl"
    if not rows_path.exists():
        return
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [r for r in rows if "ragas" not in r]
    ragas = [r["ragas"] for r in rows if "ragas" in r]

    wrong = [r for r in items if not r["refusal_correct"]]
    print(f"\nWrong answer/refusal decisions: {len(wrong)}")
    for r in wrong:
        print(f"  - [{r['category']}] got {r['status']}: {r['question'][:100]}")

    misses = [r for r in items if r["retrieval_hit"] is False]
    print(f"\nRetrieval misses (reference chunk not in context): {len(misses)}")
    for r in misses:
        print(f"  - {r['id']}: {r['question'][:100]}")
        print(f"    retrieved: {r['context_ids']}")

    if ragas:
        print("\nWeakest RAGAS items (by faithfulness, then context precision):")
        key = lambda x: (x.get("faithfulness") or 0, x.get("llm_context_precision_with_reference") or 0)  # noqa: E731
        for r in sorted(ragas, key=key)[:5]:
            print(f"  faith={r.get('faithfulness', 0):.2f}  ctx_prec={r.get('llm_context_precision_with_reference', 0):.2f}"
                  f"  relevancy={r.get('answer_relevancy', 0):.2f}  | {r['user_input'][:80]}")


if __name__ == "__main__":
    main()
