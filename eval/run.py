"""Offline evaluation: runs the golden set through the pipeline, computes RAGAS + custom metrics, gates on thresholds.
Usage: python -m eval.run [--golden "eval/golden/*.jsonl"] [--verified-only] [--no-ragas] [--limit N]
                          [--save-baseline] [--no-gate]
Exit code 1 when a threshold is missed or a metric regresses beyond the tolerance vs the baseline (CI-ready)."""
from __future__ import annotations

import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

from app.components import get_components, make_chat_llm  # noqa: E402
from app.config import ROOT  # noqa: E402
from app.prompts import prompt_ref  # noqa: E402
from app.rag.citations import strip_citations  # noqa: E402
from app.rag.service import RagService  # noqa: E402
from app.utils import percentile  # noqa: E402

REPORTS = ROOT / "eval" / "reports"
RAGAS_NAME_MAP = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy",
                  "llm_context_precision_with_reference": "context_precision", "context_recall": "context_recall"}


def load_golden(pattern: str, verified_only: bool) -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(ROOT / pattern) if not os.path.isabs(pattern) else pattern)):
        with open(f, encoding="utf-8") as fh:
            items += [json.loads(line) for line in fh if line.strip()]
    return [i for i in items if i.get("verified") or not verified_only]


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_ragas(rows: list[dict]) -> tuple[dict, list[dict]]:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    c = get_components()
    s = c.settings
    samples = [SingleTurnSample(user_input=r["question"], retrieved_contexts=r["contexts"] or [""],
                                response=r["response"], reference=r["reference_answer"]) for r in rows]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(make_chat_llm(s.judge_llm, s.ollama.base_url)),
        embeddings=LangchainEmbeddingsWrapper(c.embedder.as_langchain()),
        run_config=RunConfig(timeout=s.eval.ragas.timeout_s, max_workers=s.eval.ragas.max_workers, max_retries=2),
        show_progress=True,
    )
    df = result.to_pandas()
    summary, per_row = {}, df.to_dict(orient="records")
    for col, key in RAGAS_NAME_MAP.items():
        if col in df.columns:
            vals = [float(v) if v is not None else float("nan") for v in df[col].tolist()]
            summary[key] = _mean(vals)
            summary[f"{key}_nan_rate"] = round(sum(math.isnan(v) for v in vals) / len(vals), 3) if vals else None
    return summary, per_row


def main() -> int:
    c = get_components()
    s = c.settings
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden/*.jsonl")
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    items = load_golden(args.golden, args.verified_only)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No golden items found. Run: python -m eval.generate_testset")
        return 1

    svc = RagService(c)
    rows = []
    for n, item in enumerate(items, start=1):
        res = svc.ask(item["question"], source="eval", persist=False, include_context=True)
        unanswerable = item.get("category") == "unanswerable"
        ctx_ids = [x["chunk_id"] for x in res.get("context", [])]
        refs = item.get("reference_chunk_ids") or []
        rows.append({
            "id": item.get("id"), "question": item["question"], "category": item.get("category"),
            "verified": item.get("verified", False), "reference_answer": item.get("reference_answer", ""),
            "status": res["status"], "response": strip_citations(res["answer"]), "sources": res["sources"],
            "contexts": [x["text"] for x in res.get("context", [])], "context_ids": ctx_ids,
            "latency_ms": res["latency_ms"], "trace_id": res["trace_id"],
            "refusal_correct": (res["status"] == "NOT_IN_DOCS") if unanswerable else (res["status"] == "ANSWERED"),
            "retrieval_hit": (any(r in ctx_ids for r in refs) if refs else None),
        })
        print(f"[{n}/{len(items)}] {res['status']:<11} {res['latency_ms'] / 1000:6.1f}s  {item['question'][:80]}")

    answered = [r for r in rows if r["status"] == "ANSWERED"]
    metrics = {
        "items": len(rows),
        "answerable": sum(r["category"] != "unanswerable" for r in rows),
        "unanswerable": sum(r["category"] == "unanswerable" for r in rows),
        "errors": sum(r["status"] == "ERROR" for r in rows),
        "refusal_correctness": _mean([float(r["refusal_correct"]) for r in rows]),
        "citation_coverage": _mean([float(bool(r["sources"])) for r in answered]),
        "retrieval_hit_rate": _mean([float(r["retrieval_hit"]) for r in rows if r["retrieval_hit"] is not None]),
        "latency_ms_p50": percentile([r["latency_ms"] for r in rows], 50),
        "latency_ms_p95": percentile([r["latency_ms"] for r in rows], 95),
    }

    ragas_rows: list[dict] = []
    ragas_input = [r for r in answered if r["category"] != "unanswerable"]
    if s.eval.ragas.enabled and not args.no_ragas and ragas_input:
        print(f"\nRunning RAGAS on {len(ragas_input)} answered items (judge: {s.judge_llm.model})...")
        ragas_summary, ragas_rows = run_ragas(ragas_input)
        metrics.update(ragas_summary)

    # ---- gate ----
    failures = []
    for key, minimum in s.eval.thresholds.items():
        val = metrics.get(key)
        if val is not None and val < minimum:
            failures.append(f"{key}={val} < threshold {minimum}")
    baseline_path = REPORTS / "baseline.json"
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
        for key in s.eval.thresholds:
            if metrics.get(key) is not None and base.get(key) is not None \
                    and metrics[key] < base[key] - s.eval.regression_tolerance:
                failures.append(f"{key} regressed: {metrics[key]} vs baseline {base[key]}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {"timestamp": stamp, "llm": s.llm.model, "judge": s.judge_llm.model, "embeddings": s.embeddings.model,
              "reranker": s.reranker.model if s.reranker.enabled else None, "prompt": prompt_ref("answer"),
              "golden": args.golden, "verified_only": args.verified_only, "metrics": metrics,
              "failures": failures, "passed": not failures}
    (REPORTS / f"report-{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(REPORTS / f"rows-{stamp}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        for r in ragas_rows:
            f.write(json.dumps({"ragas": r}, ensure_ascii=False, default=str) + "\n")
    if args.save_baseline:
        baseline_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        thr = s.eval.thresholds.get(k)
        print(f"  {k:<32} {v!s:<10} {'(min ' + str(thr) + ')' if thr is not None else ''}")
    print(f"\nReport: {REPORTS / f'report-{stamp}.json'}")
    if failures:
        print("\nGATE FAILED:\n  - " + "\n  - ".join(failures))
        return 0 if args.no_gate else 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
