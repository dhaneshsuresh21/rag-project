"""Generate a golden Q&A set from the indexed docs using the local LLM.
Usage: python -m eval.generate_testset [--questions-per-doc 8] [--unanswerable-per-doc 2]
Output: eval/golden/generated-<timestamp>.jsonl with verified=false (review, then set verified=true)."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from app.components import get_components, make_chat_llm
from app.config import ROOT
from app.prompts import load_prompt, prompt_ref


def _parse_items(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    return [i for i in items if isinstance(i, dict) and str(i.get("question", "")).strip()]


def main() -> None:
    c = get_components()
    s = c.settings
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-per-doc", type=int, default=s.eval.testset.questions_per_doc)
    ap.add_argument("--unanswerable-per-doc", type=int, default=s.eval.testset.unanswerable_per_doc)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    llm = make_chat_llm(s.judge_llm.model_copy(update={"temperature": 0.3}), s.ollama.base_url, json_mode=True)
    p = load_prompt("testset_gen")
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for ch in c.store.all_chunks():
        by_doc[ch["doc_id"]].append(ch)

    records: list[dict] = []
    for doc in c.store.list_documents():
        chunks = [ch for ch in by_doc[doc["doc_id"]] if ch["tokens"] >= 30]
        if not chunks:
            continue
        random.shuffle(chunks)
        alloc = defaultdict(int)  # spread the questions over the chunks
        for i in range(args.questions_per_doc):
            alloc[chunks[i % len(chunks)]["chunk_id"]] += 1
        chunk_map = {ch["chunk_id"]: ch for ch in chunks}

        for cid, n in alloc.items():
            ch = chunk_map[cid]
            msg = llm.invoke([SystemMessage(p["system"]), HumanMessage(p["user"].format(
                doc_title=doc["title"], section_path=ch["section_path"], text=ch["text"], n=n))])
            for item in _parse_items(msg.content)[:n]:
                records.append({
                    "question": item["question"].strip(), "reference_answer": str(item.get("answer", "")).strip(),
                    "reference_chunk_ids": [cid], "reference_sections": [f"{doc['title']} > {ch['section_path']}"],
                    "doc_id": doc["doc_id"], "category": "answerable"})
            print(f"  {doc['doc_id']} :: {ch['section_path']} -> {n} question(s)")

        if args.unanswerable_per_doc:
            sections = "\n".join(sorted({f"- {ch['section_path']}" for ch in by_doc[doc["doc_id"]]}))
            msg = llm.invoke([SystemMessage(p["unanswerable_system"]), HumanMessage(p["unanswerable_user"].format(
                doc_title=doc["title"], sections=sections, n=args.unanswerable_per_doc))])
            for item in _parse_items(msg.content)[: args.unanswerable_per_doc]:
                records.append({"question": item["question"].strip(), "reference_answer": "NOT_IN_DOCS",
                                "reference_chunk_ids": [], "reference_sections": [], "doc_id": doc["doc_id"],
                                "category": "unanswerable"})

    out_dir = ROOT / "eval" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"generated-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, start=1):
            r.update(id=f"q{i:04d}", generated_by=s.judge_llm.model, prompt_version=prompt_ref("testset_gen"),
                     verified=False)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} items to {out}\nReview them and set \"verified\": true for correct ones.")


if __name__ == "__main__":
    main()
