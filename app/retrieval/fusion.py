from __future__ import annotations


def rrf_fuse(ranked_lists: dict[str, list[tuple[str, float]]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). Ranks start at 1."""
    fused: dict[str, dict] = {}
    for name, results in ranked_lists.items():
        for rank, (cid, score) in enumerate(results, start=1):
            entry = fused.setdefault(cid, {"chunk_id": cid, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (k + rank)
            entry[f"{name}_rank"] = rank
            entry[f"{name}_score"] = round(score, 4)
    return sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)
