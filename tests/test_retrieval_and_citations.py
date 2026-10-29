from app.rag.citations import extract_ids, is_refusal, strip_citations, validate_citations
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.fusion import rrf_fuse


def test_rrf_prefers_items_ranked_by_both():
    fused = rrf_fuse({"bm25": [("a", 9.0), ("b", 5.0)], "vector": [("b", 0.9), ("c", 0.8)]}, k=60)
    assert [f["chunk_id"] for f in fused] == ["b", "a", "c"]
    assert fused[0]["bm25_rank"] == 2 and fused[0]["vector_rank"] == 1


def test_bm25_exact_ids_and_fields():
    assert tokenize("Error RF-409 in refund_limit_ccy") == ["error", "rf", "409", "refund_limit_ccy"]
    idx = BM25Index(["x", "y", "z"], ["refund limits per currency", "error RF-409 currency mismatch",
                                      "nightly reconciliation job"])
    assert idx.search("what is RF-409", 2)[0][0] == "y"


def test_citation_validation_keeps_valid_and_drops_invalid():
    ans = "Go to Settings [C1]. Set the EUR limit [C2][C9]. Invented fact [C7].\n1. Click Save [C1]."
    res = validate_citations(ans, {"C1", "C2"})
    assert res.status == "ANSWERED"
    assert res.used_ids == ["C1", "C2"]
    assert "Invented fact" not in res.answer and "[C9]" not in res.answer
    assert res.removed_sentences == 1 and "C7" in res.invalid_ids


def test_citation_only_and_invented_identifier_answers_are_rejected():
    # phi4-mini failure modes seen in the first smoke test
    assert validate_citations("[C1][C2][C3]", {"C1", "C2", "C3"}).reason == "empty_answer"
    ctx = "[C1] Fire Job\n| Job Id | CCFOP_EXT |"
    res = validate_citations("CCCOP_EXT", {"C1"}, context_text=ctx)
    assert res.status == "NOT_IN_DOCS" and res.unknown_identifiers == ["CCCOP_EXT"]
    ok = validate_citations("The manual fire job is CCFOP_EXT [C1].", {"C1"}, context_text=ctx)
    assert ok.status == "ANSWERED" and ok.used_ids == ["C1"]
    mixed = validate_citations("The job is CCFOP_EXT [C1]. It also uses CCX_FAKE [C1].", {"C1"}, context_text=ctx)
    assert mixed.status == "ANSWERED" and "CCX_FAKE" not in mixed.answer and mixed.removed_sentences == 1


def test_citation_validation_refusals():
    assert validate_citations("NOT_IN_DOCS: no info about pricing", {"C1"}).status == "NOT_IN_DOCS"
    assert validate_citations("Some answer without citations.", {"C1"}).status == "NOT_IN_DOCS"
    assert validate_citations("Some answer without citations.", {"C1"}, require_citations=False).status == "ANSWERED"
    assert is_refusal("not in docs: missing")
    assert extract_ids("a [C1, C3] b [c2]") == ["C1", "C3", "C2"]
    assert strip_citations("Go to Settings [C1]. Save [C2][C3].") == "Go to Settings. Save."
