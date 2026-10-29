from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = set("""a an and are as at be by can do does for from has have how i if in is it its me my of on or
our should so that the their then there these this to was we what when where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        corpus = [tokenize(t) or ["_"] for t in texts]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not self.bm25 or not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:k] if s > 0]
