"""Citation enforcement: keep only claims backed by provided chunk IDs, detect refusals and invented identifiers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE_RE = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]", re.I)
IDENT_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")  # CCFOP_EXT, ENABLE_EDIT_PII_SPECIFIC_INFO, ...
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)")
_WORD = re.compile(r"[A-Za-z0-9]{2,}")
NOT_IN_DOCS = "NOT_IN_DOCS"
MIN_CONTENT_WORDS = 3


@dataclass
class CitationResult:
    status: str  # ANSWERED | NOT_IN_DOCS
    answer: str
    used_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    unknown_identifiers: list[str] = field(default_factory=list)
    removed_sentences: int = 0
    uncited_sentences: int = 0
    reason: str = ""


def extract_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in CITE_RE.finditer(text):
        ids += [x.strip().upper() for x in re.split(r"[,;]", m.group(1))]
    return ids


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"[ \t]*" + CITE_RE.pattern, "", text, flags=re.I)).strip()


def is_refusal(text: str) -> bool:
    return NOT_IN_DOCS in text.strip()[:200].upper().replace(" ", "_")


def validate_citations(answer: str, valid_ids: set[str], require_citations: bool = True,
                       context_text: str | None = None) -> CitationResult:
    """Drops sentences that cite only unknown chunk IDs or contain identifiers not found in the context.
    Returns NOT_IN_DOCS when the model refused, nothing valid is cited, or no real content remains."""
    if is_refusal(answer):
        return CitationResult(status=NOT_IN_DOCS, answer=answer.strip(), reason="model_refused")

    valid_ids = {v.upper() for v in valid_ids}
    used, invalid, unknown, removed, uncited = [], [], [], 0, 0
    out_lines = []
    for line in answer.strip().splitlines():
        kept = []
        for sent in _SENT_SPLIT.split(line):
            if context_text is not None:
                bad_idents = [i for i in IDENT_RE.findall(sent) if i not in context_text]
                if bad_idents:  # invented / mis-copied identifier -> unsupported claim
                    unknown += bad_idents
                    removed += 1
                    continue
            ids = extract_ids(sent)
            if not ids:
                if sent.strip():
                    uncited += 1
                kept.append(sent)
                continue
            good = [i for i in ids if i in valid_ids]
            bad = [i for i in ids if i not in valid_ids]
            invalid += bad
            if not good:
                removed += 1
                continue
            used += good
            for b in set(bad):  # drop invalid markers but keep the sentence
                sent = re.sub(rf"\[\s*{b}\s*\]", "", sent, flags=re.I)
            kept.append(sent)
        if kept or not line.strip():
            out_lines.append(" ".join(kept))

    cleaned = "\n".join(out_lines).strip()
    used_unique = list(dict.fromkeys(used))
    common = dict(answer=cleaned, invalid_ids=invalid, unknown_identifiers=unknown, removed_sentences=removed,
                  uncited_sentences=uncited)
    if len(_WORD.findall(strip_citations(cleaned))) < MIN_CONTENT_WORDS:
        reason = "unknown_identifiers" if unknown else "empty_answer"
        return CitationResult(status=NOT_IN_DOCS, reason=reason, **common)
    if require_citations and not used_unique:
        return CitationResult(status=NOT_IN_DOCS, reason="no_valid_citations", **common)
    return CitationResult(status="ANSWERED", used_ids=used_unique, **common)
