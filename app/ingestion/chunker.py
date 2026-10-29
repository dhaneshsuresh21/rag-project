"""Structure-aware chunking: split by heading sections, then 500-800 token parts with overlap for long sections."""
from __future__ import annotations

import re

from app.config import ChunkingCfg
from app.ingestion.docx_parser import Block, ParsedDoc, table_to_markdown
from app.utils import estimate_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def build_sections(blocks: list[Block]) -> list[tuple[list[str], list[Block]]]:
    """Group blocks under their heading path, e.g. ["Configuration", "Refund limits"]."""
    sections: list[tuple[list[str], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    path: list[str] = []
    current: list[Block] = []
    for b in blocks:
        if b.kind == "heading":
            if current:
                sections.append((path, current))
            current = []
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            path = [t for _, t in stack]
        else:
            current.append(b)
    if current:
        sections.append((path, current))
    return sections


def _block_units(b: Block, cfg: ChunkingCfg) -> list[tuple[str, bool]]:
    """Render a block into text units (text, is_table). Oversized blocks are pre-split."""
    if b.kind == "table":
        if estimate_tokens(b.text) <= cfg.max_tokens or not b.rows or len(b.rows) < 3:
            return [(b.text, True)]
        header, body, units, group = b.rows[0], b.rows[1:], [], []
        for row in body:  # split by row groups, repeating the header
            candidate = table_to_markdown([header] + group + [row])
            if group and estimate_tokens(candidate) > cfg.target_tokens:
                units.append((table_to_markdown([header] + group), True))
                group = []
            group.append(row)
        if group:
            units.append((table_to_markdown([header] + group), True))
        return units

    text = ("  " * b.level + "- " + b.text) if b.kind == "list" else b.text
    if estimate_tokens(text) <= cfg.max_tokens:
        return [(text, False)]
    units, buf = [], ""
    for sent in _SENTENCE_SPLIT.split(text):
        if buf and estimate_tokens(buf + " " + sent) > cfg.target_tokens:
            units.append((buf, False))
            buf = ""
        buf = f"{buf} {sent}".strip()
    if buf:
        units.append((buf, False))
    return units


def _overlap_tail(units: list[tuple[str, bool]], overlap_tokens: int) -> list[tuple[str, bool]]:
    tail, total = [], 0
    for text, is_table in reversed(units):
        if is_table:
            break
        t = estimate_tokens(text)
        if total + t > overlap_tokens:
            if not tail:  # take the end of a long paragraph, cut at a word boundary
                snippet = text[-overlap_tokens * 4:]
                snippet = snippet.split(" ", 1)[-1] if " " in snippet else snippet
                tail.insert(0, ("..." + snippet, False))
            break
        tail.insert(0, (text, False))
        total += t
    return tail


def pack_units(units: list[tuple[str, bool]], cfg: ChunkingCfg) -> list[str]:
    total = sum(estimate_tokens(t) for t, _ in units)
    if total <= cfg.max_tokens:
        return ["\n".join(t for t, _ in units)]
    parts: list[list[tuple[str, bool]]] = []
    cur: list[tuple[str, bool]] = []
    cur_new = 0  # tokens added beyond the overlap carry-over
    for unit in units:
        t = estimate_tokens(unit[0])
        cur_tok = sum(estimate_tokens(x) for x, _ in cur)
        if cur_new and (cur_tok + t > cfg.max_tokens or cur_tok >= cfg.target_tokens):
            parts.append(cur)
            cur = _overlap_tail(cur, cfg.overlap_tokens) if not unit[1] else []
            cur_new = 0
        cur.append(unit)
        cur_new += t
    if cur and cur_new:
        parts.append(cur)
    return ["\n".join(t for t, _ in p) for p in parts]


def chunk_document(doc: ParsedDoc, cfg: ChunkingCfg) -> list[dict]:
    markers = {m.lower().rstrip(".") for m in cfg.empty_markers}
    sections = []
    for path, blocks in build_sections(doc.blocks):
        units = [u for b in blocks for u in _block_units(b, cfg)]
        body = " ".join(t for t, _ in units).strip().lower().rstrip(".")
        if body and body not in markers:  # drop empty template sections ("NA", "Not applicable", ...)
            sections.append((path or ["Overview"], units))

    chunks: list[dict] = []
    pending: list[tuple[list[str], list[tuple[str, bool]]]] = []  # tiny sibling sections awaiting merge

    def emit(group: list[tuple[list[str], list[tuple[str, bool]]]]) -> None:
        if len(group) == 1:
            section_path, units = " > ".join(group[0][0]), group[0][1]
        else:  # merged siblings: "Parent > A / B / C", each part labelled with its heading
            parent = " > ".join(group[0][0][:-1])
            names = " / ".join(p[-1] for p, _ in group)
            section_path = f"{parent} > {names}" if parent else names
            units = [u for p, us in group for u in [(f"{p[-1]}:", False)] + us]
        _append_chunks(doc, section_path, pack_units(units, cfg), chunks)

    for path, units in sections:
        if pending and pending[0][0][:-1] != path[:-1]:  # only merge siblings under the same parent
            if len(pending) == 1 and pending[0][0] == path[:-1]:  # tiny parent intro -> into its first child
                units = [(f"{pending[0][0][-1]}:", False)] + pending[0][1] + units
            else:
                emit(pending)
            pending = []
        tokens = sum(estimate_tokens(t) for t, _ in units)
        if tokens < cfg.min_tokens:
            pending.append((path, units))
            if sum(estimate_tokens(t) for _, us in pending for t, _ in us) >= cfg.min_tokens:
                emit(pending)
                pending = []
            continue
        emit(pending + [(path, units)])
        pending = []
    if pending:
        emit(pending)
    return chunks


def _append_chunks(doc: ParsedDoc, section_path: str, texts: list[str], chunks: list[dict]) -> None:
    for part, text in enumerate(texts, start=1):
        ord_ = len(chunks)
        chunks.append({
            "chunk_id": f"{doc.doc_id}::{ord_:04d}",
            "doc_id": doc.doc_id,
            "ord": ord_,
            "section_path": section_path,
            "text": text,
            "tokens": estimate_tokens(text),
            "part": part,
            "parts": len(texts),
            "metadata": {
                "doc_title": doc.title,
                "feature_id": doc.feature_id,
                "client": doc.client,
                "doc_type": doc.doc_type,
                "version": doc.version,
                "source_file": doc.source_file,
            },
        })


def context_header(chunk: dict) -> str:
    md = chunk["metadata"]
    parts = [md.get("doc_title", ""), md.get("feature_id", ""), chunk["section_path"]]
    return " | ".join(p for p in parts if p)


def embed_text(chunk: dict) -> str:
    """Text used for embeddings and BM25: contextual header + chunk text."""
    return f"{context_header(chunk)}\n{chunk['text']}"
