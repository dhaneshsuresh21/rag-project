"""Parse a Word solution doc into ordered structural blocks (headings, paragraphs, list items, tables)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.config import ParsingCfg
from app.utils import sha256_file, slugify

_HEADING_STYLE = re.compile(r"heading\s*(\d)", re.I)
_NUMBERED_TITLE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Za-z].*)$")


@dataclass
class Block:
    kind: str  # heading | paragraph | list | table
    text: str = ""
    level: int = 0  # heading level (1..n) or list indent level
    rows: list[list[str]] | None = None


@dataclass
class ParsedDoc:
    doc_id: str
    source_file: str
    title: str
    feature_id: str
    client: str
    doc_type: str
    content_hash: str
    version: str = ""
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _iter_body(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _style_name(p: Paragraph) -> str:
    try:
        return (p.style.name or "") if p.style is not None else ""
    except (KeyError, ValueError):
        return ""


def _styled_heading_level(p: Paragraph) -> int | None:
    name = _style_name(p)
    if name.lower() == "title":
        return 0
    if m := _HEADING_STYLE.match(name):
        return int(m.group(1))
    ppr = p._p.pPr
    if ppr is not None and (ol := ppr.find(qn("w:outlineLvl"))) is not None:
        lvl = int(ol.get(qn("w:val")))
        return lvl + 1 if lvl < 9 else None
    return None


def _list_level(p: Paragraph) -> int | None:
    ppr = p._p.pPr
    if ppr is not None and ppr.numPr is not None:
        ilvl = ppr.numPr.ilvl
        return int(ilvl.val) if ilvl is not None and ilvl.val is not None else 0
    return 0 if "list" in _style_name(p).lower() else None


def _is_all_bold(p: Paragraph) -> bool:
    runs = [r for r in p.runs if r.text.strip()]
    return bool(runs) and all(r.bold or (r.style is not None and r.style.font.bold) for r in runs)


def _table_rows(t: Table) -> list[list[str]]:
    rows = []
    for row in t.rows:
        cells, prev = [], None
        for cell in row.cells:
            if cell._tc is prev:  # horizontally merged cell repeats the same element
                continue
            prev = cell._tc
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(cells)
    return rows


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[c.replace("|", "\\|") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def title_from_filename(stem: str) -> str:
    """'PRAC-234109_Solution_SPG _SSA Reconciliation_v1' -> 'PRAC-234109 Solution SPG SSA Reconciliation'."""
    t = re.sub(r"[_]+", " ", stem)
    t = re.sub(r"\s*\bv(?:ersion)?\s*\d+(?:\.\d+)*(?:\s+\d+)?\s*$", "", t, flags=re.I)
    return " ".join(t.split())


def parse_docx(path: Path, cfg: ParsingCfg) -> ParsedDoc:
    path = Path(path)
    doc = Document(str(path))
    feature_re = re.compile(cfg.feature_id_pattern, re.I)
    client_re = re.compile(cfg.client_label_pattern, re.I)
    skip_prefixes = tuple(p.lower() for p in cfg.skip_style_prefixes)
    generic = {t.lower() for t in cfg.generic_titles}
    ignore_headings = {h.lower() for h in cfg.ignore_headings}
    front_matter = {s.lower() for s in cfg.front_matter_styles}

    raw: list[tuple[str, object, str]] = []  # (kind, element, style name)
    styled_headings = 0
    for el in _iter_body(doc):
        if isinstance(el, Paragraph):
            style = _style_name(el)
            if el.text.strip() and not style.lower().startswith(skip_prefixes):
                raw.append(("p", el, style))
                if (_styled_heading_level(el) or 0) >= 1:
                    styled_headings += 1
        else:
            raw.append(("t", el, ""))

    warnings: list[str] = []
    use_fallback = styled_headings == 0
    if use_fallback:
        warnings.append("No Word heading styles found; using numbered/bold-line heading fallback.")

    blocks: list[Block] = []
    title_candidates: dict[str, str] = {}
    version = ""
    client = ""
    current_level = 0
    for kind, el, style in raw:
        if kind == "t":
            rows = _table_rows(el)
            if rows:
                blocks.append(Block("table", table_to_markdown(rows), rows=rows))
                if not client:
                    for r in rows:
                        if len(r) >= 2 and re.fullmatch(r"\s*(client|customer)\s*:?\s*", r[0], re.I) and r[1]:
                            client = r[1].strip()
            continue

        text = " ".join(el.text.split())
        if style.lower() in front_matter:  # title page: title/subtitle/version/date are metadata, not content
            if style in cfg.title_styles and text.lower() not in generic:
                title_candidates.setdefault(style, re.sub(cfg.title_prefix_pattern, "", text, flags=re.I))
            if not version and (m := re.search(r"\bversion\s*([\d.]+)", text, re.I)):
                version = m.group(1)
            continue

        level = _styled_heading_level(el)
        if level == 0:
            if text.lower() not in generic:
                title_candidates.setdefault("Title", text)
            continue
        is_short_label = len(text) <= cfg.pseudo_heading_max_chars and not text.endswith((".", ":", ";", ","))
        if level is None and use_fallback and is_short_label:
            if m := _NUMBERED_TITLE.match(text):
                level = m.group(1).count(".") + 1
            elif _is_all_bold(el) and len(text) <= 80:
                level = 2
        elif level is None and cfg.bold_subheadings and current_level and is_short_label \
                and len(text) <= 90 and _is_all_bold(el):
            level = current_level + 1  # bold sub-heading inside a styled section
            blocks.append(Block("heading", text, level=level))
            continue
        if level:
            if text.lower() in ignore_headings:
                continue
            current_level = level
            blocks.append(Block("heading", text, level=level))
            continue
        if not client and (m := client_re.match(text)):
            client = m.group(1).strip()
        lvl = _list_level(el)
        blocks.append(Block("list", text, level=lvl) if lvl is not None else Block("paragraph", text))

    props = doc.core_properties
    title = next((title_candidates[s] for s in cfg.title_styles if title_candidates.get(s)), "") \
        or (props.title or "").strip() or title_from_filename(path.stem)

    head_text = " ".join([path.stem, title, props.subject or "", props.keywords or ""] +
                         [b.text for b in blocks[:30]])
    m = feature_re.search(head_text)
    feature_id = re.sub(r"[\s_]+", "-", m.group(0).upper()) if m else ""
    prefix = feature_id.split("-")[0] if feature_id else ""
    doc_type = {"GAP": "gap", "FEAT": "feature", "FEATURE": "feature", "PRJ": "project",
                "PROJ": "project", "PRAC": "solution"}.get(prefix, "solution")

    if not blocks:
        warnings.append("Document has no extractable text.")
    if not feature_id:
        warnings.append("No feature/gap ID found in filename or title.")

    return ParsedDoc(
        doc_id=slugify(path.stem),
        source_file=str(path),
        title=title,
        feature_id=feature_id,
        client=client or "general",
        doc_type=doc_type,
        content_hash=sha256_file(path),
        version=version,
        blocks=blocks,
        warnings=warnings,
    )
