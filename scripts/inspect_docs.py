"""Doc audit: print style usage, heading outline and tables for each .docx in a folder.
Usage: python scripts/inspect_docs.py <folder>"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def main(folder: str) -> None:
    for path in sorted(Path(folder).glob("*.docx")):
        doc = Document(str(path))
        print("=" * 100)
        print(path.name)
        props = doc.core_properties
        print(f"core: title={props.title!r} subject={props.subject!r} keywords={props.keywords!r}")
        styles, n_tables, outline = Counter(), 0, []
        for child in doc.element.body.iterchildren():
            if child.tag == qn("w:tbl"):
                n_tables += 1
                t = Table(child, doc)
                first = [" ".join(c.text.split())[:25] for c in t.rows[0].cells] if t.rows else []
                outline.append(f"    [TABLE {len(t.rows)}x{len(t.columns)}] {first}")
            elif child.tag == qn("w:p"):
                p = Paragraph(child, doc)
                text = " ".join(p.text.split())
                if not text:
                    continue
                name = p.style.name if p.style is not None else "?"
                styles[name] += 1
                bold = all(r.bold for r in p.runs if r.text.strip()) and any(r.text.strip() for r in p.runs)
                if "heading" in name.lower() or name.lower() == "title" or (bold and len(text) < 90):
                    outline.append(f"  {name:<18} {'(bold) ' if bold else ''}{text[:90]}")
        print(f"tables={n_tables}  paragraph styles={dict(styles.most_common(12))}")
        print("\n".join(outline[:80]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/docs")
