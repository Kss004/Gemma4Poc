from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import opendataloader_pdf

from .config import DATA_DIR, DOCS_DIR, RAG_PAGE_READ_LIMIT


# ---------- OpenDataLoader parsing ----------

def _iter_nodes(data: Any):
    """Yield every dict node in the OpenDataLoader JSON, handling nested 'kids' and flat lists."""
    if isinstance(data, list):
        for item in data:
            yield from _iter_nodes(item)
    elif isinstance(data, dict):
        yield data
        for key in ("kids", "children"):
            if key in data:
                yield from _iter_nodes(data[key])


def _is_heading(node: Dict) -> bool:
    if str(node.get("type", "")).lower() == "heading":
        return True
    # OpenDataLoader uses "heading level" (numeric) on heading nodes.
    return "heading level" in node


def _heading_level(node: Dict) -> int:
    lvl = node.get("heading level")
    try:
        return int(lvl)
    except (TypeError, ValueError):
        return 1


def _page_number(node: Dict) -> int:
    for key in ("page number", "page_number", "page"):
        if key in node:
            try:
                return int(node[key])
            except (TypeError, ValueError):
                continue
    return 1


def _node_text(node: Dict) -> str:
    for key in ("content", "text", "description"):
        val = node.get(key)
        if val:
            return str(val).strip()
    return ""


def _flatten_in_order(data: Any) -> List[Dict]:
    """Walk the JSON and return nodes in document order with the fields we care about."""
    ordered: List[Dict] = []
    seen_ids = set()

    def walk(n: Any):
        if isinstance(n, list):
            for item in n:
                walk(item)
        elif isinstance(n, dict):
            nid = id(n)
            if nid in seen_ids:
                return
            seen_ids.add(nid)
            text = _node_text(n)
            if text or _is_heading(n):
                ordered.append(n)
            for key in ("kids", "children"):
                if key in n:
                    walk(n[key])

    walk(data)
    return ordered


# ---------- Document model ----------

@dataclass
class Section:
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    text: str
    summary: str


@dataclass
class Document:
    doc_id: str
    filename: str
    pages: Dict[int, str]
    sections: List[Section]
    page_count: int
    created_at: float

    def get_full_text(self) -> str:
        parts = []
        for p in sorted(self.pages.keys()):
            if self.pages[p]:
                parts.append(f"=== Page {p} ===\n{self.pages[p]}")
        return "\n\n".join(parts)

    def get_section(self, section_id: str) -> Optional[Section]:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    def read_pages(self, start: int, end: int) -> Tuple[str, List[int]]:
        start = max(1, int(start))
        end = min(self.page_count, int(end))
        end = min(end, start + RAG_PAGE_READ_LIMIT - 1)
        if start > end:
            return "", []
        parts = []
        hit = []
        for p in range(start, end + 1):
            if p in self.pages and self.pages[p]:
                parts.append(f"=== Page {p} ===\n{self.pages[p]}")
                hit.append(p)
        return "\n\n".join(parts), hit


# ---------- Tree building ----------

def _build_sections(nodes: List[Dict], total_pages: int) -> Tuple[Dict[int, str], List[Section]]:
    """Given ordered OpenDataLoader nodes, produce per-page text and a flat list of sections with page ranges."""
    pages: Dict[int, List[str]] = {}
    headings: List[Tuple[int, Dict]] = []  # (order_idx, node)

    for idx, node in enumerate(nodes):
        page = _page_number(node)
        text = _node_text(node)
        if _is_heading(node) and text:
            headings.append((idx, node))
        if text and not _is_heading(node):
            pages.setdefault(page, []).append(text)
        elif text and _is_heading(node):
            # Keep the heading line in the page text too for downstream reads.
            pages.setdefault(page, []).append(f"# {text}")

    merged_pages: Dict[int, str] = {p: "\n".join(parts) for p, parts in pages.items()}

    if not headings:
        return merged_pages, []

    sections: List[Section] = []
    for i, (node_idx, node) in enumerate(headings):
        title = _node_text(node)
        level = _heading_level(node)
        page_start = _page_number(node)
        # End at the page before the next heading, else last known page.
        if i + 1 < len(headings):
            next_page = _page_number(headings[i + 1][1])
            page_end = max(page_start, next_page - 1) if next_page > page_start else page_start
        else:
            page_end = max(page_start, max(merged_pages.keys()) if merged_pages else page_start)

        text_parts = []
        for p in range(page_start, page_end + 1):
            if p in merged_pages:
                text_parts.append(merged_pages[p])
        text = "\n".join(text_parts).strip()
        summary = _crude_summary(text)
        sections.append(
            Section(
                id=f"S{i+1}",
                title=title,
                level=level,
                page_start=page_start,
                page_end=page_end,
                text=text,
                summary=summary,
            )
        )
    return merged_pages, sections


def _crude_summary(text: str, max_chars: int = 240) -> str:
    """Cheap, deterministic per-section 'summary': first sentence-ish of the section."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return ""
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    # Try to cut on sentence boundary.
    m = re.search(r"[.!?]\s", cut[::-1])
    if m:
        boundary = len(cut) - m.start()
        return cut[:boundary].strip() + " …"
    return cut.rstrip() + " …"


# ---------- Store ----------

class DocumentStore:
    def __init__(self) -> None:
        os.makedirs(DOCS_DIR, exist_ok=True)
        self._index_path = os.path.join(DATA_DIR, "documents.json")
        self.documents: Dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._index_path):
            return
        with open(self._index_path, "r", encoding="utf-8") as h:
            raw = json.load(h)
        for item in raw.get("documents", []):
            doc_dir = os.path.join(DOCS_DIR, item["doc_id"])
            tree_path = os.path.join(doc_dir, "tree.json")
            if not os.path.exists(tree_path):
                continue
            with open(tree_path, "r", encoding="utf-8") as h:
                tree = json.load(h)
            pages = {int(k): v for k, v in tree["pages"].items()}
            sections = [Section(**s) for s in tree["sections"]]
            self.documents[item["doc_id"]] = Document(
                doc_id=item["doc_id"],
                filename=item["filename"],
                pages=pages,
                sections=sections,
                page_count=item["page_count"],
                created_at=item["created_at"],
            )

    def _save_index(self) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "filename": d.filename,
                    "page_count": d.page_count,
                    "created_at": d.created_at,
                }
                for d in self.documents.values()
            ]
        }
        with open(self._index_path, "w", encoding="utf-8") as h:
            json.dump(payload, h, ensure_ascii=True, indent=2)

    def ingest_pdf(self, file_path: str, filename: str, created_at: float) -> Document:
        doc_id = str(uuid.uuid4())
        doc_dir = os.path.join(DOCS_DIR, doc_id)
        os.makedirs(doc_dir, exist_ok=True)

        opendataloader_pdf.convert(
            input_path=[file_path],
            output_dir=doc_dir,
            format="markdown,json",
        )

        json_path = self._first_file(doc_dir, ".json")
        with open(json_path, "r", encoding="utf-8") as h:
            data = json.load(h)

        ordered_nodes = _flatten_in_order(data)
        pages, sections = _build_sections(ordered_nodes, total_pages=0)
        page_count = max(pages.keys()) if pages else 0

        tree_path = os.path.join(doc_dir, "tree.json")
        with open(tree_path, "w", encoding="utf-8") as h:
            json.dump(
                {
                    "pages": {str(k): v for k, v in pages.items()},
                    "sections": [s.__dict__ for s in sections],
                },
                h,
                ensure_ascii=False,
                indent=2,
            )

        doc = Document(
            doc_id=doc_id,
            filename=filename,
            pages=pages,
            sections=sections,
            page_count=page_count,
            created_at=created_at,
        )
        self.documents[doc_id] = doc
        self._save_index()
        return doc

    def get(self, doc_id: str) -> Document:
        return self.documents[doc_id]

    @staticmethod
    def _first_file(folder: str, suffix: str) -> str:
        for name in sorted(os.listdir(folder)):
            if name.endswith(suffix):
                return os.path.join(folder, name)
        raise FileNotFoundError(f"No {suffix} file produced in {folder}")


# ---------- Tool executor (PageIndex navigation) ----------

def rag_system_prompt(doc: Document, user_system_prompt: str) -> str:
    return (
        f"{user_system_prompt.strip()}\n\n"
        "You are an assistant analyzing a document. The entire document text is provided below.\n"
        "Read it carefully and use it to answer the user's questions.\n"
        "- If the answer is not in the document, say so plainly.\n"
        "- When quoting or referring to parts of the document, mention the page number if available (e.g., [p.7]).\n\n"
        f"--- Document: {doc.filename} ({doc.page_count} pages) ---\n\n"
        f"{doc.get_full_text()}"
    )
