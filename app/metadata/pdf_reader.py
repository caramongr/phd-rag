"""Read-only PDF metadata and first-page layout reader (RAG-006).

Unlike the RAG-004 structural parser, this reader *is* allowed to look at font
sizes, because deciding whether a first-page line is a title is inherently a
typographic question. It reads the embedded document-info dictionary and the
per-line geometry of page one, plus the raw text needed for DOI/year scanning.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from app.metadata.layout import PageLine, PageLayout
from app.parsing.exceptions import DocumentParseError, EncryptedDocumentError


class EmbeddedPdfMetadata(BaseModel):
    """The PDF document-info dictionary fields, unvalidated."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    mod_date: str | None = None


class PdfReadResult(BaseModel):
    """Everything the metadata extractor needs from a PDF."""

    model_config = ConfigDict(extra="forbid")

    embedded: EmbeddedPdfMetadata = Field(default_factory=EmbeddedPdfMetadata)
    first_page: PageLayout | None = None
    full_text: str = ""
    page_count: int = 0


def _page_lines(page: pymupdf.Page, page_number: int) -> PageLayout:
    """Build a :class:`PageLayout` for one page from its text dictionary."""
    height = float(page.rect.height) or 1.0
    lines: list[PageLine] = []
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return PageLayout(page_number=page_number, lines=[], width=float(page.rect.width), height=height)

    index = 0
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text")]
            if not spans:
                continue
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            size = max(float(span.get("size", 0.0)) for span in spans)
            bold = any(bool(int(span.get("flags", 0)) & 16) for span in spans)
            bbox = line.get("bbox") or block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            top_fraction = max(0.0, min(1.0, float(bbox[1]) / height))
            bottom_fraction = max(0.0, min(1.0, float(bbox[3]) / height))
            lines.append(
                PageLine(
                    text=text,
                    size=round(size, 2),
                    top_fraction=round(top_fraction, 4),
                    bottom_fraction=round(bottom_fraction, 4),
                    index=index,
                    bold=bold,
                )
            )
            index += 1

    return PageLayout(
        page_number=page_number,
        lines=lines,
        width=float(page.rect.width),
        height=height,
    )


def read_pdf(path: str | Path) -> PdfReadResult:
    """Read embedded metadata, first-page layout, and text from ``path``.

    Raises:
        EncryptedDocumentError: if the PDF is password-protected.
        DocumentParseError: if the file is missing, corrupt, or unreadable.
    """
    path = Path(path)
    if not path.is_file():
        raise DocumentParseError(f"Document not found: {path}")

    try:
        document = pymupdf.open(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Failed to open PDF '{path}': {exc}") from exc

    try:
        if document.needs_pass:
            raise EncryptedDocumentError(f"PDF is password-protected: {path}")

        raw_meta = document.metadata or {}
        embedded = EmbeddedPdfMetadata(
            title=_clean(raw_meta.get("title")),
            author=_clean(raw_meta.get("author")),
            subject=_clean(raw_meta.get("subject")),
            keywords=_clean(raw_meta.get("keywords")),
            creator=_clean(raw_meta.get("creator")),
            producer=_clean(raw_meta.get("producer")),
            creation_date=_clean(raw_meta.get("creationDate")),
            mod_date=_clean(raw_meta.get("modDate")),
        )

        page_texts: list[str] = []
        first_page: PageLayout | None = None
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            if page_index == 0:
                first_page = _page_lines(page, 1)
            page_texts.append(page.get_text("text"))

        full_text = "\n\n".join(page_texts)

        return PdfReadResult(
            embedded=embedded,
            first_page=first_page,
            full_text=full_text,
            page_count=document.page_count,
        )
    finally:
        document.close()


def _clean(value: object) -> str | None:
    """Return a stripped string, or ``None`` for empty/missing values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
