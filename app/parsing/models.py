"""Normalized parsing models.

These models describe what a document *contains and where*, before any chunking.
A :class:`ParsedBlock` is deliberately **not** a retrieval chunk: the chunking
stage decides how to group this content later.

Ordering note: ``ParsedDocument.content_items`` is the canonical, globally
ordered stream where blocks and tables share one ``sequence_index`` space.

Provenance: PDF text carries a :class:`TextSource` (``NATIVE_PDF`` or ``OCR``) so
OCR-fallback content is always distinguishable from native extraction. Pages
keep the original native text even when OCR replaces it as the effective text.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParsedBlockType(StrEnum):
    """The kind of prose unit a block represents."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    QUOTE = "quote"
    OTHER = "other"


class HeadingSource(StrEnum):
    """How a heading block was detected."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"


class ContentItemType(StrEnum):
    """Whether a canonical content item is prose or a table."""

    BLOCK = "block"
    TABLE = "table"


class TextExtractionStatus(StrEnum):
    """How much text could be extracted natively from a PDF page."""

    TEXT = "text"
    EMPTY = "empty"
    POSSIBLY_SCANNED = "possibly_scanned"


class TextSource(StrEnum):
    """Where a PDF text block came from.

    ``NATIVE_PDF`` is text extracted by PyMuPDF; ``OCR`` is text recovered by the
    selective OCR fallback. The two are never mixed without provenance.
    """

    NATIVE_PDF = "native_pdf"
    OCR = "ocr"


class ParsedBlock(BaseModel):
    """One prose block (heading, paragraph, list item, ...) from a document."""

    model_config = ConfigDict(extra="forbid")

    sequence_index: int
    block_index: int
    block_type: ParsedBlockType
    text: str
    heading_level: int | None = None
    heading_source: HeadingSource | None = None
    heading_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    page_number: int | None = None
    text_source: TextSource | None = None
    source_locator: str | None = None


class ParsedTableCell(BaseModel):
    """A single table cell's preserved text."""

    model_config = ConfigDict(extra="forbid")

    text: str


class ParsedTableRow(BaseModel):
    """A table row made of ordered cells."""

    model_config = ConfigDict(extra="forbid")

    cells: list[ParsedTableCell] = Field(default_factory=list)


class ParsedTable(BaseModel):
    """A table preserved separately from prose (never flattened permanently)."""

    model_config = ConfigDict(extra="forbid")

    sequence_index: int
    table_index: int
    page_number: int | None = None
    rows: list[ParsedTableRow] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    source_locator: str | None = None


class ContentItem(BaseModel):
    """One entry in the canonical, globally-ordered content stream."""

    model_config = ConfigDict(extra="forbid")

    sequence_index: int
    item_type: ContentItemType
    block: ParsedBlock | None = None
    table: ParsedTable | None = None

    @model_validator(mode="after")
    def _check_item_matches_type(self) -> "ContentItem":
        if self.item_type == ContentItemType.BLOCK and self.block is None:
            raise ValueError("item_type 'block' requires a block")
        if self.item_type == ContentItemType.TABLE and self.table is None:
            raise ValueError("item_type 'table' requires a table")
        return self


class OcrPageResult(BaseModel):
    """The raw result of OCR-ing a single PDF page.

    ``confidence`` stays ``None`` unless a reliable aggregate confidence is
    available; it is never invented.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int
    text: str
    language: str
    confidence: float | None = None
    source: TextSource = TextSource.OCR
    character_count: int = 0

    @model_validator(mode="after")
    def _set_character_count(self) -> "OcrPageResult":
        self.character_count = len(self.text)
        return self


class ParsedPage(BaseModel):
    """One PDF page with its own blocks, tables, and provenance.

    Native extraction is never destroyed: ``native_text`` holds what PyMuPDF
    produced, ``ocr_text`` holds any OCR fallback, and ``raw_text`` is the
    effective text (OCR when it succeeded, otherwise native).
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int
    blocks: list[ParsedBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    raw_text: str = ""
    character_count: int = 0
    text_extraction_status: TextExtractionStatus = TextExtractionStatus.TEXT

    # OCR provenance (RAG-005)
    native_text: str | None = None
    ocr_text: str | None = None
    text_source: TextSource = TextSource.NATIVE_PDF
    ocr_applied: bool = False
    ocr_language: str | None = None
    ocr_error: str | None = None


class ParsedDocument(BaseModel):
    """The normalized result of parsing one PDF, DOCX, or Markdown file."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    filename: str
    file_hash: str | None = None
    extension: str
    title: str | None = None
    blocks: list[ParsedBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    content_items: list[ContentItem] = Field(default_factory=list)
    pages: list[ParsedPage] = Field(default_factory=list)
    page_count: int | None = None
    raw_text: str = ""
