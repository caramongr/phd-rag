"""PDF parser.

Uses PyMuPDF (``pymupdf``) as the primary parser, page by page, read-only.
Native extraction is authoritative; a conservative, page-level OCR fallback
(RAG-005) recovers text only from pages that have no usable native text and look
like scans. OCR output is always tagged with ``TextSource.OCR`` and the native
extraction is never destroyed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

# PyMuPDF prints a one-off advertisement for an optional layout package the
# first time table extraction runs. Silence it so CLI output stays concise.
if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()

from app.config import Settings, get_settings
from app.parsing import ocr
from app.parsing.exceptions import DocumentParseError, EncryptedDocumentError
from app.parsing.models import (
    ContentItem,
    ContentItemType,
    HeadingSource,
    ParsedBlock,
    ParsedBlockType,
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    ParsedTableCell,
    ParsedTableRow,
    TextExtractionStatus,
    TextSource,
)

#: A page with fewer than this many stripped characters that also contains
#: images is treated as ``POSSIBLY_SCANNED``. Sourced from settings (RAG-005).
_DEFAULT_MIN_NATIVE_CHARS = 50

_MAX_HEADING_CHARS = 90
_MAX_HEADING_WORDS = 12
_MAX_NUMBER_COMPONENT_DIGITS = 3

_NUMBERED_HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>\S.*)$")
_NAMED_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "methods",
        "methodology",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "works cited",
        "acknowledgements",
        "acknowledgments",
        "appendix",
    }
)


def _block_text(block: dict) -> str:
    """Rebuild a text block's content from PyMuPDF's structured output."""
    lines: list[str] = []
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        lines.append(line_text.rstrip())
    return "\n".join(lines).strip()


def _infer_heading_level(text: str) -> int | None:
    """Return a conservative heading level, or ``None`` to keep it as prose."""
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return None
    if len(stripped) > _MAX_HEADING_CHARS or len(stripped.split()) > _MAX_HEADING_WORDS:
        return None
    if stripped.endswith((".", "!", "?", ";", ",")):
        return None

    if stripped.rstrip(":").casefold() in _NAMED_HEADINGS:
        return 1

    match = _NUMBERED_HEADING_RE.match(stripped)
    if match is None:
        return None
    title = match.group("title").strip()
    if not title or not (title[0].isupper() or title[0].isdigit()):
        return None
    components = match.group("num").split(".")
    if any(len(component) > _MAX_NUMBER_COMPONENT_DIGITS for component in components):
        return None
    return min(len(components), 6)


def _extract_table_rows(table: object) -> list[ParsedTableRow] | None:
    """Convert a PyMuPDF table to rows, or ``None`` if extraction is unreliable."""
    try:
        data = table.extract()  # type: ignore[attr-defined]
    except Exception:
        return None
    if not data:
        return None
    rows: list[ParsedTableRow] = []
    for raw_row in data:
        cells = [ParsedTableCell(text=(cell or "").strip()) for cell in raw_row]
        rows.append(ParsedTableRow(cells=cells))
    return rows


def _page_has_images(page: object) -> bool:
    """Return ``True`` if the page contains any image XObjects."""
    try:
        return len(page.get_images(full=True)) > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _page_status(
    native_text: str, has_images: bool, min_native_chars: int
) -> TextExtractionStatus:
    """Classify a page from its native extraction only."""
    text_chars = len(native_text.strip())
    if text_chars == 0:
        return (
            TextExtractionStatus.POSSIBLY_SCANNED
            if has_images
            else TextExtractionStatus.EMPTY
        )
    if text_chars < min_native_chars and has_images:
        return TextExtractionStatus.POSSIBLY_SCANNED
    return TextExtractionStatus.TEXT


def _is_ocr_eligible(
    native_text: str,
    status: TextExtractionStatus,
    has_images: bool,
    min_native_chars: int,
) -> bool:
    """Decide whether a page should get the OCR fallback.

    A page qualifies only when it looks like a scan with no usable native text:
    it is ``POSSIBLY_SCANNED``, or it has images and fewer than
    ``min_native_chars`` characters of native text. Ordinary ``TEXT`` pages are
    never OCR-ed.
    """
    if status == TextExtractionStatus.POSSIBLY_SCANNED:
        return True
    return has_images and len(native_text.strip()) < min_native_chars


def parse_pdf(
    path: Path,
    file_hash: str | None = None,
    *,
    ocr_enabled: bool | None = None,
    settings: Settings | None = None,
) -> ParsedDocument:
    """Parse a ``.pdf`` file into a :class:`ParsedDocument` with page provenance.

    Args:
        ocr_enabled: overrides ``settings.ocr_enabled`` when not ``None``.
        settings: configuration override (mainly for tests).

    Raises:
        EncryptedDocumentError: if the PDF is password-protected.
        DocumentParseError: if the file is missing, corrupt, or unreadable.
    """
    settings = settings or get_settings()
    use_ocr = settings.ocr_enabled if ocr_enabled is None else ocr_enabled
    min_native_chars = settings.ocr_min_native_chars

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

        blocks: list[ParsedBlock] = []
        tables: list[ParsedTable] = []
        content_items: list[ContentItem] = []
        pages: list[ParsedPage] = []
        raw_parts: list[str] = []

        headings: dict[int, str] = {}
        block_counter = 0
        table_counter = 0
        sequence_counter = 0

        def heading_path() -> list[str]:
            return [value for _level, value in sorted(headings.items())]

        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document.load_page(page_index)

            try:
                page_dict = page.get_text("dict")
            except Exception as exc:
                raise DocumentParseError(
                    f"Failed to extract text from '{path}' page {page_number}: {exc}"
                ) from exc

            has_images = _page_has_images(page)
            page_blocks: list[ParsedBlock] = []
            native_texts: list[str] = []

            for raw_block in page_dict.get("blocks", []):
                if raw_block.get("type", 0) != 0:  # 0 = text, 1 = image
                    continue
                text = _block_text(raw_block)
                if not text:
                    continue

                level = _infer_heading_level(text)
                if level is not None:
                    _update_headings(headings, level, text)

                block = ParsedBlock(
                    sequence_index=sequence_counter,
                    block_index=block_counter,
                    block_type=(
                        ParsedBlockType.HEADING
                        if level is not None
                        else ParsedBlockType.PARAGRAPH
                    ),
                    text=text,
                    heading_level=level,
                    heading_source=(
                        HeadingSource.INFERRED if level is not None else None
                    ),
                    heading_path=heading_path(),
                    page_number=page_number,
                    text_source=TextSource.NATIVE_PDF,
                    source_locator=f"page:{page_number}:block:{block_counter}",
                )
                page_blocks.append(block)
                blocks.append(block)
                content_items.append(
                    ContentItem(
                        sequence_index=sequence_counter,
                        item_type=ContentItemType.BLOCK,
                        block=block,
                    )
                )
                native_texts.append(text)
                block_counter += 1
                sequence_counter += 1

            page_tables = _extract_page_tables(
                page, page_number, table_counter, sequence_counter, heading_path
            )
            for parsed_table, seq in page_tables:
                tables.append(parsed_table)
                content_items.append(
                    ContentItem(
                        sequence_index=seq,
                        item_type=ContentItemType.TABLE,
                        table=parsed_table,
                    )
                )
            if page_tables:
                table_counter += len(page_tables)
                sequence_counter += len(page_tables)

            native_text = "\n\n".join(native_texts)
            status = _page_status(native_text, has_images, min_native_chars)

            ocr_text: str | None = None
            ocr_applied = False
            ocr_error: str | None = None
            effective_text = native_text
            text_source = TextSource.NATIVE_PDF

            eligible = use_ocr and _is_ocr_eligible(
                native_text, status, has_images, min_native_chars
            )
            if eligible:
                try:
                    result = ocr.ocr_pdf_page(
                        page, language=settings.ocr_language, dpi=settings.ocr_dpi
                    )
                    ocr_applied = True
                    ocr_text = result.text
                    if ocr_text.strip():
                        effective_text = ocr_text
                        text_source = TextSource.OCR
                        ocr_block = ParsedBlock(
                            sequence_index=sequence_counter,
                            block_index=block_counter,
                            block_type=ParsedBlockType.PARAGRAPH,
                            text=ocr_text,
                            heading_path=heading_path(),
                            page_number=page_number,
                            text_source=TextSource.OCR,
                            source_locator=f"page:{page_number}:ocr:1",
                        )
                        page_blocks.append(ocr_block)
                        blocks.append(ocr_block)
                        content_items.append(
                            ContentItem(
                                sequence_index=sequence_counter,
                                item_type=ContentItemType.BLOCK,
                                block=ocr_block,
                            )
                        )
                        block_counter += 1
                        sequence_counter += 1
                except Exception as exc:  # OCR failure must not abort the document
                    ocr_error = f"{type(exc).__name__}: {exc}"

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    blocks=page_blocks,
                    tables=[parsed for parsed, _seq in page_tables],
                    raw_text=effective_text,
                    character_count=len(effective_text),
                    text_extraction_status=status,
                    native_text=native_text or None,
                    ocr_text=ocr_text,
                    text_source=text_source,
                    ocr_applied=ocr_applied,
                    ocr_language=settings.ocr_language if ocr_applied else None,
                    ocr_error=ocr_error,
                )
            )
            raw_parts.append(f"[PAGE {page_number}]\n{effective_text}")

        metadata = document.metadata or {}
        raw_title = (metadata.get("title") or "").strip()
        title = raw_title or None

        return ParsedDocument(
            source_path=str(path),
            filename=path.name,
            file_hash=file_hash,
            extension=".pdf",
            title=title,
            blocks=blocks,
            tables=tables,
            content_items=content_items,
            pages=pages,
            page_count=document.page_count,
            raw_text="\n\n".join(raw_parts),
        )
    finally:
        document.close()


def _extract_page_tables(
    page: object,
    page_number: int,
    table_counter: int,
    sequence_counter: int,
    heading_path,
) -> list[tuple[ParsedTable, int]]:
    """Best-effort table extraction. Table text also remains in the blocks."""
    if not hasattr(page, "find_tables"):
        return []
    try:
        finder = page.find_tables()
        found = list(finder.tables)
    except Exception:
        return []

    results: list[tuple[ParsedTable, int]] = []
    for offset, table in enumerate(found):
        rows = _extract_table_rows(table)
        if rows is None:
            continue
        index = table_counter + offset
        parsed = ParsedTable(
            sequence_index=sequence_counter + offset,
            table_index=index,
            page_number=page_number,
            rows=rows,
            heading_path=heading_path(),
            source_locator=f"page:{page_number}:table:{index}",
        )
        results.append((parsed, sequence_counter + offset))
    return results


def _update_headings(headings: dict[int, str], level: int, text: str) -> None:
    """Set ``level`` to ``text`` and drop any deeper levels."""
    for existing in [key for key in headings if key >= level]:
        del headings[existing]
    headings[level] = text
