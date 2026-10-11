"""Shared fixtures for the RAG-007 chunking tests.

The helpers build :class:`~app.parsing.models.ParsedDocument` objects directly, so
chunking behaviour is tested against exact structural input (headings, page
numbers, text sources, tables) rather than against whatever a real file happens
to produce.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class BlockSpec:
    """One prose block to place in a synthetic parsed document."""

    text: str
    block_type: ParsedBlockType = ParsedBlockType.PARAGRAPH
    page: int | None = 1
    heading_path: tuple[str, ...] = ()
    heading_level: int | None = None
    heading_source: HeadingSource | None = None
    text_source: TextSource | None = TextSource.NATIVE_PDF
    locator: str | None = None


@dataclass
class TableSpec:
    """One table to place in a synthetic parsed document."""

    rows: list[list[str]]
    page: int | None = 1
    heading_path: tuple[str, ...] = ()
    locator: str | None = None


@dataclass
class PageSpec:
    """A page whose effective text is used for running-header detection."""

    number: int
    text: str
    text_source: TextSource = TextSource.NATIVE_PDF
    ocr_applied: bool = False
    native_text: str | None = None
    ocr_text: str | None = None


def heading(
    text: str,
    level: int = 1,
    *,
    page: int | None = 1,
    path: tuple[str, ...] | None = None,
) -> BlockSpec:
    """A heading block whose own heading path already includes it."""
    return BlockSpec(
        text=text,
        block_type=ParsedBlockType.HEADING,
        page=page,
        heading_path=path if path is not None else (text,),
        heading_level=level,
        heading_source=HeadingSource.EXPLICIT,
    )


def paragraph(
    text: str,
    *,
    page: int | None = 1,
    path: tuple[str, ...] = (),
    source: TextSource | None = TextSource.NATIVE_PDF,
    text_source: TextSource | None = None,
) -> BlockSpec:
    """A prose paragraph block under ``path``."""
    return BlockSpec(
        text=text,
        page=page,
        heading_path=path,
        text_source=source if text_source is None else text_source,
    )


def table(rows: list[list[str]], *, page: int | None = 1, path: tuple[str, ...] = ()) -> TableSpec:
    """A structured table under ``path``."""
    return TableSpec(rows=rows, page=page, heading_path=path)


def build_document(
    items: list[BlockSpec | TableSpec],
    *,
    pages: list[PageSpec] | None = None,
    filename: str = "paper.pdf",
    extension: str = ".pdf",
    source_path: str | None = None,
    title: str | None = None,
    file_hash: str = "a" * 64,
) -> ParsedDocument:
    """Assemble a synthetic parsed document, preserving the given item order."""
    blocks: list[ParsedBlock] = []
    tables: list[ParsedTable] = []
    content_items: list[ContentItem] = []
    block_index = 0
    table_index = 0

    for sequence, spec in enumerate(items):
        if isinstance(spec, TableSpec):
            parsed_table = ParsedTable(
                sequence_index=sequence,
                table_index=table_index,
                page_number=spec.page,
                rows=[
                    ParsedTableRow(cells=[ParsedTableCell(text=cell) for cell in row])
                    for row in spec.rows
                ],
                heading_path=list(spec.heading_path),
                source_locator=spec.locator or f"page:{spec.page}:table:{table_index}",
            )
            tables.append(parsed_table)
            content_items.append(
                ContentItem(
                    sequence_index=sequence,
                    item_type=ContentItemType.TABLE,
                    table=parsed_table,
                )
            )
            table_index += 1
            continue

        block = ParsedBlock(
            sequence_index=sequence,
            block_index=block_index,
            block_type=spec.block_type,
            text=spec.text,
            heading_level=spec.heading_level,
            heading_source=spec.heading_source,
            heading_path=list(spec.heading_path),
            page_number=spec.page,
            text_source=spec.text_source,
            source_locator=spec.locator or f"page:{spec.page}:block:{block_index}",
        )
        blocks.append(block)
        content_items.append(
            ContentItem(sequence_index=sequence, item_type=ContentItemType.BLOCK, block=block)
        )
        block_index += 1

    parsed_pages = [
        ParsedPage(
            page_number=page.number,
            raw_text=page.text,
            character_count=len(page.text),
            text_extraction_status=TextExtractionStatus.TEXT,
            native_text=page.native_text if page.native_text is not None else page.text,
            ocr_text=page.ocr_text,
            text_source=page.text_source,
            ocr_applied=page.ocr_applied,
        )
        for page in (pages or [])
    ]

    return ParsedDocument(
        source_path=source_path or filename,
        filename=filename,
        file_hash=file_hash,
        extension=extension,
        title=title,
        blocks=blocks,
        tables=tables,
        content_items=content_items,
        pages=parsed_pages,
        page_count=len(parsed_pages) or None,
        raw_text="\n\n".join(block.text for block in blocks),
    )


def long_text(sentences: int, *, prefix: str = "This paragraph examines") -> str:
    """A paragraph of roughly ``sentences`` sentences, for size tests."""
    parts = [
        f"{prefix} aspect {index} of immersive virtual reality in higher education "
        f"and reports what the study found about learning outcomes and engagement."
        for index in range(sentences)
    ]
    return " ".join(parts)
