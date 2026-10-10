"""Markdown parser.

Reads UTF-8 (BOM tolerated), preserves heading hierarchy, paragraphs, list
items, block quotes, and simple pipe-style tables. Markdown is **not** rendered
to HTML and inline syntax (citations, emphasis) is left intact.

``raw_text`` is the decoded source text verbatim (BOM removed, line endings
normalised to LF) so later stages can re-chunk without reparsing the original
file.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.parsing.exceptions import DocumentParseError
from app.parsing.models import (
    ContentItem,
    ContentItemType,
    HeadingSource,
    ParsedBlock,
    ParsedBlockType,
    ParsedDocument,
    ParsedTable,
    ParsedTableCell,
    ParsedTableRow,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")


def _is_table_delimiter(line: str) -> bool:
    """Return ``True`` for a Markdown table delimiter row like ``|---|:--:|``."""
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    return all(character in "|-: " for character in stripped)


def _split_table_row(line: str) -> list[str]:
    """Split a pipe-style table row into trimmed cell strings."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _update_headings(headings: dict[int, str], level: int, text: str) -> None:
    """Set ``level`` to ``text`` and drop any deeper levels."""
    for existing in [key for key in headings if key >= level]:
        del headings[existing]
    headings[level] = text


def parse_markdown(path: Path, file_hash: str | None = None) -> ParsedDocument:
    """Parse a ``.md`` file into a :class:`ParsedDocument`.

    Raises:
        DocumentParseError: if the file is missing, unreadable, or not UTF-8.
    """
    path = Path(path)
    if not path.is_file():
        raise DocumentParseError(f"Document not found: {path}")

    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(f"Markdown file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise DocumentParseError(f"Could not read Markdown file '{path}': {exc}") from exc

    # Universal-newline handling: normalise CRLF/CR to LF so ``raw_text`` is
    # deterministic regardless of how the file was saved. This is the only
    # normalisation applied; all other content is preserved verbatim.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.splitlines()
    blocks: list[ParsedBlock] = []
    tables: list[ParsedTable] = []
    content_items: list[ContentItem] = []
    headings: dict[int, str] = {}

    title: str | None = None
    block_counter = 0
    table_counter = 0
    paragraph_counter = 0
    sequence_counter = 0

    def heading_path() -> list[str]:
        return [text_value for _level, text_value in sorted(headings.items())]

    def add_block(
        block_type: ParsedBlockType,
        block_text: str,
        *,
        heading_level: int | None = None,
        paragraph_index: int | None = None,
        line_number: int,
    ) -> None:
        nonlocal block_counter, sequence_counter
        block = ParsedBlock(
            sequence_index=sequence_counter,
            block_index=block_counter,
            block_type=block_type,
            text=block_text,
            heading_level=heading_level,
            heading_source=(
                HeadingSource.EXPLICIT
                if block_type == ParsedBlockType.HEADING
                else None
            ),
            heading_path=heading_path(),
            paragraph_index=paragraph_index,
            source_locator=f"line:{line_number}",
        )
        blocks.append(block)
        content_items.append(
            ContentItem(
                sequence_index=sequence_counter,
                item_type=ContentItemType.BLOCK,
                block=block,
            )
        )
        block_counter += 1
        sequence_counter += 1

    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        line_number = index + 1
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level == 1 and title is None and heading_text:
                title = heading_text
            _update_headings(headings, level, heading_text)
            add_block(
                ParsedBlockType.HEADING,
                heading_text,
                heading_level=level,
                line_number=line_number,
            )
            index += 1
            continue

        if "|" in stripped and index + 1 < total and _is_table_delimiter(lines[index + 1]):
            table_lines = [line]
            cursor = index + 2
            while cursor < total and lines[cursor].strip() and "|" in lines[cursor]:
                table_lines.append(lines[cursor])
                cursor += 1
            rows = [
                ParsedTableRow(
                    cells=[ParsedTableCell(text=cell) for cell in _split_table_row(row_line)]
                )
                for row_line in table_lines
            ]
            parsed_table = ParsedTable(
                sequence_index=sequence_counter,
                table_index=table_counter,
                rows=rows,
                heading_path=heading_path(),
                source_locator=f"line:{line_number}",
            )
            tables.append(parsed_table)
            content_items.append(
                ContentItem(
                    sequence_index=sequence_counter,
                    item_type=ContentItemType.TABLE,
                    table=parsed_table,
                )
            )
            table_counter += 1
            sequence_counter += 1
            index = cursor
            continue

        if _QUOTE_RE.match(stripped):
            quote_lines: list[str] = []
            while index < total and lines[index].strip().startswith(">"):
                quote_match = _QUOTE_RE.match(lines[index].strip())
                quote_lines.append(quote_match.group(1).strip() if quote_match else "")
                index += 1
            add_block(
                ParsedBlockType.QUOTE,
                "\n".join(quote_lines).strip(),
                line_number=line_number,
            )
            continue

        if _LIST_RE.match(line):
            add_block(ParsedBlockType.LIST_ITEM, stripped, line_number=line_number)
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < total:
            next_line = lines[index]
            next_stripped = next_line.strip()
            if not next_stripped:
                break
            looks_like_table = (
                "|" in next_stripped
                and index + 1 < total
                and _is_table_delimiter(lines[index + 1])
            )
            if (
                _HEADING_RE.match(next_stripped)
                or _QUOTE_RE.match(next_stripped)
                or _LIST_RE.match(next_line)
                or looks_like_table
            ):
                break
            paragraph_lines.append(next_stripped)
            index += 1

        add_block(
            ParsedBlockType.PARAGRAPH,
            " ".join(paragraph_lines).strip(),
            paragraph_index=paragraph_counter,
            line_number=line_number,
        )
        paragraph_counter += 1

    return ParsedDocument(
        source_path=str(path),
        filename=path.name,
        file_hash=file_hash,
        extension=".md",
        title=title,
        blocks=blocks,
        tables=tables,
        content_items=content_items,
        raw_text=text,
    )
