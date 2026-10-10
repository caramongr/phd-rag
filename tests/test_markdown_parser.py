"""Tests for the Markdown parser."""

from __future__ import annotations

from pathlib import Path

from app.parsing.markdown_parser import parse_markdown
from app.parsing.models import ParsedBlockType


def write_md(tmp_path: Path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- 16-20: paragraphs, headings, hierarchy, order, Unicode -----------------


def test_parses_simple_paragraph(tmp_path: Path) -> None:
    parsed = parse_markdown(write_md(tmp_path, "This is a paragraph."))
    assert parsed.extension == ".md"
    assert parsed.blocks[0].block_type == ParsedBlockType.PARAGRAPH
    assert parsed.blocks[0].text == "This is a paragraph."


def test_parses_headings_h1_h2_h3(tmp_path: Path) -> None:
    parsed = parse_markdown(write_md(tmp_path, "# H1\n\n## H2\n\n### H3"))
    assert [block.block_type for block in parsed.blocks] == [
        ParsedBlockType.HEADING,
        ParsedBlockType.HEADING,
        ParsedBlockType.HEADING,
    ]
    assert [block.heading_level for block in parsed.blocks] == [1, 2, 3]


def test_maintains_heading_hierarchy(tmp_path: Path) -> None:
    parsed = parse_markdown(
        write_md(tmp_path, "# Chapter 1\n\n## Section A\n\nParagraph body.")
    )
    body = parsed.blocks[-1]
    assert body.text == "Paragraph body."
    assert body.heading_path == ["Chapter 1", "Section A"]


def test_parses_multiple_paragraphs_in_order(tmp_path: Path) -> None:
    parsed = parse_markdown(
        write_md(tmp_path, "First para.\n\nSecond para.\n\nThird para.")
    )
    assert [block.text for block in parsed.blocks] == [
        "First para.",
        "Second para.",
        "Third para.",
    ]


def test_preserves_unicode_and_greek(tmp_path: Path) -> None:
    text = "Ανάλυση δεδομένων — café ✓ 日本語"
    parsed = parse_markdown(write_md(tmp_path, text))
    assert parsed.blocks[0].text == text
    assert text in parsed.raw_text


# --- 21-23: BOM, lists, quotes ----------------------------------------------


def test_handles_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.md"
    path.write_bytes("\ufeff# Heading\n\nBody text.".encode("utf-8"))
    parsed = parse_markdown(path)
    assert "\ufeff" not in parsed.raw_text
    assert parsed.title == "Heading"
    assert parsed.blocks[0].block_type == ParsedBlockType.HEADING


def test_preserves_list_items(tmp_path: Path) -> None:
    parsed = parse_markdown(write_md(tmp_path, "- first\n- second\n\n1. one\n2. two"))
    assert [block.text for block in parsed.blocks] == [
        "- first",
        "- second",
        "1. one",
        "2. two",
    ]
    assert all(block.block_type == ParsedBlockType.LIST_ITEM for block in parsed.blocks)


def test_preserves_block_quote_content(tmp_path: Path) -> None:
    parsed = parse_markdown(
        write_md(tmp_path, "> line one\n> line two\n\nAfter quote.")
    )
    quote = parsed.blocks[0]
    assert quote.block_type == ParsedBlockType.QUOTE
    assert quote.text == "line one\nline two"
    assert parsed.blocks[1].block_type == ParsedBlockType.PARAGRAPH
    assert parsed.blocks[1].text == "After quote."


# --- 24: tables -------------------------------------------------------------


def test_markdown_table_is_structured_and_preserved(tmp_path: Path) -> None:
    text = "| A | B |\n|---|---|\n| 1 | 2 |"
    parsed = parse_markdown(write_md(tmp_path, text))

    assert len(parsed.tables) == 1
    assert [[cell.text for cell in row.cells] for row in parsed.tables[0].rows] == [
        ["A", "B"],
        ["1", "2"],
    ]
    # Text is not silently discarded.
    assert "A" in parsed.raw_text and "1" in parsed.raw_text
    assert not any("|" in block.text for block in parsed.blocks)


# --- extras -----------------------------------------------------------------


def test_derives_title_from_h1(tmp_path: Path) -> None:
    parsed = parse_markdown(write_md(tmp_path, "# My Document Title\n\nBody."))
    assert parsed.title == "My Document Title"


def test_raw_text_is_verbatim_source(tmp_path: Path) -> None:
    text = "# Title\n\nSome *emphasised* text [1].\n"
    parsed = parse_markdown(write_md(tmp_path, text))
    assert parsed.raw_text == text


def test_file_hash_is_passed_through(tmp_path: Path) -> None:
    parsed = parse_markdown(write_md(tmp_path, "Body."), file_hash="cafe")
    assert parsed.file_hash == "cafe"
