"""Tests for the parse dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document as DocxDocument

from app.parsing.exceptions import DocumentParseError, UnsupportedDocumentFormat
from app.parsing.models import ParsedBlockType
from app.parsing.parser import parse_document


def make_docx(tmp_path: Path, name: str = "doc.docx") -> Path:
    path = tmp_path / name
    doc = DocxDocument()
    doc.add_paragraph("DOCX body.")
    doc.save(str(path))
    return path


def make_md(tmp_path: Path, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text("# Title\n\nMarkdown body.", encoding="utf-8")
    return path


def test_docx_dispatches_correctly(tmp_path: Path) -> None:
    parsed = parse_document(make_docx(tmp_path))
    assert parsed.extension == ".docx"
    assert parsed.blocks[-1].text == "DOCX body."


def test_md_dispatches_correctly(tmp_path: Path) -> None:
    parsed = parse_document(make_md(tmp_path))
    assert parsed.extension == ".md"
    assert parsed.blocks[0].block_type == ParsedBlockType.HEADING


def test_extension_matching_is_case_insensitive(tmp_path: Path) -> None:
    docx_parsed = parse_document(make_docx(tmp_path, "UPPER.DOCX"))
    assert docx_parsed.extension == ".docx"

    md_parsed = parse_document(make_md(tmp_path, "NOTES.MD"))
    assert md_parsed.extension == ".md"


@pytest.mark.parametrize("name", ["file.txt", "file.xlsx", "file.docx.bak", "noext"])
def test_unsupported_extension_raises(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"irrelevant")
    with pytest.raises(UnsupportedDocumentFormat):
        parse_document(path)


def test_unsupported_format_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "file.xlsx"
    path.write_bytes(b"irrelevant")
    with pytest.raises(DocumentParseError):
        parse_document(path)


def test_file_hash_is_passed_through(tmp_path: Path) -> None:
    parsed = parse_document(make_md(tmp_path), file_hash="abc123")
    assert parsed.file_hash == "abc123"
