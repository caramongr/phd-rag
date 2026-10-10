"""Tests for the PDF parser (synthetic PDFs only, never the real corpus)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.parsing.exceptions import DocumentParseError, EncryptedDocumentError
from app.parsing.models import HeadingSource, ParsedBlockType, TextExtractionStatus
from app.parsing.parser import parse_document
from app.parsing.pdf_parser import parse_pdf

_UNICODE_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def unicode_font() -> Path:
    for candidate in _UNICODE_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    pytest.skip("no Unicode-capable font available on this machine")


def new_pdf() -> pymupdf.Document:
    return pymupdf.open()


def add_text_page(
    doc: pymupdf.Document,
    text: str,
    *,
    font_path: Path | None = None,
    y: int = 72,
) -> None:
    page = doc.new_page()
    if font_path is None:
        page.insert_text((72, y), text, fontsize=12)
    else:
        font = pymupdf.Font(fontfile=str(font_path))
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, y), text, font=font, fontsize=12)
        writer.write_text(page)


def save(doc: pymupdf.Document, path: Path) -> Path:
    doc.save(str(path))
    doc.close()
    return path


def make_pdf(tmp_path: Path, name: str = "doc.pdf", texts: tuple[str, ...] = ("Body.",)) -> Path:
    doc = new_pdf()
    for text in texts:
        add_text_page(doc, text)
    return save(doc, tmp_path / name)


# --- 1-9: basic parsing, pages, provenance ----------------------------------


def test_parses_simple_one_page_pdf(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, "simple.pdf", ("Hello PDF world.",))
    parsed = parse_pdf(path)

    assert parsed.extension == ".pdf"
    assert parsed.filename == "simple.pdf"
    assert parsed.page_count == 1
    assert parsed.blocks
    assert "Hello PDF world." in parsed.raw_text


def test_parses_multipage_pdf(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "multi.pdf", ("Page one text.", "Page two text.")))
    assert parsed.page_count == 2
    assert len(parsed.pages) == 2


def test_page_numbers_are_one_based(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "pages.pdf", ("First page.", "Second page.")))
    assert [page.page_number for page in parsed.pages] == [1, 2]


def test_blocks_retain_correct_page_number(tmp_path: Path) -> None:
    parsed = parse_pdf(
        make_pdf(tmp_path, "pagenum.pdf", ("Alpha on page one.", "Beta on page two."))
    )
    by_text = {block.text: block.page_number for block in parsed.blocks}
    assert by_text["Alpha on page one."] == 1
    assert by_text["Beta on page two."] == 2


def test_block_indexes_are_deterministic(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, "idx.pdf", ("One.", "Two."))
    first = parse_pdf(path)
    second = parse_pdf(path)
    assert [b.block_index for b in first.blocks] == [b.block_index for b in second.blocks]
    assert [b.block_index for b in first.blocks] == list(range(len(first.blocks)))


def test_source_locator_is_deterministic(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "loc.pdf", ("Locate me.",)))
    block = parsed.blocks[0]
    assert block.source_locator == f"page:1:block:{block.block_index}"


def test_preserves_unicode(tmp_path: Path) -> None:
    doc = new_pdf()
    add_text_page(doc, "Café — naïve résumé", font_path=unicode_font())
    parsed = parse_pdf(save(doc, tmp_path / "unicode.pdf"))
    # PyMuPDF may encode spaces as NBSP in synthetic PDFs, so assert on content.
    assert "Café" in parsed.raw_text
    assert "naïve" in parsed.raw_text
    assert "résumé" in parsed.raw_text


def test_preserves_greek_text(tmp_path: Path) -> None:
    doc = new_pdf()
    add_text_page(doc, "Ανάλυση δεδομένων", font_path=unicode_font())
    parsed = parse_pdf(save(doc, tmp_path / "greek.pdf"))
    assert "Ανάλυση" in parsed.raw_text
    assert "δεδομένων" in parsed.raw_text


def test_page_count_is_correct_for_three_pages(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "three.pdf", ("a", "b", "c")))
    assert parsed.page_count == 3
    assert len(parsed.pages) == 3


# --- 10-11: title -----------------------------------------------------------


def test_uses_pdf_metadata_title(tmp_path: Path) -> None:
    doc = new_pdf()
    doc.set_metadata({"title": "A Study of Algorithms"})
    add_text_page(doc, "Body text.")
    parsed = parse_pdf(save(doc, tmp_path / "title.pdf"))
    assert parsed.title == "A Study of Algorithms"


def test_missing_metadata_title_is_none(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "notitle.pdf", ("Body.",)))
    assert parsed.title is None


# --- 12-13: empty pages and raw-text boundaries -----------------------------


def test_empty_page_is_handled(tmp_path: Path) -> None:
    doc = new_pdf()
    add_text_page(doc, "Has text.")
    doc.new_page()  # completely empty page
    parsed = parse_pdf(save(doc, tmp_path / "empty.pdf"))

    assert parsed.page_count == 2
    assert parsed.pages[1].blocks == []
    assert parsed.pages[1].text_extraction_status == TextExtractionStatus.EMPTY


def test_raw_text_contains_page_boundaries(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "boundaries.pdf", ("Page one.", "Page two.")))
    assert "[PAGE 1]" in parsed.raw_text
    assert "[PAGE 2]" in parsed.raw_text


# --- 14-16: errors ----------------------------------------------------------


def test_corrupt_pdf_gives_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is definitely not a PDF file at all")
    with pytest.raises(DocumentParseError):
        parse_pdf(path)


def test_missing_pdf_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentParseError):
        parse_pdf(tmp_path / "missing.pdf")


def test_encrypted_pdf_raises_clear_error(tmp_path: Path) -> None:
    doc = new_pdf()
    add_text_page(doc, "Secret content.")
    path = tmp_path / "secret.pdf"
    doc.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()

    with pytest.raises(EncryptedDocumentError):
        parse_pdf(path)
    assert issubclass(EncryptedDocumentError, DocumentParseError)


def test_encrypted_pdf_error_preserves_path(tmp_path: Path) -> None:
    doc = new_pdf()
    add_text_page(doc, "Secret.")
    path = tmp_path / "named.pdf"
    doc.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="o",
        user_pw="u",
    )
    doc.close()
    with pytest.raises(DocumentParseError, match="named.pdf"):
        parse_pdf(path)


# --- 17: dispatcher ---------------------------------------------------------


def test_dispatcher_is_case_insensitive_for_pdf(tmp_path: Path) -> None:
    doc = new_pdf()
    add_text_page(doc, "Body.")
    path = save(doc, tmp_path / "UPPER.PDF")
    parsed = parse_document(path)
    assert parsed.extension == ".pdf"


# --- 15 (scanned) + conservative headings + ordering ------------------------


def test_possibly_scanned_page_is_detected(tmp_path: Path) -> None:
    doc = new_pdf()
    page = doc.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60))
    page.insert_image(pymupdf.Rect(72, 72, 260, 260), pixmap=pixmap)
    parsed = parse_pdf(save(doc, tmp_path / "scanned.pdf"))

    assert (
        parsed.pages[0].text_extraction_status == TextExtractionStatus.POSSIBLY_SCANNED
    )


def test_named_heading_is_inferred(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "refs.pdf", ("References",)))
    heading = parsed.blocks[0]
    assert heading.block_type == ParsedBlockType.HEADING
    assert heading.heading_level == 1
    assert heading.heading_source == HeadingSource.INFERRED


def test_numbered_heading_is_inferred(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "numbered.pdf", ("2.1 Background",)))
    heading = parsed.blocks[0]
    assert heading.block_type == ParsedBlockType.HEADING
    assert heading.heading_level == 2


def test_plain_sentence_is_not_a_heading(tmp_path: Path) -> None:
    parsed = parse_pdf(
        make_pdf(tmp_path, "prose.pdf", ("This is an ordinary sentence of prose.",))
    )
    assert all(
        block.block_type == ParsedBlockType.PARAGRAPH for block in parsed.blocks
    )


def test_pages_without_tables_produce_no_tables(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "notables.pdf", ("No tables here.",)))
    assert parsed.tables == []


def test_content_items_share_sequence_with_blocks(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "seq.pdf", ("One.", "Two.")))
    assert [item.sequence_index for item in parsed.content_items] == list(
        range(len(parsed.blocks))
    )


def test_page_blocks_are_the_same_objects_as_document_blocks(tmp_path: Path) -> None:
    parsed = parse_pdf(make_pdf(tmp_path, "same.pdf", ("Shared block.",)))
    assert parsed.pages[0].blocks[0] is parsed.blocks[0]
