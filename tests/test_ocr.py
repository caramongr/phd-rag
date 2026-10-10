"""Tests for the selective OCR fallback (RAG-005).

The OCR engine is mocked so the suite never depends on Tesseract being installed.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.config import Settings
from app.parsing import ocr
from app.parsing.models import (
    OcrPageResult,
    ParsedBlockType,
    TextExtractionStatus,
    TextSource,
)
from app.parsing.pdf_parser import parse_pdf

LONG_NATIVE_TEXT = (
    "This native paragraph is comfortably longer than fifty characters so that "
    "no OCR fallback is ever triggered for it."
)


def new_pdf() -> pymupdf.Document:
    return pymupdf.open()


def add_text_page(doc: pymupdf.Document, text: str) -> None:
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)


def add_image_page(doc: pymupdf.Document, text: str | None = None) -> None:
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text, fontsize=11)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60))
    page.insert_image(pymupdf.Rect(72, 130, 260, 320), pixmap=pixmap)


def save(doc: pymupdf.Document, path: Path) -> Path:
    doc.save(str(path))
    doc.close()
    return path


class FakeOcr:
    """Records calls and returns a canned OCR result or raises a canned error."""

    def __init__(self, text: str = "RECOVERED OCR TEXT", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[tuple[int, str, int]] = []

    def __call__(self, page, *, language: str, dpi: int) -> OcrPageResult:
        self.calls.append((int(getattr(page, "number", 0)) + 1, language, dpi))
        if self.error is not None:
            raise self.error
        return OcrPageResult(
            page_number=int(getattr(page, "number", 0)) + 1,
            text=self.text,
            language=language,
            source=TextSource.OCR,
        )


def install_fake(monkeypatch, fake: FakeOcr) -> FakeOcr:
    monkeypatch.setattr(ocr, "ocr_pdf_page", fake)
    return fake


# --- 1-3: eligibility -------------------------------------------------------


def test_native_text_page_does_not_trigger_ocr(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_text_page(doc, LONG_NATIVE_TEXT)
    path = save(doc, tmp_path / "native.pdf")

    fake = install_fake(monkeypatch, FakeOcr())
    parsed = parse_pdf(path)

    assert fake.calls == []
    page = parsed.pages[0]
    assert page.ocr_applied is False
    assert page.text_source == TextSource.NATIVE_PDF
    assert parsed.blocks[0].text_source == TextSource.NATIVE_PDF


def test_image_only_page_triggers_ocr(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "image_only.pdf")

    fake = install_fake(monkeypatch, FakeOcr("Recovered scanned text"))
    parsed = parse_pdf(path)

    assert len(fake.calls) == 1
    page = parsed.pages[0]
    assert page.ocr_applied is True
    assert page.text_source == TextSource.OCR
    assert page.ocr_text == "Recovered scanned text"
    assert "Recovered scanned text" in parsed.raw_text


def test_very_low_text_image_page_triggers_ocr(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc, text="tiny")
    path = save(doc, tmp_path / "low_text.pdf")

    fake = install_fake(monkeypatch, FakeOcr("Recovered"))
    parsed = parse_pdf(path)

    assert len(fake.calls) == 1
    assert parsed.pages[0].text_extraction_status == TextExtractionStatus.POSSIBLY_SCANNED


def test_threshold_is_configurable(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc, text="tiny")  # 4 characters
    path = save(doc, tmp_path / "threshold.pdf")

    fake = install_fake(monkeypatch, FakeOcr("x"))
    # With a threshold below the native length, the page is ordinary TEXT again.
    parsed = parse_pdf(path, settings=Settings(ocr_min_native_chars=3))

    assert fake.calls == []
    assert parsed.pages[0].ocr_applied is False


# --- 4-7: provenance --------------------------------------------------------


def test_ocr_block_is_marked_text_source_ocr(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "src.pdf")

    install_fake(monkeypatch, FakeOcr("OCR body"))
    parsed = parse_pdf(path)

    ocr_blocks = [b for b in parsed.blocks if b.text_source == TextSource.OCR]
    assert len(ocr_blocks) == 1
    assert ocr_blocks[0].block_type == ParsedBlockType.PARAGRAPH
    assert ocr_blocks[0].text == "OCR body"


def test_native_and_ocr_text_are_both_preserved(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc, text="weak native")
    path = save(doc, tmp_path / "both.pdf")

    install_fake(monkeypatch, FakeOcr("Strong OCR text"))
    parsed = parse_pdf(path)

    page = parsed.pages[0]
    assert page.ocr_text == "Strong OCR text"
    assert page.native_text is not None and "weak native" in page.native_text
    native_blocks = [b for b in parsed.blocks if b.text_source == TextSource.NATIVE_PDF]
    ocr_blocks = [b for b in parsed.blocks if b.text_source == TextSource.OCR]
    assert any("weak native" in b.text for b in native_blocks)
    assert len(ocr_blocks) == 1


def test_ocr_text_is_stored_separately_from_native_text(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc, text="native portion")
    path = save(doc, tmp_path / "separate.pdf")

    install_fake(monkeypatch, FakeOcr("ocr portion"))
    parsed = parse_pdf(path)

    page = parsed.pages[0]
    assert page.native_text != page.ocr_text
    assert page.raw_text == "ocr portion"  # effective text is the OCR result


# --- 8-9: page number and locator -------------------------------------------


def test_ocr_block_has_correct_one_based_page_number(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_text_page(doc, LONG_NATIVE_TEXT)
    add_image_page(doc)
    path = save(doc, tmp_path / "page_two.pdf")

    fake = install_fake(monkeypatch, FakeOcr("page two OCR"))
    parsed = parse_pdf(path)

    assert fake.calls[0][0] == 2
    ocr_blocks = [b for b in parsed.blocks if b.text_source == TextSource.OCR]
    assert len(ocr_blocks) == 1
    assert ocr_blocks[0].page_number == 2


def test_ocr_block_has_deterministic_source_locator(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "locator.pdf")

    install_fake(monkeypatch, FakeOcr("located"))
    first = parse_pdf(path)
    install_fake(monkeypatch, FakeOcr("located"))
    second = parse_pdf(path)

    locators = [
        b.source_locator for b in first.blocks if b.text_source == TextSource.OCR
    ]
    assert locators == ["page:1:ocr:1"]
    assert locators == [
        b.source_locator for b in second.blocks if b.text_source == TextSource.OCR
    ]


# --- 10-13: failure, disabling, unavailability, configuration ---------------


def test_ocr_failure_on_one_page_does_not_fail_pdf(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    add_text_page(doc, LONG_NATIVE_TEXT)
    add_image_page(doc)
    path = save(doc, tmp_path / "failure.pdf")

    class SelectiveFake(FakeOcr):
        def __call__(self, page, *, language: str, dpi: int):
            if int(getattr(page, "number", 0)) == 0:
                self.calls.append((1, language, dpi))
                raise ocr.OcrError("simulated failure")
            self.calls.append((int(getattr(page, "number", 0)) + 1, language, dpi))
            return OcrPageResult(
                page_number=int(getattr(page, "number", 0)) + 1,
                text="second page OCR",
                language=language,
            )

    install_fake(monkeypatch, SelectiveFake())
    parsed = parse_pdf(path)

    assert parsed.page_count == 3
    assert parsed.pages[0].ocr_applied is False
    assert parsed.pages[0].ocr_error is not None
    assert "simulated failure" in parsed.pages[0].ocr_error
    assert parsed.pages[1].ocr_error is None
    assert parsed.pages[2].ocr_applied is True


def test_ocr_disabled_prevents_ocr(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "disabled.pdf")

    fake = install_fake(monkeypatch, FakeOcr("should not run"))
    parsed = parse_pdf(path, ocr_enabled=False)

    assert fake.calls == []
    assert parsed.pages[0].ocr_applied is False
    assert parsed.pages[0].text_source == TextSource.NATIVE_PDF


def test_ocr_config_can_disable_ocr(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "cfg_disabled.pdf")

    fake = install_fake(monkeypatch, FakeOcr("should not run"))
    parsed = parse_pdf(path, settings=Settings(ocr_enabled=False))

    assert fake.calls == []
    assert parsed.pages[0].ocr_applied is False


def test_unavailable_ocr_records_page_error(tmp_path: Path, monkeypatch) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "unavailable.pdf")

    def unavailable(page, *, language: str, dpi: int):
        raise ocr.OcrUnavailableError(
            "OCR unavailable: Tesseract executable not found."
        )

    monkeypatch.setattr(ocr, "ocr_pdf_page", unavailable)
    parsed = parse_pdf(path)

    assert parsed.page_count == 1
    page = parsed.pages[0]
    assert page.ocr_applied is False
    assert page.ocr_error is not None
    assert "Tesseract" in page.ocr_error


def test_ocr_pdf_page_raises_when_unavailable(monkeypatch) -> None:
    diagnostics = ocr.OcrDiagnostics(
        available=False,
        executable=None,
        version=None,
        languages=(),
        message="OCR unavailable: Tesseract executable not found.",
    )
    monkeypatch.setattr(ocr, "get_ocr_diagnostics", lambda: diagnostics)

    with pytest.raises(ocr.OcrUnavailableError):
        ocr.ocr_pdf_page(object(), language="eng", dpi=300)


def test_ocr_language_and_dpi_configuration_are_honoured(
    tmp_path: Path, monkeypatch
) -> None:
    doc = new_pdf()
    add_image_page(doc)
    path = save(doc, tmp_path / "config.pdf")

    fake = install_fake(monkeypatch, FakeOcr("configured"))
    parse_pdf(path, settings=Settings(ocr_language="ell", ocr_dpi=200))

    assert fake.calls[0][1] == "ell"
    assert fake.calls[0][2] == 200


def test_default_ocr_language_is_english() -> None:
    from app.config import DEFAULT_OCR_LANGUAGE

    assert DEFAULT_OCR_LANGUAGE == "eng"
    assert Settings().ocr_language == "eng"


# --- models -----------------------------------------------------------------


def test_text_source_values() -> None:
    assert TextSource.NATIVE_PDF.value == "native_pdf"
    assert TextSource.OCR.value == "ocr"


def test_ocr_page_result_computes_character_count() -> None:
    result = OcrPageResult(page_number=3, text="hello", language="eng")
    assert result.source == TextSource.OCR
    assert result.confidence is None
    assert result.character_count == 5


# --- CLI metric regression --------------------------------------------------


def test_cli_native_text_pages_excludes_image_only_pages(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    doc = new_pdf()
    add_image_page(doc)  # genuinely image-only: zero native text
    path = save(doc, tmp_path / "image_only_cli.pdf")

    def unavailable(page, *, language: str, dpi: int):
        raise ocr.OcrUnavailableError("OCR unavailable: Tesseract executable not found.")

    monkeypatch.setattr(ocr, "ocr_pdf_page", unavailable)

    from app.parsing import parse as parse_cli

    exit_code = parse_cli.main([str(path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Native text pages: 0" in output
    assert "OCR failures: 1" in output
    assert "Pages with no text: 1" in output


def test_cli_native_text_pages_counts_real_native_text(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    doc = new_pdf()
    add_text_page(doc, LONG_NATIVE_TEXT)
    path = save(doc, tmp_path / "native_cli.pdf")

    from app.parsing import parse as parse_cli

    exit_code = parse_cli.main([str(path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Native text pages: 1" in output
    assert "Pages with no text: 0" in output
