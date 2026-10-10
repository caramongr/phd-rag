"""Selective, offline OCR fallback for PDF pages (RAG-005).

Uses Tesseract via the lightweight ``pytesseract`` wrapper. The engine is
imported lazily so the rest of the system works (and the whole test suite runs)
without Tesseract installed.

Privacy: rendering and OCR happen entirely locally — no image, page, or text is
ever sent to a cloud service or external API.
"""

from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf

from app.parsing.models import OcrPageResult, TextSource

#: Places Tesseract is commonly installed on Windows.
COMMON_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


class OcrError(Exception):
    """Raised when OCR of a page fails at runtime."""


class OcrUnavailableError(OcrError):
    """Raised when OCR cannot run because Tesseract (or its wrapper) is missing."""


@dataclass(frozen=True)
class OcrDiagnostics:
    """A snapshot of local OCR availability."""

    available: bool
    executable: str | None
    version: str | None
    languages: tuple[str, ...]
    message: str


def _find_tesseract_executable() -> str | None:
    """Locate the Tesseract binary via env var, PATH, then common locations."""
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and Path(env_path).is_file():
        return env_path
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in COMMON_WINDOWS_PATHS:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_ocr_diagnostics() -> OcrDiagnostics:
    """Return (and cache) the local OCR availability report."""
    executable = _find_tesseract_executable()

    try:
        import pytesseract
    except ImportError:
        return OcrDiagnostics(
            available=False,
            executable=executable,
            version=None,
            languages=(),
            message=(
                "pytesseract is not installed. Install with: "
                "pip install pytesseract Pillow"
            ),
        )

    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable

    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception:
        return OcrDiagnostics(
            available=False,
            executable=executable,
            version=None,
            languages=(),
            message=(
                "OCR unavailable: Tesseract executable not found. Install "
                "Tesseract OCR (https://github.com/UB-Mannheim/tesseract/wiki), "
                "ensure it is on PATH, or set the TESSERACT_CMD environment variable."
            ),
        )

    try:
        languages = tuple(sorted(pytesseract.get_languages(config="")))
    except Exception:
        languages = ()

    return OcrDiagnostics(
        available=True,
        executable=executable,
        version=version,
        languages=languages,
        message="Tesseract available",
    )


def reset_ocr_cache() -> None:
    """Clear the cached diagnostics (useful after changing ``TESSERACT_CMD``)."""
    get_ocr_diagnostics.cache_clear()


def tesseract_available() -> bool:
    """Return ``True`` when OCR can run locally."""
    return get_ocr_diagnostics().available


def render_page_png(page: object, *, dpi: int) -> bytes:
    """Render one PDF page to PNG bytes in memory at the given DPI."""
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)  # type: ignore[attr-defined]
    try:
        return pixmap.tobytes("png")
    finally:
        del pixmap


def ocr_image_bytes(image_bytes: bytes, *, language: str) -> tuple[str, float | None]:
    """OCR a PNG image in memory, returning text and aggregate confidence."""
    diagnostics = get_ocr_diagnostics()
    if not diagnostics.available:
        raise OcrUnavailableError(diagnostics.message)

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded above in practice
        raise OcrUnavailableError(
            "pytesseract/Pillow not installed. Install with: "
            "pip install pytesseract Pillow"
        ) from exc

    if diagnostics.executable:
        pytesseract.pytesseract.tesseract_cmd = diagnostics.executable

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise OcrError(f"Could not decode rendered page image: {exc}") from exc

    try:
        text = pytesseract.image_to_string(image, lang=language)
    except Exception as exc:
        raise OcrError(f"OCR failed: {exc}") from exc

    return text, None


def ocr_pdf_page(page: object, *, language: str, dpi: int) -> OcrPageResult:
    """Render and OCR a single PDF page (fallback only).

    Raises:
        OcrUnavailableError: if OCR cannot run locally.
        OcrError: if rendering or OCR fails for this page.
    """
    diagnostics = get_ocr_diagnostics()
    if not diagnostics.available:
        raise OcrUnavailableError(diagnostics.message)

    image_bytes = render_page_png(page, dpi=dpi)
    text, confidence = ocr_image_bytes(image_bytes, language=language)
    page_number = int(getattr(page, "number", 0)) + 1
    return OcrPageResult(
        page_number=page_number,
        text=text,
        language=language,
        confidence=confidence,
        source=TextSource.OCR,
    )
