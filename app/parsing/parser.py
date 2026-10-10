"""Format dispatcher for the parsing layer.

Maps a file extension to the correct parser and raises a typed error for
unsupported formats. It never returns empty content for an unsupported type.
"""

from __future__ import annotations

from pathlib import Path

from app.parsing.docx_parser import parse_docx
from app.parsing.exceptions import UnsupportedDocumentFormat
from app.parsing.markdown_parser import parse_markdown
from app.parsing.models import ParsedDocument
from app.parsing.pdf_parser import parse_pdf


def parse_document(
    path: Path,
    file_hash: str | None = None,
    *,
    ocr: bool | None = None,
) -> ParsedDocument:
    """Parse ``path`` with the parser registered for its extension.

    Extension matching is case-insensitive. ``.pdf`` -> PDF parser,
    ``.docx`` -> DOCX parser, ``.md`` -> Markdown parser.

    Args:
        ocr: for PDFs, overrides the configured OCR setting (``False`` disables
            the OCR fallback). Ignored for DOCX/Markdown.

    Raises:
        UnsupportedDocumentFormat: if no parser handles the extension.
        DocumentParseError: if the chosen parser fails.
    """
    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return parse_pdf(path, file_hash, ocr_enabled=ocr)
    if extension == ".docx":
        return parse_docx(path, file_hash)
    if extension == ".md":
        return parse_markdown(path, file_hash)

    raise UnsupportedDocumentFormat(
        f"Unsupported document format '{extension or path.name}' for: {path}"
    )
