"""Development CLI for the parsing layer.

Usage::

    python -m app.parsing.parse "paper.pdf"
    python -m app.parsing.parse "paper.pdf" --no-ocr
    python -m app.parsing.parse --ocr-info
    python -m app.parsing.parse "some.docx"

Prints a concise summary. The full document is never dumped to the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.parsing import ocr
from app.parsing.exceptions import DocumentParseError
from app.parsing.models import (
    HeadingSource,
    ParsedBlockType,
    TextExtractionStatus,
    TextSource,
)
from app.parsing.parser import parse_document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.parsing.parse",
        description="Parse a PDF, DOCX, or Markdown file and print a summary.",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="File to parse (.pdf, .docx, .md). Not needed with --ocr-info.",
    )
    parser.add_argument(
        "--show-blocks",
        type=int,
        default=0,
        metavar="N",
        help="Also print a one-line preview of the first N blocks.",
    )
    parser.add_argument(
        "--show-pages",
        type=int,
        default=0,
        metavar="N",
        help="Also print a per-page summary for the first N pages (PDF only).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable the OCR fallback (native extraction only).",
    )
    parser.add_argument(
        "--ocr-info",
        action="store_true",
        help="Print local OCR (Tesseract) availability and exit.",
    )
    return parser


def _preview(text: str, limit: int = 60) -> str:
    collapsed = text.replace("\n", " ")
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _print_ocr_info() -> int:
    diagnostics = ocr.get_ocr_diagnostics()
    if not diagnostics.available:
        print(diagnostics.message)
        print(f"Executable: {diagnostics.executable or 'not found'}")
        return 1
    print("Tesseract: available")
    print(f"Executable: {diagnostics.executable}")
    print(f"Version: {diagnostics.version}")
    print("Languages:")
    for language in diagnostics.languages:
        print(f"- {language}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a file and print a summary. Returns an exit code."""
    args = _build_parser().parse_args(argv)

    if args.ocr_info:
        return _print_ocr_info()

    if args.path is None:
        print("Error: a path is required (or use --ocr-info).", file=sys.stderr)
        return 2

    try:
        document = parse_document(args.path, ocr=False if args.no_ocr else None)
    except DocumentParseError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    heading_count = sum(
        1 for block in document.blocks if block.block_type == ParsedBlockType.HEADING
    )
    explicit_headings = sum(
        1 for block in document.blocks if block.heading_source == HeadingSource.EXPLICIT
    )
    inferred_headings = sum(
        1 for block in document.blocks if block.heading_source == HeadingSource.INFERRED
    )

    print(f"File: {document.source_path}")
    print(f"Format: {document.extension}")
    print(f"Title: {document.title if document.title is not None else '-'}")
    if document.page_count is not None:
        print(f"Pages: {document.page_count}")
    print(f"Blocks: {len(document.blocks)}")
    print(f"Tables: {len(document.tables)}")
    print(f"Headings: {heading_count}")
    print(f"Explicit headings: {explicit_headings}")
    print(f"Inferred headings: {inferred_headings}")

    if document.pages:
        # A page counts as "native text" only if native extraction actually
        # produced usable text. An image-only page whose OCR failed has
        # text_source == NATIVE_PDF but no text, and must not be reported as a
        # native-text page.
        native_pages = sum(
            1
            for page in document.pages
            if page.text_source == TextSource.NATIVE_PDF and page.raw_text.strip()
        )
        ocr_pages = sum(1 for page in document.pages if page.ocr_applied)
        ocr_failures = sum(1 for page in document.pages if page.ocr_error)
        no_text_pages = sum(1 for page in document.pages if not page.raw_text.strip())
        empty_pages = sum(
            1
            for page in document.pages
            if page.text_extraction_status == TextExtractionStatus.EMPTY
        )
        scanned_pages = sum(
            1
            for page in document.pages
            if page.text_extraction_status == TextExtractionStatus.POSSIBLY_SCANNED
        )
        print(f"Native text pages: {native_pages}")
        print(f"OCR pages: {ocr_pages}")
        print(f"OCR failures: {ocr_failures}")
        print(f"Pages with no text: {no_text_pages}")
        print(f"Empty pages: {empty_pages}")
        print(f"Possibly scanned pages: {scanned_pages}")
        if ocr_failures:
            first_error = next(
                page.ocr_error for page in document.pages if page.ocr_error
            )
            print(f"OCR note: {first_error}")

    print(f"Characters: {len(document.raw_text)}")

    if args.show_pages > 0 and document.pages:
        print()
        print("First pages:")
        for page in document.pages[: args.show_pages]:
            print(
                f"  page {page.page_number}: {page.text_extraction_status.value} "
                f"source={page.text_source.value} ocr={page.ocr_applied} "
                f"blocks={len(page.blocks)} tables={len(page.tables)} "
                f"chars={page.character_count}"
            )

    if args.show_blocks > 0:
        print()
        print("First blocks:")
        for block in document.blocks[: args.show_blocks]:
            level = "" if block.heading_level is None else f" lvl={block.heading_level}"
            page = "" if block.page_number is None else f" p{block.page_number}"
            source = "" if block.text_source is None else f" {block.text_source.value}"
            print(
                f"  [{block.block_index}]{page} {block.block_type.value}{level}"
                f"{source}: {_preview(block.text)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
