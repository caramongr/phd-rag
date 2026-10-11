"""RAG-006 tests 14-15: year extraction and weak PDF-creation-date handling."""

from __future__ import annotations

from pathlib import Path

from metadata_support import make_pdf

from app.metadata.enums import MetadataSource
from app.metadata.extractor import extract_metadata


def test_publication_year_extracted_where_explicit(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("Ordinary body text on the first page.", 10.0, 120.0),
            ("\u00a9 2019 Example Publisher. All rights reserved.", 9.0, 160.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.year is not None
    assert metadata.year.value == 2019
    assert metadata.year.source is MetadataSource.DOCUMENT_TEXT


def test_pdf_creation_year_does_not_override_publication_year(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        metadata={"creationDate": "D:20240101120000Z"},
        lines=[
            ("Ordinary body text on the first page.", 10.0, 120.0),
            ("Published 2019 by Example Press.", 9.0, 160.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.year is not None
    assert metadata.year.value == 2019
    assert metadata.year.source is MetadataSource.DOCUMENT_TEXT
    # The weaker creation-date candidate must remain inspectable in history.
    sources = {candidate.source for candidate in metadata.candidates["year"]}
    assert MetadataSource.PDF_METADATA in sources
