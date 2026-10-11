"""RAG-006 tests 1-4: PDF embedded metadata acceptance and rejection."""

from __future__ import annotations

from pathlib import Path

from metadata_support import make_pdf

from app.metadata.enums import MetadataSource
from app.metadata.extractor import extract_metadata

_BODY = [("Some ordinary body text on the first page.", 10.0, 200.0)]


def test_meaningful_embedded_title_accepted(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        metadata={"title": "A Meaningful Academic Title", "author": "Jane Doe"},
        lines=_BODY,
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "A Meaningful Academic Title"
    assert metadata.title.source is MetadataSource.PDF_METADATA
    assert metadata.title.confidence.value == "MEDIUM"


def test_blank_embedded_title_rejected(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, metadata={"title": "   "}, lines=_BODY)
    metadata = extract_metadata(path)

    assert metadata.title is None


def test_useless_untitled_rejected(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, metadata={"title": "untitled"}, lines=_BODY)
    metadata = extract_metadata(path)

    assert metadata.title is None


def test_embedded_author_captured(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        metadata={"title": "A Meaningful Academic Title", "author": "Jane Doe; John Smith"},
        lines=_BODY,
    )
    metadata = extract_metadata(path)

    assert metadata.authors is not None
    assert [author.full_name for author in metadata.authors.value] == ["Jane Doe", "John Smith"]
    assert metadata.authors.source is MetadataSource.PDF_METADATA
