"""RAG-006: metadata value model, confidence, and field-specific resolution."""

from __future__ import annotations

import pytest

from app.metadata.enums import Confidence, MetadataSource
from app.metadata.models import MetadataValue
from app.metadata.resolver import add_candidate, resolve_all, resolve_field


def _value(value, source: MetadataSource, confidence: Confidence, evidence: str = "e"):
    return MetadataValue(value=value, source=source, confidence=confidence, evidence=evidence)


def test_metadata_value_requires_evidence() -> None:
    with pytest.raises(ValueError):
        MetadataValue(value="x", source=MetadataSource.PDF_METADATA, confidence=Confidence.LOW, evidence="  ")


def test_title_prefers_crossref_over_pdf_metadata() -> None:
    candidates = [
        _value("Local Embedded Title", MetadataSource.PDF_METADATA, Confidence.MEDIUM),
        _value("Verified Title", MetadataSource.CROSSREF, Confidence.HIGH),
    ]
    resolved = resolve_field("title", candidates)
    assert resolved is not None
    assert resolved.value == "Verified Title"
    assert resolved.source is MetadataSource.CROSSREF


def test_year_prefers_document_text_over_pdf_creation_date() -> None:
    candidates = [
        _value(2024, MetadataSource.PDF_METADATA, Confidence.LOW),
        _value(2019, MetadataSource.DOCUMENT_TEXT, Confidence.MEDIUM),
    ]
    resolved = resolve_field("year", candidates)
    assert resolved is not None
    assert resolved.value == 2019
    assert resolved.source is MetadataSource.DOCUMENT_TEXT


def test_resolution_is_deterministic_and_keeps_history() -> None:
    candidates: dict = {}
    add_candidate(candidates, "title", _value("Local", MetadataSource.DOCUMENT_TEXT, Confidence.MEDIUM))
    add_candidate(candidates, "title", _value("External", MetadataSource.CROSSREF, Confidence.HIGH))

    resolved = resolve_all(candidates)

    assert resolved["title"].value == "External"
    # The local candidate must remain inspectable (never destructively replaced).
    assert len(candidates["title"]) == 2
    assert {candidate.value for candidate in candidates["title"]} == {"Local", "External"}


def test_confidence_ordering_is_controlled_vocabulary() -> None:
    assert {level.value for level in Confidence} == {"HIGH", "MEDIUM", "LOW"}


def test_metadata_source_has_distinct_bibliographic_sources() -> None:
    values = {source.value for source in MetadataSource}
    assert {
        "PDF_METADATA",
        "DOCX_PROPERTIES",
        "FILENAME",
        "DOCUMENT_TEXT",
        "CROSSREF",
        "OPENALEX",
        "MANUAL",
    }.issubset(values)
