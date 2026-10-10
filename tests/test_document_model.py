"""Tests for the :class:`Document` domain model."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.document import Document
from app.models.enums import (
    AuthorityLevel,
    DocumentType,
    PublicationStatus,
    RelatedProject,
)
from app.models.physical_location import PhysicalLocation


def make_document(**overrides: object) -> Document:
    """Build a valid Document, overriding individual fields as needed."""
    defaults: dict[str, object] = {
        "id": uuid4(),
        "file_hash": "abc123",
        "filename": "paper.pdf",
        "source_path": "F:/library/paper.pdf",
        "document_type": DocumentType.PUBLISHED_PAPER,
        "publication_status": PublicationStatus.PUBLISHED,
        "authority": AuthorityLevel.PRIMARY_EVIDENCE,
        "imported_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def test_minimal_document_defaults() -> None:
    doc = make_document()
    assert doc.original_folders == []
    assert doc.authors == []
    assert doc.topics == []
    assert doc.related_projects == []
    assert doc.title is None
    assert doc.year is None
    assert doc.supersedes_document_id is None


def test_document_accepts_full_metadata() -> None:
    supersedes = uuid4()
    doc = make_document(
        original_folders=["Thesis", "Papers"],
        title="A Study of Retrieval",
        authors=["Ada Lovelace", "Alan Turing"],
        year=2024,
        doi="10.1000/xyz",
        url="https://example.org/paper",
        topics=["rag", "nlp"],
        related_projects=[RelatedProject.THESIS, RelatedProject.GENERAL_REFERENCE],
        language="en",
        page_count=12,
        version="v2",
        supersedes_document_id=supersedes,
        notes="imported by hand",
    )
    assert doc.title == "A Study of Retrieval"
    assert doc.authors == ["Ada Lovelace", "Alan Turing"]
    assert doc.related_projects == [
        RelatedProject.THESIS,
        RelatedProject.GENERAL_REFERENCE,
    ]
    assert doc.supersedes_document_id == supersedes


def test_document_serialises_enums_to_values() -> None:
    payload = make_document().model_dump(mode="json")
    assert payload["document_type"] == "published_paper"
    assert payload["publication_status"] == "published"
    assert payload["authority"] == "primary_evidence"


def test_document_round_trips_through_json() -> None:
    doc = make_document()
    restored = Document.model_validate(doc.model_dump(mode="json"))
    assert restored == doc


def test_invalid_enum_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_document(document_type="not_a_type")


def test_missing_required_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Document(filename="only-a-name.pdf")  # type: ignore[call-arg]


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_document(unexpected="boom")


def test_physical_locations_default_to_empty() -> None:
    assert make_document().physical_locations == []


def test_document_accepts_physical_locations() -> None:
    locations = [
        PhysicalLocation(path="F:/data/AI/smith2024.pdf", folder="AI"),
        PhysicalLocation(path="F:/data/Algorithms/smith2024-copy.pdf", folder="Algorithms"),
    ]
    doc = make_document(physical_locations=locations)
    assert doc.physical_locations == locations
