"""The :class:`Document` domain model.

A ``Document`` is a single imported source file together with the metadata the
RAG system needs to cite, filter, and reason over it. It intentionally holds no
retrieval state or storage concerns.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    AuthorityLevel,
    DocumentType,
    PublicationStatus,
    RelatedProject,
)
from app.models.physical_location import PhysicalLocation


class Document(BaseModel):
    """Metadata for one source document in the academic corpus.

    Identity and provenance are captured by ``id``, ``file_hash``,
    ``filename``, ``source_path`` and ``original_folders``. Bibliographic
    fields (``title``, ``authors``, ``year``, ``doi``, ``url``) may be unknown
    until enrichment and therefore default to ``None`` or empty lists.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity & provenance
    id: UUID
    file_hash: str
    filename: str
    source_path: str
    physical_locations: list[PhysicalLocation] = Field(default_factory=list)
    original_folders: list[str] = Field(default_factory=list)

    # Bibliographic metadata
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None

    # Classification
    document_type: DocumentType
    publication_status: PublicationStatus
    authority: AuthorityLevel

    topics: list[str] = Field(default_factory=list)
    related_projects: list[RelatedProject] = Field(default_factory=list)

    # Physical / format details
    language: str | None = None
    page_count: int | None = None

    # Versioning
    version: str | None = None
    supersedes_document_id: UUID | None = None

    # Lifecycle
    imported_at: datetime
    updated_at: datetime
    notes: str | None = None
