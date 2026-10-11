"""Metadata value, author, and document metadata models (RAG-006).

The core type is :class:`MetadataValue`, a generic wrapper that binds a value to
its provenance (``source``), a controlled ``confidence`` level and a short
human-readable ``evidence`` string. Nothing in this module invents a value: a
missing field is represented by ``None`` and is a valid state.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.metadata.enums import (
    Confidence,
    CorpusRole,
    ExternalMatchStatus,
    MetadataSource,
    PublicationType,
)

T = TypeVar("T")


class MetadataValue(BaseModel, Generic[T]):
    """A single metadata value together with its provenance.

    ``evidence`` is a concise pointer to where the value came from (for example
    ``PDF metadata field "title"`` or ``Crossref DOI lookup``), never a large
    excerpt of document text.
    """

    model_config = ConfigDict(extra="forbid")

    value: T
    source: MetadataSource
    confidence: Confidence
    evidence: str

    @field_validator("evidence")
    @classmethod
    def _evidence_not_empty(cls, value: str) -> str:
        """Require a non-empty evidence string so provenance is never blank."""
        if not value.strip():
            raise ValueError("evidence cannot be empty")
        return value


class Author(BaseModel):
    """A normalized author.

    ``full_name`` is always authoritative. ``given_name`` and ``family_name``
    are populated only when they can be derived reliably; otherwise they stay
    ``None`` so no name part is invented.
    """

    model_config = ConfigDict(extra="forbid")

    full_name: str
    given_name: str | None = None
    family_name: str | None = None
    orcid: str | None = None

    @field_validator("full_name")
    @classmethod
    def _full_name_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("full_name cannot be empty")
        return value.strip()


class PhysicalRole(BaseModel):
    """One physical copy of a logical document and the role its path implies.

    Duplicate copies are binary-identical (same SHA-256), so they are one logical
    document, but they can live in different corpus folders and therefore carry
    different roles. Keeping the pair makes the effective role explainable.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    role: CorpusRole


class RoleResolution(BaseModel):
    """Why a logical document has the corpus role it has.

    A logical document may have several physical locations. When those locations
    imply different roles the most conservative role wins, so that a copy filed
    under ``literature`` can never promote the author's own work, a summary or a
    supervision note to scholarly evidence. The record keeps every location, every
    detected role, the effective role, the reason and the precedence that was
    applied, so the decision is inspectable rather than implicit in discovery
    order.
    """

    model_config = ConfigDict(extra="forbid")

    effective_role: CorpusRole
    reason: str
    physical_roles: list[PhysicalRole] = Field(default_factory=list)
    applied_precedence: list[CorpusRole] = Field(default_factory=list)

    @property
    def distinct_roles(self) -> list[CorpusRole]:
        """Detected roles, deduplicated, in precedence order."""
        seen: list[CorpusRole] = []
        for physical in self.physical_roles:
            if physical.role not in seen:
                seen.append(physical.role)
        return seen

    @property
    def has_conflicting_roles(self) -> bool:
        """Whether the physical locations disagree about the role."""
        return len(self.distinct_roles) > 1

    @property
    def physical_paths(self) -> list[str]:
        """Every physical path of the logical document, sorted for determinism."""
        return [physical.path for physical in self.physical_roles]

    def describe(self) -> str:
        """One-line explanation of the effective role for logs and reports."""
        roles = "+".join(role.value for role in self.distinct_roles) or "unknown"
        return f"{self.effective_role.value} ({self.reason}; locations={len(self.physical_roles)}, roles={roles})"


class MetadataConflict(BaseModel):
    """A recorded disagreement between a local value and an external one.

    A conflict preserves *both* candidates for inspection: the local value stays
    in :attr:`AcademicMetadata.candidates`, and the external value is captured
    here instead of being merged. Nothing is silently overwritten.
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    local_value: str
    external_value: str
    external_source: MetadataSource
    similarity: float
    route: str
    reason: str = "external record disagrees with a strong local value"


class AcademicMetadata(BaseModel):
    """All resolved academic metadata for one logical document.

    Every resolved field is a :class:`MetadataValue` (or ``None`` when missing).
    Resolved fields are selections over ``candidates``; the candidate history is
    retained so that a resolved value never destructively replaces the local
    value it superseded.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity (mirrors the discovery/parse stage for correlation)
    file_hash: str | None = None
    filename: str
    source_path: str
    extension: str
    corpus_role: CorpusRole = CorpusRole.OTHER
    #: Audit trail for ``corpus_role``: every physical location, every role the
    #: path implied, the effective role and the precedence that produced it.
    role_resolution: RoleResolution | None = None

    # Resolved bibliographic fields
    title: MetadataValue[str] | None = None
    authors: MetadataValue[list[Author]] | None = None
    year: MetadataValue[int] | None = None
    doi: MetadataValue[str] | None = None
    venue: MetadataValue[str] | None = None
    publisher: MetadataValue[str] | None = None
    publication_type: MetadataValue[PublicationType] | None = None
    url: MetadataValue[str] | None = None
    abstract: MetadataValue[str] | None = None
    arxiv_id: MetadataValue[str] | None = None

    # Audit trail: every candidate seen for every field, in discovery order.
    candidates: dict[str, list[MetadataValue[Any]]] = Field(default_factory=dict)

    # Result of the most recent external enrichment attempt.
    external_match_status: ExternalMatchStatus | None = None

    # Recorded disagreements between local values and external records.
    conflicts: list[MetadataConflict] = Field(default_factory=list)

    def candidate_count(self, field_name: str) -> int:
        """Return how many candidates were recorded for ``field_name``."""
        return len(self.candidates.get(field_name, []))

    def field_names(self) -> list[str]:
        """Return the sorted names of fields that have at least one candidate."""
        return sorted(self.candidates)
