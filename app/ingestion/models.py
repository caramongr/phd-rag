"""Typed results produced by the corpus scanner.

These models describe the *discovery* stage only: files found on disk, their
hashes, their physical copies, and deterministic folder hints. They deliberately
exclude academic metadata (title, authors, year, ...) and any parsed content,
which arrive in later tasks.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import RelatedProject
from app.models.physical_location import PhysicalLocation


class DiscoveredDocument(BaseModel):
    """One logical document discovered on disk prior to metadata extraction.

    A logical document groups every physical copy that shares the same SHA-256
    hash. ``canonical_path`` and ``filename`` come from the first copy that was
    discovered.
    """

    model_config = ConfigDict(extra="forbid")

    file_hash: str
    filename: str
    canonical_path: str
    extension: str
    physical_locations: list[PhysicalLocation] = Field(default_factory=list)
    original_folders: list[str] = Field(default_factory=list)
    topic_hints: list[str] = Field(default_factory=list)
    related_project_hints: list[RelatedProject] = Field(default_factory=list)


class ScanError(BaseModel):
    """A file or directory that could not be processed during a scan."""

    model_config = ConfigDict(extra="forbid")

    path: str
    message: str


class ScanResult(BaseModel):
    """Aggregate outcome of scanning a directory tree."""

    model_config = ConfigDict(extra="forbid")

    scanned_files: int = 0
    supported_files: int = 0
    unsupported_files: int = 0
    unique_documents: int = 0
    duplicate_copies: int = 0
    errors: list[ScanError] = Field(default_factory=list)
    documents: list[DiscoveredDocument] = Field(default_factory=list)
    unsupported_paths: list[str] = Field(default_factory=list)
