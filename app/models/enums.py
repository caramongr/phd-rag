"""Controlled metadata vocabularies used across the domain models.

Every controlled field on :class:`~app.models.document.Document` and
:class:`~app.models.chunk.Chunk` is backed by a :class:`~enum.StrEnum` so that
values are validated, serialisable, and easy to audit.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    """The nature of a source document in the academic corpus."""

    PUBLISHED_PAPER = "published_paper"
    CONFERENCE_PAPER = "conference_paper"
    BOOK = "book"
    REPORT = "report"
    THESIS = "thesis"
    AUTHOR_PAPER = "author_paper"
    SUMMARY = "summary"
    RESEARCH_NOTE = "research_note"
    SUPERVISION_RECORD = "supervision_record"
    SUPERVISOR_FEEDBACK = "supervisor_feedback"
    GUIDELINE = "guideline"


class PublicationStatus(StrEnum):
    """Lifecycle state of a document at import time."""

    PUBLISHED = "published"
    SUBMITTED = "submitted"
    DRAFT = "draft"
    WORKING_DRAFT = "working_draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class AuthorityLevel(StrEnum):
    """How much evidential weight the document carries.

    ``PRIMARY_EVIDENCE`` marks peer-reviewed or otherwise first-hand academic
    sources, while ``NAVIGATION_ONLY`` and ``INSTRUCTION`` cover documents that
    guide reading or behaviour rather than supply evidence.
    """

    PRIMARY_EVIDENCE = "primary_evidence"
    HIGH = "high"
    MEDIUM = "medium"
    NAVIGATION_ONLY = "navigation_only"
    INSTRUCTION = "instruction"


class RelatedProject(StrEnum):
    """The research project(s) a document is filed under."""

    THESIS = "thesis"
    SUBMITTED_PAPER = "submitted_paper"
    SECOND_PAPER = "second_paper"
    GENERAL_REFERENCE = "general_reference"


class ContentType(StrEnum):
    """The kind of content a chunk represents within its document."""

    BODY = "body"
    ABSTRACT = "abstract"
    TABLE = "table"
    REFERENCES = "references"
    CONCLUSION = "conclusion"
    SUMMARY_ENTRY = "summary_entry"
    #: A reliably identified figure/table caption or other labelled exhibit text.
    CAPTION = "caption"
    #: Prose from an appendix section, kept distinct from the main body.
    APPENDIX = "appendix"
    OTHER = "other"
