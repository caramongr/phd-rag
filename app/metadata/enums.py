"""Controlled vocabularies for academic metadata (RAG-006).

Every provenance, confidence, publication-type, match-status and corpus-role
value is a :class:`~enum.StrEnum` so metadata stays validated, serialisable and
auditable. The vocabularies are intentionally small: an unknown value is
represented by an explicit ``UNKNOWN`` sentinel rather than a fabricated one.
"""

from __future__ import annotations

from enum import StrEnum


class MetadataSource(StrEnum):
    """Where a single metadata value came from.

    The distinct sources are never collapsed into one label: a title read from
    PDF metadata is distinguishable from a title inferred from the first page,
    which is again distinguishable from a title verified by Crossref.
    """

    PDF_METADATA = "PDF_METADATA"
    DOCX_PROPERTIES = "DOCX_PROPERTIES"
    FILENAME = "FILENAME"
    DOCUMENT_TEXT = "DOCUMENT_TEXT"
    CROSSREF = "CROSSREF"
    OPENALEX = "OPENALEX"
    MANUAL = "MANUAL"
    #: Folder-context provenance. Used only for the weak, non-scholarly
    #: classification of the author's own work (see the task's guidance on
    #: ``data/my_work``). Always carries ``LOW`` confidence so it never competes
    #: with a real bibliographic source.
    FOLDER = "FOLDER"


class Confidence(StrEnum):
    """A controlled confidence level, not pseudo-precise floating point.

    ``HIGH`` means externally verified or directly authoritative; ``MEDIUM``
    means a strong local signal that is still unverified; ``LOW`` means a weak
    local hint such as a filename token.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PublicationType(StrEnum):
    """The bibliographic nature of a document.

    ``UNKNOWN`` is used whenever the evidence does not support a more specific
    value. Folder layout is a hint, never authority, so it cannot by itself
    promote a document out of ``UNKNOWN``.
    """

    JOURNAL_ARTICLE = "JOURNAL_ARTICLE"
    CONFERENCE_PAPER = "CONFERENCE_PAPER"
    BOOK = "BOOK"
    BOOK_CHAPTER = "BOOK_CHAPTER"
    THESIS = "THESIS"
    REPORT = "REPORT"
    PREPRINT = "PREPRINT"
    WEB_DOCUMENT = "WEB_DOCUMENT"
    AUTHOR_MANUSCRIPT = "AUTHOR_MANUSCRIPT"
    UNKNOWN = "UNKNOWN"


class ExternalMatchStatus(StrEnum):
    """How well an external record matches a local document.

    ``AMBIGUOUS`` forbids enrichment when the *search* returned several equally
    plausible records. ``CONFLICT`` is distinct: an exact DOI resolution that
    disagrees materially with a strong local title, i.e. the identifier and the
    document's own first-page identity do not describe the same work.
    """

    EXACT = "EXACT"
    STRONG = "STRONG"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT = "CONFLICT"
    NO_MATCH = "NO_MATCH"


class CorpusRole(StrEnum):
    """The role a document plays in the corpus, derived from its folder.

    The role gates external enrichment: scholarly literature may be enriched,
    whereas the author's own work, rules, supervision records, guidance, future
    chapters and internal summaries must not be treated as external
    publications.
    """

    LITERATURE = "literature"
    MY_WORK = "my_work"
    RULES = "rules"
    SUPERVISION = "supervision"
    FUTURE_CHAPTERS = "future_chapters"
    SUMMARIES = "summaries"
    GUIDANCE = "guidance"
    OTHER = "other"
