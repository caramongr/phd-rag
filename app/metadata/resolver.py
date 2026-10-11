"""Deterministic, field-specific metadata resolution (RAG-006).

Resolution selects one *preferred* value per field from the recorded candidates.
It never discards the others: every candidate stays in the history so a resolved
value never destructively replaces the local value it superseded.

Priority is field-specific rather than one global ordering. For example a
publication year read from document text outranks a PDF creation year, while a
PDF embedded title outranks a first-page text candidate.
"""

from __future__ import annotations

from app.metadata.enums import Confidence, MetadataSource
from app.metadata.models import MetadataValue

_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.HIGH: 3,
    Confidence.MEDIUM: 2,
    Confidence.LOW: 1,
}

_DEFAULT_PRIORITY: tuple[MetadataSource, ...] = (
    MetadataSource.MANUAL,
    MetadataSource.CROSSREF,
    MetadataSource.OPENALEX,
    MetadataSource.PDF_METADATA,
    MetadataSource.DOCX_PROPERTIES,
    MetadataSource.DOCUMENT_TEXT,
    MetadataSource.FILENAME,
)

#: Per-field source priority, most authoritative first.
FIELD_PRIORITY: dict[str, tuple[MetadataSource, ...]] = {
    "title": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.PDF_METADATA,
        MetadataSource.DOCX_PROPERTIES,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.FILENAME,
    ),
    "authors": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.PDF_METADATA,
        MetadataSource.DOCX_PROPERTIES,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.FILENAME,
    ),
    # Publication year: a text-derived publication line outranks a weak PDF
    # creation date.
    "year": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.PDF_METADATA,
        MetadataSource.DOCX_PROPERTIES,
        MetadataSource.FILENAME,
    ),
    "doi": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.PDF_METADATA,
        MetadataSource.FILENAME,
    ),
    "venue": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCUMENT_TEXT,
    ),
    "publisher": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCX_PROPERTIES,
        MetadataSource.DOCUMENT_TEXT,
    ),
    "publication_type": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCX_PROPERTIES,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.FILENAME,
    ),
    "url": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.PDF_METADATA,
    ),
    "abstract": (
        MetadataSource.MANUAL,
        MetadataSource.CROSSREF,
        MetadataSource.OPENALEX,
        MetadataSource.DOCUMENT_TEXT,
    ),
    "arxiv_id": (
        MetadataSource.MANUAL,
        MetadataSource.DOCUMENT_TEXT,
        MetadataSource.FILENAME,
        MetadataSource.PDF_METADATA,
    ),
}


def add_candidate(
    candidates: dict[str, list[MetadataValue]], field_name: str, value: MetadataValue
) -> None:
    """Append ``value`` to the candidate list for ``field_name``."""
    candidates.setdefault(field_name, []).append(value)


def resolve_field(
    field_name: str, candidates: list[MetadataValue]
) -> MetadataValue | None:
    """Return the preferred candidate for ``field_name``, or ``None``.

    Sources are ranked per field; ties are broken by confidence and then by
    discovery order, so resolution is fully deterministic.
    """
    if not candidates:
        return None
    priority = FIELD_PRIORITY.get(field_name, _DEFAULT_PRIORITY)
    rank = {source: index for index, source in enumerate(priority)}

    best: MetadataValue | None = None
    best_key: tuple[int, int, int] | None = None
    for order, candidate in enumerate(candidates):
        key = (
            rank.get(candidate.source, len(priority)),
            -_CONFIDENCE_RANK[candidate.confidence],
            order,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = candidate
    return best


def resolve_all(
    candidates: dict[str, list[MetadataValue]],
) -> dict[str, MetadataValue | None]:
    """Resolve every field that has at least one candidate."""
    return {
        field_name: resolve_field(field_name, values)
        for field_name, values in candidates.items()
    }
