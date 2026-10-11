"""Academic metadata extraction and controlled enrichment (RAG-006).

This package identifies document-level bibliographic metadata (title, authors,
year, DOI, venue, publication type, URL) with explicit provenance and
confidence. It never fabricates values and never silently overwrites one
candidate with another: every value records where it came from and the full
candidate history stays inspectable.

It does not chunk, embed, index, rerank, or generate anything.
"""

from app.metadata.enums import (
    Confidence,
    CorpusRole,
    ExternalMatchStatus,
    MetadataSource,
    PublicationType,
)
from app.metadata.models import (
    AcademicMetadata,
    Author,
    MetadataConflict,
    MetadataValue,
)

__all__ = [
    "AcademicMetadata",
    "Author",
    "Confidence",
    "CorpusRole",
    "ExternalMatchStatus",
    "MetadataConflict",
    "MetadataSource",
    "MetadataValue",
    "PublicationType",
]
