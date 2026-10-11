"""Qdrant payload construction for indexed chunks (RAG-008).

A payload must be enough to filter, cite and render a result **without
re-parsing the source document**, while staying small: the full RAG-006
metadata record (with its candidate history) never travels with a point, it stays
addressable through ``file_hash``.

Provenance is copied from the RAG-007 chunk and never re-derived, so the role
resolved from all physical locations (RAG-007 §25) is the role that gets indexed.
"""

from __future__ import annotations

from typing import Any

from app.models.chunk import Chunk

#: Fields RAG-009 filtering and citation rendering rely on.
INDEXED_PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "file_hash",
    "filename",
    "chunk_index",
    "page_start",
    "page_end",
    "section_title",
    "heading_path",
    "content_type",
    "corpus_role",
    "authority",
    "evidence_eligible",
    "retrieval_eligible",
    "text_sources",
    "ocr_involved",
    "title",
    "first_author",
    "year",
    "doi",
    "publication_type",
    "source_locators",
    "sequence_start",
    "sequence_end",
    "token_count",
    "text",
    "retrieval_text",
)

#: Optional audit fields, present only for documents with several physical copies.
AUDIT_PAYLOAD_FIELDS: tuple[str, ...] = ("duplicate_locations", "role_resolution")


def build_payload(chunk: Chunk, *, file_hash: str, filename: str | None = None) -> dict[str, Any]:
    """Build the Qdrant payload for ``chunk``.

    ``retrieval_text`` is what gets embedded; ``text`` is stored alongside it so a
    retrieved result can quote the bare source text, and the two are never mixed
    up (the synthetic labels exist only in ``retrieval_text``).
    """
    metadata = chunk.metadata_ref
    payload: dict[str, Any] = {
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "file_hash": file_hash,
        "filename": filename or (metadata.filename if metadata is not None else ""),
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "heading_path": list(chunk.heading_path),
        "content_type": chunk.content_type.value,
        "corpus_role": chunk.corpus_role.value if chunk.corpus_role else None,
        "authority": chunk.authority.value if chunk.authority else None,
        "evidence_eligible": bool(chunk.evidence_eligible),
        "retrieval_eligible": bool(chunk.retrieval_eligible),
        "text_sources": [source.value for source in chunk.text_sources],
        "ocr_involved": bool(chunk.ocr_involved),
        "source_locators": list(chunk.source_locators),
        "sequence_start": chunk.sequence_start,
        "sequence_end": chunk.sequence_end,
        "token_count": chunk.token_count,
        "text": chunk.text,
        "retrieval_text": chunk.retrieval_text if chunk.retrieval_text else chunk.text,
    }

    if metadata is not None:
        payload["title"] = metadata.title
        payload["first_author"] = metadata.first_author
        payload["year"] = metadata.year
        payload["doi"] = metadata.doi
        payload["publication_type"] = metadata.publication_type
        if metadata.duplicate_locations:
            payload["duplicate_locations"] = list(metadata.duplicate_locations)
        if metadata.role_resolution:
            payload["role_resolution"] = metadata.role_resolution
    else:
        payload["title"] = None
        payload["first_author"] = None
        payload["year"] = None
        payload["doi"] = None
        payload["publication_type"] = None

    return payload


def payload_is_complete(payload: dict[str, Any]) -> bool:
    """Whether every required field is present (``None`` values are allowed)."""
    return all(field in payload for field in INDEXED_PAYLOAD_FIELDS)
