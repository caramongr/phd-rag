"""The :class:`Chunk` domain model.

A ``Chunk`` is a contiguous, citable slice of a :class:`~app.models.document.Document`
destined for embedding and retrieval. It records where the slice came from
(page range, section, heading trail) so answers can point back to the source.

RAG-007 extended the model additively (RAG-001 usage keeps working) with what
retrieval needs in order to stay honest:

* ``retrieval_text`` is the contextualised text destined for embedding, while
  ``text`` stays the normalized *source* content. Synthetic context labels such
  as ``[Section: …]`` only ever appear in ``retrieval_text``.
* ``text_sources`` records whether native extraction, OCR, or both produced the
  text, and ``source_locators`` / ``sequence_start`` … ``sequence_end`` point
  back to the parsed source items, so a chunk is never anonymous text.
* ``corpus_role``, ``authority``, ``retrieval_eligible`` and ``evidence_eligible``
  carry the corpus-role policy (a summary, a supervision note or a future chapter
  is retrievable but is not scholarly evidence).
* ``metadata_ref`` is a minimal bibliographic payload; the full metadata stays
  addressable by ``file_hash``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.metadata.enums import CorpusRole, PublicationType
from app.models.enums import AuthorityLevel, ContentType
from app.parsing.models import TextSource


class ChunkMetadataRef(BaseModel):
    """A minimal bibliographic reference for retrieval and ranking.

    This is deliberately *not* a copy of the full
    :class:`~app.metadata.models.AcademicMetadata`: the complete record (with its
    candidate history and provenance) stays addressable through ``file_hash``.
    """

    model_config = ConfigDict(extra="forbid")

    file_hash: str
    filename: str
    title: str | None = None
    year: int | None = None
    first_author: str | None = None
    doi: str | None = None
    publication_type: PublicationType | None = None
    #: All physical copies of this logical document when the same bytes were found
    #: in more than one place (sorted paths; empty for the ordinary single-copy
    #: case). Populated so a chunk can explain why it carries its role.
    duplicate_locations: list[str] = Field(default_factory=list)
    #: Why the effective corpus role was chosen when copies disagreed; ``None``
    #: when the document has a single location.
    role_resolution: str | None = None


class Chunk(BaseModel):
    """A retrievable slice of a document.

    ``chunk_index`` is the zero-based position of the chunk within its parent
    document. Page numbers, when present, are one-based. ``heading_path`` is the
    ordered trail of headings the chunk sits under (outermost first).

    RAG-007 fields default to the "one section, native text, eligible" case, so a
    minimal chunk stays exactly as valid as it was in RAG-001.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID
    document_id: UUID
    chunk_index: int = Field(ge=0)
    text: str

    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)

    token_count: int | None = None
    content_type: ContentType

    created_at: datetime

    # --- RAG-007: retrieval text -------------------------------------------------
    #: Contextualised text for embedding. May carry synthetic section labels that
    #: are *not* present in the source document; ``text`` never does.
    retrieval_text: str | None = None

    # --- RAG-007: provenance -----------------------------------------------------
    #: Distinct extraction sources that contributed text (``NATIVE_PDF``/``OCR``).
    text_sources: list[TextSource] = Field(default_factory=list)
    #: ``source_locator`` values of the parsed items that produced this chunk.
    source_locators: list[str] = Field(default_factory=list)
    #: Inclusive range over the parsed canonical content stream (``sequence_index``).
    sequence_start: int | None = None
    sequence_end: int | None = None
    #: Minimal bibliographic payload; ``None`` when no metadata was available.
    metadata_ref: ChunkMetadataRef | None = None

    # --- RAG-007: policy ---------------------------------------------------------
    corpus_role: CorpusRole | None = None
    authority: AuthorityLevel | None = None
    #: ``False`` for content that must not answer normal evidence retrieval
    #: (reference lists, and roles such as summaries or future chapters).
    retrieval_eligible: bool = True
    #: ``False`` for anything that must not silently support a scholarly claim.
    evidence_eligible: bool = True

    # --- RAG-007: structure ------------------------------------------------------
    #: Sibling section titles merged into this chunk to avoid a tiny chunk.
    merged_sections: list[str] = Field(default_factory=list)
    #: ``True`` when this chunk continues a paragraph that exceeded the hard limit.
    split_from_previous: bool = False
    #: Tokens copied from the previous chunk. Always ``0`` in RAG-007: structural
    #: context (section, heading path, metadata) replaces blind token overlap.
    overlap_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_page_range(self) -> "Chunk":
        """Ensure ``page_end`` is never lower than ``page_start``."""
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end cannot be lower than page_start")
        return self

    @model_validator(mode="after")
    def _validate_sequence_range(self) -> "Chunk":
        """Ensure ``sequence_end`` is never lower than ``sequence_start``."""
        if (
            self.sequence_start is not None
            and self.sequence_end is not None
            and self.sequence_end < self.sequence_start
        ):
            raise ValueError("sequence_end cannot be lower than sequence_start")
        return self

    @property
    def ocr_involved(self) -> bool:
        """Whether any part of this chunk came from OCR (RAG-005)."""
        return TextSource.OCR in self.text_sources
