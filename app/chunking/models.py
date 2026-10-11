"""Chunking-layer models: options, per-document results and diagnostics (RAG-007).

The retrieval payload itself is :class:`~app.models.chunk.Chunk`; this module
adds the configuration that produced it and the audit trail that explains what
was skipped and why.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.chunking.tokenizer import DEFAULT_TOKENIZER
from app.models.chunk import Chunk

#: Soft target band the accumulator aims for.
DEFAULT_TARGET_MIN_TOKENS = 700
DEFAULT_TARGET_MAX_TOKENS = 1000
#: No chunk may exceed this without a structural justification.
DEFAULT_HARD_MAX_TOKENS = 1200
#: Absolute ceiling (RAG-007): a chunk above this is a failure.
DEFAULT_EXTENDED_MAX_TOKENS = 1500


class ChunkingOptions(BaseModel):
    """Deterministic, auditable chunking configuration.

    ``tokenizer`` is a plain object rather than a model field: it is injected by
    the caller and excluded from serialisation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_min_tokens: int = Field(default=DEFAULT_TARGET_MIN_TOKENS, ge=1)
    target_max_tokens: int = Field(default=DEFAULT_TARGET_MAX_TOKENS, ge=1)
    hard_max_tokens: int = Field(default=DEFAULT_HARD_MAX_TOKENS, ge=1)
    extended_max_tokens: int = Field(default=DEFAULT_EXTENDED_MAX_TOKENS, ge=1)

    #: Merge a short section into an adjacent sibling within the same parent.
    merge_short_sections: bool = True
    #: Remove confidently detected running headers/footers.
    strip_running_lines: bool = True
    #: Drop prose blocks whose text is already represented by a structured table.
    deduplicate_table_text: bool = True
    #: Build ``retrieval_text`` with synthetic context labels.
    include_context_labels: bool = True

    #: Fixed timestamp for deterministic comparisons; ``None`` means "now".
    created_at: datetime | None = None

    #: Token counter (a plain object, not a validated model field).
    tokenizer: Any = Field(default_factory=lambda: DEFAULT_TOKENIZER)

    def model_post_init(self, _context: object) -> None:  # noqa: D105 - pydantic hook
        if not (
            self.target_min_tokens
            <= self.target_max_tokens
            <= self.hard_max_tokens
            <= self.extended_max_tokens
        ):
            raise ValueError(
                "token limits must satisfy target_min <= target_max <= hard_max "
                "<= extended_max"
            )

    def resolve_created_at(self) -> datetime:
        """Return the timestamp to stamp on chunks."""
        return self.created_at or datetime.now(timezone.utc)


class SkippedItem(BaseModel):
    """One parsed item that did not become chunk text, with the reason why."""

    model_config = ConfigDict(extra="forbid")

    sequence_index: int
    reason: str
    source_locator: str | None = None


class ChunkedDocument(BaseModel):
    """The chunking result for one logical document, with its audit trail."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    file_hash: str
    filename: str
    extension: str
    chunks: list[Chunk] = Field(default_factory=list)

    # Diagnostics
    skipped_items: list[SkippedItem] = Field(default_factory=list)
    duplicate_table_blocks_removed: int = 0
    running_lines_removed: int = 0
    headings_without_content: int = 0
    normalization_version: str = ""
    tokenizer_name: str = ""

    @property
    def chunk_count(self) -> int:
        """Number of chunks produced for this document."""
        return len(self.chunks)

    @property
    def total_tokens(self) -> int:
        """Sum of the chunk token counts."""
        return sum(chunk.token_count or 0 for chunk in self.chunks)


#: Reason codes recorded in :attr:`ChunkedDocument.skipped_items`.
REASON_DUPLICATE_TABLE_TEXT = "duplicate_table_text"
REASON_HEADING_WITHOUT_CONTENT = "heading_without_content"
REASON_EMPTY_AFTER_NORMALIZATION = "empty_after_normalization"
REASON_OCR_PAGE_WITHOUT_TEXT = "ocr_page_without_text"
