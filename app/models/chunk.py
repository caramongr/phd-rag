"""The :class:`Chunk` domain model.

A ``Chunk`` is a contiguous, citable slice of a :class:`~app.models.document.Document`
destined for embedding and retrieval. It records where the slice came from
(page range, section, heading trail) so answers can point back to the source.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import ContentType


class Chunk(BaseModel):
    """A retrievable slice of a document.

    ``chunk_index`` is the zero-based position of the chunk within its parent
    document. Page numbers, when present, are one-based. ``heading_path`` is the
    ordered trail of headings the chunk sits under (outermost first).
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
