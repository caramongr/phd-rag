"""The :class:`PhysicalLocation` model.

A physical location records one concrete file on disk that belongs to a logical
document. A single logical document may have several physical copies (binary
duplicates detected via SHA-256), and each copy keeps the folder it was found in
as a provenance hint.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PhysicalLocation(BaseModel):
    """One physical copy of a document on disk."""

    model_config = ConfigDict(extra="forbid")

    path: str
    folder: str | None = None
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp for when this copy was discovered.",
    )

    @field_validator("path")
    @classmethod
    def _path_must_not_be_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only paths."""
        if not value.strip():
            raise ValueError("path cannot be empty")
        return value
