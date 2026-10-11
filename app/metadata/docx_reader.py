"""Read-only DOCX core-properties reader (RAG-006).

Reads the ``docProps/core.xml`` values that ``python-docx`` exposes as
``core_properties``. These are structured document properties (source
``DOCX_PROPERTIES``) and are distinct from a ``Title``-styled paragraph, which
is document text.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from docx import Document as DocxDocument
from pydantic import BaseModel, ConfigDict

from app.parsing.exceptions import DocumentParseError


class DocxProperties(BaseModel):
    """The DOCX core properties, unvalidated."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    created: datetime | None = None
    modified: datetime | None = None
    last_modified_by: str | None = None


def read_docx_properties(path: str | Path) -> DocxProperties:
    """Read the core properties of a ``.docx`` file.

    Raises:
        DocumentParseError: if the file is missing or cannot be opened.
    """
    path = Path(path)
    if not path.is_file():
        raise DocumentParseError(f"Document not found: {path}")

    try:
        document = DocxDocument(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Failed to parse DOCX '{path}': {exc}") from exc

    core = document.core_properties
    return DocxProperties(
        title=_clean(core.title),
        author=_clean(core.author),
        subject=_clean(core.subject),
        keywords=_clean(core.keywords),
        created=core.created if isinstance(core.created, datetime) else None,
        modified=core.modified if isinstance(core.modified, datetime) else None,
        last_modified_by=_clean(core.last_modified_by),
    )


def _clean(value: object) -> str | None:
    """Return a stripped string, or ``None`` for empty/missing values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
