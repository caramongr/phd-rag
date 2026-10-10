"""Explicit parse errors.

Parsing never swallows failures into an empty document: callers either receive a
fully parsed :class:`~app.parsing.models.ParsedDocument` or one of these
exceptions with the offending path preserved in the message.
"""

from __future__ import annotations


class DocumentParseError(Exception):
    """Raised when a document cannot be parsed.

    The message always includes the source path so failures are traceable.
    """


class UnsupportedDocumentFormat(DocumentParseError):
    """Raised when no parser is registered for a file's extension."""


class EncryptedDocumentError(DocumentParseError):
    """Raised when a document is password-protected and cannot be opened.

    RAG-004 never attempts password guessing.
    """
