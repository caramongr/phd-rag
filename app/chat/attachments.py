"""Attachment storage for the simple chat (RAG-015 §5, §6).

An attachment is a *working document for one conversation*. It is copied into
``data/chat/attachments/<chat id>/`` — never into ``data/literature`` or ``data/my_work``, and
never chunked, embedded or indexed — and its extracted text is stored with the chat so
reopening the conversation does not need the file again.

Text extraction reuses the project's own parsing layer: :func:`app.parsing.parser.parse_document`
handles PDF, DOCX and Markdown exactly as the ingestion pipeline does. Plain ``.txt`` is the
one exception, and deliberately so: RAG-003 decided that ``.txt`` is not a corpus document
format, so the dispatcher still rejects it (a test asserts that) and this module decodes a text
file directly into the same :class:`~app.parsing.models.ParsedDocument` shape instead of
changing that decision.

Two safety properties are built in, not bolted on:

* the stored path is always built from identifiers *this* module generates, so a filename like
  ``../../../etc/passwd`` can never become a path — it is only ever displayed as text;
* the resolved path is checked to be inside the attachment root before anything is written.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.chat.models import (
    AttachmentLimitError,
    AttachmentTextStatus,
    AttachmentTooLargeError,
    ChatAttachment,
    UnsupportedAttachmentError,
    utc_now,
)
from app.config import Settings, get_settings
from app.generation.context import estimate_tokens

#: Content types for the formats the chat accepts. A small fixed table beats a dependency.
CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

#: Characters kept in a filename before it is displayed; longer names are clipped.
MAX_FILENAME_CHARS = 200
#: Identifier shape this module generates: a hex id, optionally with a short extension.
#: Anything that does not match is refused outright, so a filename from a request can never
#: become part of a path.
_IDENTIFIER_RE = re.compile(r"^[0-9a-f]{8,64}(\.[a-z0-9]{1,8})?$")
#: Per-document overhead counted against the context budget (header, delimiters).
DOCUMENT_OVERHEAD_TOKENS = 32


@dataclass(frozen=True)
class StoredAttachment:
    """What writing one attachment produced, before it is recorded in the store."""

    id: str
    filename: str
    stored_path: str
    content_type: str
    extension: str
    size_bytes: int
    text_status: AttachmentTextStatus
    text: str
    char_count: int
    token_estimate: int
    parse_error: str | None = None
    created_at: str = ""


class AttachmentStore:
    """Writes chat attachments to disk and extracts their text."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        settings: Settings | None = None,
        allowed_extensions: tuple[str, ...] | None = None,
        max_bytes: int | None = None,
        max_per_chat: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.root = Path(
            root if root is not None else (self.settings.chat_attachment_dir or "data/chat/attachments")
        )
        self.allowed_extensions = tuple(
            extension.lower()
            for extension in (
                allowed_extensions
                if allowed_extensions is not None
                else self.settings.chat_allowed_extensions
            )
        )
        self.max_bytes = int(
            max_bytes if max_bytes is not None else self.settings.chat_max_attachment_bytes
        )
        self.max_per_chat = int(
            max_per_chat if max_per_chat is not None else self.settings.chat_max_attachments_per_chat
        )

    # --- validation -----------------------------------------------------------
    @property
    def accepted_formats(self) -> str:
        """The accepted extensions, for a readable error message."""
        return ", ".join(extension.upper().lstrip(".") for extension in self.allowed_extensions)

    @staticmethod
    def display_filename(filename: str) -> str:
        """A safe name to show for an uploaded file (never used to build a path)."""
        candidate = (filename or "").replace("\\", "/").split("/")[-1]
        candidate = "".join(
            character
            for character in unicodedata.normalize("NFC", candidate)
            if character.isprintable()
        ).strip()
        if not candidate:
            return "attachment"
        if len(candidate) > MAX_FILENAME_CHARS:
            stem = Path(candidate).stem[: MAX_FILENAME_CHARS - 20]
            candidate = stem + Path(candidate).suffix
        return candidate

    def validate(self, filename: str, size_bytes: int, *, existing: int = 0) -> tuple[str, str]:
        """Check a name and a size, returning ``(display name, extension)``.

        Raises:
            UnsupportedAttachmentError: the extension is not accepted.
            AttachmentTooLargeError: the file exceeds the configured limit.
            AttachmentLimitError: the chat already holds the maximum number of attachments.
        """
        display = self.display_filename(filename)
        extension = Path(display).suffix.lower()
        if extension not in self.allowed_extensions:
            raise UnsupportedAttachmentError(
                f"'{display}' is not a supported attachment. Attach {self.accepted_formats} "
                "files: PDF, DOCX and Markdown are read with the project's parsers, and plain "
                "text is read as text."
            )
        if size_bytes <= 0:
            raise UnsupportedAttachmentError(
                f"'{display}' is empty. Attach a file with content in it."
            )
        if size_bytes > self.max_bytes:
            raise AttachmentTooLargeError(
                f"'{display}' is {_megabytes(size_bytes)}; the limit for one attachment is "
                f"{_megabytes(self.max_bytes)}. Attach a smaller file, or split the document."
            )
        if existing >= self.max_per_chat:
            raise AttachmentLimitError(
                f"This chat already has {existing} attachments, which is the maximum. "
                "Remove one, or start a new chat for a different document."
            )
        return display, extension

    # --- storage --------------------------------------------------------------
    def save(self, chat_id: str, filename: str, data: bytes, *, existing: int = 0) -> StoredAttachment:
        """Validate, store and parse one attachment.

        The file is written by this module's own identifiers, so the caller's filename never
        reaches the filesystem.
        """
        from app.chat.store import new_id

        display, extension = self.validate(filename, len(data), existing=existing)
        attachment_id = new_id()
        directory = self.directory_for(chat_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = self._checked_path(chat_id, f"{attachment_id}{extension}")
        target.write_bytes(data)

        text, status, error = self.extract_text(target, extension)
        return StoredAttachment(
            id=attachment_id,
            filename=display,
            stored_path=str(target),
            content_type=CONTENT_TYPES.get(extension, "application/octet-stream"),
            extension=extension,
            size_bytes=len(data),
            text_status=status,
            text=text,
            char_count=len(text),
            token_estimate=estimate_tokens(text) + DOCUMENT_OVERHEAD_TOKENS if text else 0,
            parse_error=error,
            created_at=utc_now(),
        )

    def extract_text(
        self, path: Path, extension: str
    ) -> tuple[str, AttachmentTextStatus, str | None]:
        """Extract the text of a stored attachment with the project's parsers.

        A parse failure never removes the attachment: the file is kept, the status says
        ``failed``, and the message explains what happened.
        """
        if extension == ".txt":
            return self._read_text(path)
        try:
            from app.parsing.parser import parse_document

            document = parse_document(path)
        except Exception as exc:  # a typed DocumentParseError, or a parser surprise
            return "", AttachmentTextStatus.FAILED, _readable_error(exc)
        text = (document.raw_text or "").strip()
        if not text:
            return "", AttachmentTextStatus.EMPTY, None
        return text, AttachmentTextStatus.PARSED, None

    def _read_text(self, path: Path) -> tuple[str, AttachmentTextStatus, str | None]:
        """Read a plain-text file (the one format RAG-003's dispatcher rejects)."""
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = raw.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
            if not text:
                return "", AttachmentTextStatus.EMPTY, None
            return text, AttachmentTextStatus.PARSED, None
        return "", AttachmentTextStatus.EMPTY, None

    # --- paths ----------------------------------------------------------------
    def directory_for(self, chat_id: str) -> Path:
        """The directory holding one chat's attachments."""
        return self._checked_path(chat_id)

    def _checked_path(self, *parts: str) -> Path:
        """Build a path under the root, refusing anything that would escape it."""
        for part in parts:
            if not _IDENTIFIER_RE.match(part):
                raise ValueError(f"refusing an unsafe attachment path component: {part!r}")
        candidate = self.root.joinpath(*parts)
        root = self.root.resolve()
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("refusing an attachment path outside the attachment directory")
        return candidate

    # --- deletion -------------------------------------------------------------
    def delete_file(self, attachment: ChatAttachment) -> None:
        """Remove one stored attachment file, if it is still there."""
        path = Path(attachment.stored_path)
        try:
            if path.is_file():
                path.unlink()
        except OSError:  # pragma: no cover - a file we cannot remove is not fatal
            pass

    def delete_chat_files(self, chat_id: str) -> None:
        """Remove every stored file for one chat (called when a chat is deleted)."""
        try:
            directory = self.directory_for(chat_id)
        except ValueError:  # pragma: no cover - defensive
            return
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _readable_error(exc: Exception) -> str:
    """One sentence a person can act on, without a traceback."""
    message = " ".join(str(exc).split())
    if not message:
        return f"the document could not be read ({type(exc).__name__})"
    return message[:400]


__all__ = ["AttachmentStore", "CONTENT_TYPES", "StoredAttachment"]
