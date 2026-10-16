"""Local persistence for chats, messages and attachment metadata (RAG-015 §4).

One SQLite file, three tables, no ORM: the schema is small enough to read, and the store is
the only place that knows SQL. It is deliberately the *only* storage for conversations, so
"restart the server" and "close the browser" are not different events — reopening a chat
replays exactly what was written.

The connection is shared by the web process's threads (FastAPI runs synchronous endpoints in
a thread pool) and every statement is serialised behind one lock. Writes are small and local,
so the lock costs nothing measurable and removes a class of "database is locked" noise.

An attachment's *text* is stored in the database, next to its metadata, because that is what
makes reopening a chat enough: the parsed working document has to survive a restart without
the file being attached again. The file itself is stored by
:mod:`app.chat.attachments`; this module only records where it is.

Schema changes are additive and backed up (RAG-016 §2). A database written by an earlier stage
is *altered* in place — columns are added to ``messages``, nothing is dropped, recreated or
reordered — and a timestamped copy of the file is taken first, inside the same module that
knows the schema. A failure during the migration rolls back and raises, so the server stops
cleanly instead of running against a half-changed database.
"""

from __future__ import annotations

import datetime as _datetime
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.chat.models import (
    AttachmentTextStatus,
    Chat,
    ChatAttachment,
    ChatMessage,
    ChatNotFoundError,
    ChatRole,
    ChatSummary,
    utc_now,
)

#: What a chat is called before the user names it or writes the first message.
DEFAULT_CHAT_TITLE = "New chat"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    chat_id         TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    ordering        INTEGER NOT NULL,
    context_report  TEXT,
    provider        TEXT,
    model           TEXT
);

CREATE TABLE IF NOT EXISTS attachments (
    id              TEXT PRIMARY KEY,
    chat_id         TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    stored_path     TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    extension       TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    text_status     TEXT NOT NULL,
    parsed_text     TEXT,
    char_count      INTEGER NOT NULL DEFAULT 0,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    parse_error     TEXT,
    created_at      TEXT NOT NULL,
    ordering        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_id, ordering);
CREATE INDEX IF NOT EXISTS attachments_by_chat ON attachments(chat_id, ordering);
"""

#: Columns RAG-016 adds to ``messages``, and what they hold. Deliberately nullable: a message
#: written before this stage has no provider, and must keep loading exactly as it was.
_MESSAGES_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("provider", "TEXT"),
    ("model", "TEXT"),
)

#: The prefix of a pre-migration backup file (RAG-016 §2).
BACKUP_PREFIX = "chat-before-rag016"

#: The subdirectory of the database's own directory that holds those backups.
BACKUP_DIRECTORY_NAME = "backups"


@dataclass(frozen=True)
class ChatMigration:
    """What one startup had to change in an existing database.

    Reported by the status endpoint and by the RAG-016 report; ``applied`` is ``False`` when
    the database was already current, which is the normal case after the first start.
    """

    applied: bool
    columns: tuple[str, ...] = ()
    backup_path: str | None = None


class ChatMigrationError(RuntimeError):
    """The database could not be brought up to date, so startup must stop.

    The original file is untouched (the ``ALTER TABLE`` runs in a transaction that is rolled
    back) and a backup exists, so nothing the user wrote is lost either way.
    """

    def __init__(self, detail: str, *, backup_path: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.backup_path = backup_path


def new_id() -> str:
    """A random identifier for a chat, message or attachment."""
    return uuid.uuid4().hex


class ChatStore:
    """SQLite-backed conversations."""

    def __init__(self, path: str | Path, *, backup_directory: str | Path | None = None) -> None:
        self.path = str(path)
        self._backup_directory = str(backup_directory) if backup_directory else None
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
            #: Which additive migration this database needed, if any (RAG-016 §2).
            self.migration = self._migrate()

    # --- schema ---------------------------------------------------------------
    def _columns(self, table: str) -> set[str]:
        """The column names of a table, as SQLite currently has them."""
        rows = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _migrate(self) -> ChatMigration:
        """Add RAG-016's message columns to an older database, after backing it up.

        Raises:
            ChatMigrationError: when the columns cannot be added. The transaction is rolled
                back, so the database keeps exactly the schema it had before this ran.
        """
        existing = self._columns("messages")
        missing = tuple(
            (name, declaration)
            for name, declaration in _MESSAGES_ADDED_COLUMNS
            if name not in existing
        )
        if not missing:
            return ChatMigration(applied=False)

        backup_path = self._backup("before-rag016")
        try:
            for name, declaration in missing:
                self._connection.execute(
                    f"ALTER TABLE messages ADD COLUMN {name} {declaration}"
                )
            self._connection.commit()
        except Exception as exc:
            self._connection.rollback()
            raise ChatMigrationError(
                "the chat database could not be extended with the RAG-016 message columns "
                f"({type(exc).__name__}: {exc}). The database was left as it was"
                + (f" and a copy is at {backup_path}" if backup_path else "")
                + ".",
                backup_path=backup_path,
            ) from exc
        return ChatMigration(
            applied=True,
            columns=tuple(name for name, _ in missing),
            backup_path=backup_path,
        )

    def _backup(self, label: str) -> str | None:
        """Copy the database file before it is altered, and return the copy's path.

        Uses SQLite's own backup API rather than a file copy, so the snapshot is consistent
        whatever journal mode the file is in. A database that does not exist yet (or an
        in-memory one) needs no copy: there is nothing to lose.
        """
        if self.path == ":memory:":
            return None
        source = Path(self.path)
        if not source.exists() or source.stat().st_size == 0:
            return None
        directory = (
            Path(self._backup_directory)
            if self._backup_directory
            else source.parent / BACKUP_DIRECTORY_NAME
        )
        directory.mkdir(parents=True, exist_ok=True)
        stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        target = directory / f"{BACKUP_PREFIX}-{stamp}.sqlite3"
        suffix = 2
        while target.exists():
            target = directory / f"{BACKUP_PREFIX}-{stamp}-{suffix}.sqlite3"
            suffix += 1
        destination = sqlite3.connect(str(target))
        try:
            with self._lock:
                self._connection.backup(destination)
        finally:
            destination.close()
        return str(target)

    # --- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        """Close the connection (the store can be reopened on the same file)."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ChatStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- chats ----------------------------------------------------------------
    def create_chat(self, title: str | None = None) -> Chat:
        """Create an empty conversation and return it."""
        chat_id = new_id()
        timestamp = utc_now()
        clean_title = (title or "").strip() or DEFAULT_CHAT_TITLE
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, clean_title, timestamp, timestamp),
            )
        return Chat(id=chat_id, title=clean_title, created_at=timestamp, updated_at=timestamp)

    def list_chats(self) -> tuple[ChatSummary, ...]:
        """Every chat, most recently updated first."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS messages,
                       (SELECT COUNT(*) FROM attachments a WHERE a.chat_id = c.id) AS attachments
                FROM chats c
                ORDER BY c.updated_at DESC, c.rowid DESC
                """
            ).fetchall()
        return tuple(
            ChatSummary(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["messages"],
                attachment_count=row["attachments"],
            )
            for row in rows
        )

    def get_chat(self, chat_id: str) -> Chat:
        """One chat with its messages and attachment metadata.

        Raises:
            ChatNotFoundError: if there is no such chat.
        """
        with self._lock:
            row = self._connection.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
            if row is None:
                raise ChatNotFoundError(chat_id)
            messages = tuple(
                _message_from_row(item)
                for item in self._connection.execute(
                    "SELECT * FROM messages WHERE chat_id = ? ORDER BY ordering, id", (chat_id,)
                ).fetchall()
            )
            attachments = tuple(
                _attachment_from_row(item)
                for item in self._connection.execute(
                    "SELECT * FROM attachments WHERE chat_id = ? ORDER BY ordering, id",
                    (chat_id,),
                ).fetchall()
            )
        return Chat(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=messages,
            attachments=attachments,
        )

    def chat_exists(self, chat_id: str) -> bool:
        """Whether the chat is stored."""
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
        return row is not None

    def rename_chat(self, chat_id: str, title: str) -> Chat:
        """Set a chat's title explicitly (an empty title is refused by the API)."""
        clean = (title or "").strip()
        if not clean:
            raise ValueError("a chat title cannot be empty")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (clean, utc_now(), chat_id),
            )
            if cursor.rowcount == 0:
                raise ChatNotFoundError(chat_id)
        return self.get_chat(chat_id)

    def name_chat_from_message(self, chat_id: str, message: str, *, limit: int = 60) -> None:
        """Give a still-unnamed chat a title taken from its first user message."""
        title = _title_from(message, limit)
        if not title:
            return
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND title = ?",
                (title, utc_now(), chat_id, DEFAULT_CHAT_TITLE),
            )

    def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and everything stored with it."""
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            if cursor.rowcount == 0:
                raise ChatNotFoundError(chat_id)

    def touch_chat(self, chat_id: str) -> None:
        """Mark a chat as just used, so the history list stays in use order."""
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (utc_now(), chat_id)
            )

    # --- messages -------------------------------------------------------------
    def add_message(
        self,
        chat_id: str,
        role: ChatRole | str,
        content: str,
        *,
        context_report: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ChatMessage:
        """Append one message to a chat and return it.

        ``provider`` and ``model`` are the RAG-016 metadata of the answer (which model wrote
        it). They stay ``None`` for user messages and for anything written before this stage.
        """
        role_value = ChatRole(role).value
        timestamp = utc_now()
        message_id = new_id()
        with self._lock, self._connection:
            if not self.chat_exists(chat_id):
                raise ChatNotFoundError(chat_id)
            row = self._connection.execute(
                "SELECT COALESCE(MAX(ordering), 0) AS last FROM messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            ordering = int(row["last"]) + 1
            self._connection.execute(
                "INSERT INTO messages (id, chat_id, role, content, created_at, ordering, "
                "context_report, provider, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    chat_id,
                    role_value,
                    content,
                    timestamp,
                    ordering,
                    json.dumps(context_report) if context_report else None,
                    provider,
                    model,
                ),
            )
            self._connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (timestamp, chat_id)
            )
        return ChatMessage(
            id=message_id,
            chat_id=chat_id,
            role=ChatRole(role_value),
            content=content,
            created_at=timestamp,
            ordering=ordering,
            context_report=context_report,
            provider=provider,
            model=model,
        )

    def list_messages(self, chat_id: str, *, limit: int | None = None) -> tuple[ChatMessage, ...]:
        """A chat's messages in order, optionally only the last ``limit`` of them."""
        with self._lock:
            if not self.chat_exists(chat_id):
                raise ChatNotFoundError(chat_id)
            if limit is None:
                rows = self._connection.execute(
                    "SELECT * FROM messages WHERE chat_id = ? ORDER BY ordering, id", (chat_id,)
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE chat_id = ? "
                    "ORDER BY ordering DESC, id DESC LIMIT ?) ORDER BY ordering, id",
                    (chat_id, int(limit)),
                ).fetchall()
        return tuple(_message_from_row(row) for row in rows)

    # --- attachments ----------------------------------------------------------
    def add_attachment(
        self,
        chat_id: str,
        *,
        attachment_id: str,
        filename: str,
        stored_path: str,
        content_type: str,
        extension: str,
        size_bytes: int,
        text_status: AttachmentTextStatus | str,
        parsed_text: str | None,
        char_count: int,
        token_estimate: int,
        parse_error: str | None = None,
    ) -> ChatAttachment:
        """Record one stored attachment (the file itself is written by the caller)."""
        timestamp = utc_now()
        with self._lock, self._connection:
            if not self.chat_exists(chat_id):
                raise ChatNotFoundError(chat_id)
            row = self._connection.execute(
                "SELECT COALESCE(MAX(ordering), 0) AS last FROM attachments WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            ordering = int(row["last"]) + 1
            self._connection.execute(
                "INSERT INTO attachments (id, chat_id, filename, stored_path, content_type, "
                "extension, size_bytes, text_status, parsed_text, char_count, token_estimate, "
                "parse_error, created_at, ordering) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attachment_id,
                    chat_id,
                    filename,
                    stored_path,
                    content_type,
                    extension,
                    int(size_bytes),
                    AttachmentTextStatus(text_status).value,
                    parsed_text,
                    int(char_count),
                    int(token_estimate),
                    parse_error,
                    timestamp,
                    ordering,
                ),
            )
            self._connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (timestamp, chat_id)
            )
        return ChatAttachment(
            id=attachment_id,
            chat_id=chat_id,
            filename=filename,
            stored_path=stored_path,
            content_type=content_type,
            extension=extension,
            size_bytes=int(size_bytes),
            text_status=AttachmentTextStatus(text_status),
            char_count=int(char_count),
            token_estimate=int(token_estimate),
            parse_error=parse_error,
            created_at=timestamp,
            ordering=ordering,
        )

    def list_attachments(self, chat_id: str) -> tuple[ChatAttachment, ...]:
        """A chat's attachments in the order they were attached."""
        with self._lock:
            if not self.chat_exists(chat_id):
                raise ChatNotFoundError(chat_id)
            rows = self._connection.execute(
                "SELECT * FROM attachments WHERE chat_id = ? ORDER BY ordering, id", (chat_id,)
            ).fetchall()
        return tuple(_attachment_from_row(row) for row in rows)

    def get_attachment(self, chat_id: str, attachment_id: str) -> ChatAttachment | None:
        """One attachment, or ``None`` when it is not part of that chat."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM attachments WHERE id = ? AND chat_id = ?",
                (attachment_id, chat_id),
            ).fetchone()
        return _attachment_from_row(row) if row is not None else None

    def attachment_text(self, chat_id: str, attachment_id: str) -> str:
        """The stored parsed text of one attachment (``""`` when there is none)."""
        with self._lock:
            row = self._connection.execute(
                "SELECT parsed_text FROM attachments WHERE id = ? AND chat_id = ?",
                (attachment_id, chat_id),
            ).fetchone()
        if row is None or row["parsed_text"] is None:
            return ""
        return str(row["parsed_text"])

    def delete_attachment(self, chat_id: str, attachment_id: str) -> ChatAttachment | None:
        """Delete one attachment's record, returning it so the file can be removed too."""
        attachment = self.get_attachment(chat_id, attachment_id)
        if attachment is None:
            return None
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM attachments WHERE id = ? AND chat_id = ?",
                (attachment_id, chat_id),
            )
            self._connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (utc_now(), chat_id)
            )
        return attachment


def _title_from(message: str, limit: int) -> str:
    """A short, single-line title from the first user message."""
    collapsed = " ".join((message or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= limit:
        return collapsed
    clipped = collapsed[:limit].rstrip()
    # Prefer a word boundary, but never return an empty title.
    return (clipped.rsplit(" ", 1)[0] or clipped) + "…"


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    raw_report = row["context_report"]
    report: dict[str, Any] | None = None
    if raw_report:
        try:
            loaded = json.loads(raw_report)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            loaded = None
        if isinstance(loaded, dict):
            report = loaded
    return ChatMessage(
        id=row["id"],
        chat_id=row["chat_id"],
        role=ChatRole(row["role"]),
        content=row["content"],
        created_at=row["created_at"],
        ordering=int(row["ordering"]),
        context_report=report,
        # A message written before RAG-016 simply has no provider: the columns are nullable, and
        # reading them back as ``None`` is what keeps an existing chat loading as before (§7).
        provider=_optional_text(row, "provider"),
        model=_optional_text(row, "model"),
    )


def _optional_text(row: sqlite3.Row, column: str) -> str | None:
    """A nullable text column, as a non-empty string or ``None``."""
    value = row[column]
    if value is None:
        return None
    text = str(value)
    return text or None


def _attachment_from_row(row: sqlite3.Row) -> ChatAttachment:
    return ChatAttachment(
        id=row["id"],
        chat_id=row["chat_id"],
        filename=row["filename"],
        stored_path=row["stored_path"],
        content_type=row["content_type"],
        extension=row["extension"],
        size_bytes=int(row["size_bytes"]),
        text_status=AttachmentTextStatus(row["text_status"]),
        char_count=int(row["char_count"]),
        token_estimate=int(row["token_estimate"]),
        parse_error=row["parse_error"],
        created_at=row["created_at"],
        ordering=int(row["ordering"]),
    )


__all__ = [
    "BACKUP_DIRECTORY_NAME",
    "BACKUP_PREFIX",
    "ChatMigration",
    "ChatMigrationError",
    "ChatStore",
    "DEFAULT_CHAT_TITLE",
    "new_id",
]
