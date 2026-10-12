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
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
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
    context_report  TEXT
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


def new_id() -> str:
    """A random identifier for a chat, message or attachment."""
    return uuid.uuid4().hex


class ChatStore:
    """SQLite-backed conversations."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

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
    ) -> ChatMessage:
        """Append one message to a chat and return it."""
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
                "context_report) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    chat_id,
                    role_value,
                    content,
                    timestamp,
                    ordering,
                    json.dumps(context_report) if context_report else None,
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
    )


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


__all__ = ["DEFAULT_CHAT_TITLE", "ChatStore", "new_id"]
