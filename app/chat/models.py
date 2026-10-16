"""Data model and errors for the simple persistent chat (RAG-015 §4, §5, §15).

Everything a saved conversation is made of: the chat, its messages in order, the working
documents attached to it, and the report that says how the model's context window was spent.
The models are pydantic so the API can serialise them without a second schema, and the store
returns exactly these shapes, so a chat reopened after a restart is the same object as a
live one.

The failure types live here too, one per condition the API has to turn into a readable
sentence: an unknown chat, an attachment the chat cannot accept, and a request whose
mandatory parts do not fit the context window.
"""

from __future__ import annotations

import datetime as _datetime
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict


def utc_now() -> str:
    """The timestamp format every chat record uses (RFC 3339, UTC, microseconds).

    Microseconds matter: chats created in the same second must still order predictably, and
    the saved-chat list is ordered by when a chat was last used.
    """
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="microseconds")


class ChatRole(str, Enum):
    """Who wrote a stored message. Only these two are ever stored."""

    USER = "user"
    ASSISTANT = "assistant"


class AttachmentTextStatus(str, Enum):
    """What happened when an attachment's text was extracted."""

    #: Text was extracted and is stored with the attachment.
    PARSED = "parsed"
    #: The parser succeeded but found no text (for example a scanned PDF without OCR).
    EMPTY = "empty"
    #: The file was stored, but its text could not be extracted.
    FAILED = "failed"


class ChatSummary(BaseModel):
    """One row of the saved-chat list (no messages, no attachments)."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    attachment_count: int = 0


class ChatMessage(BaseModel):
    """One stored turn."""

    model_config = ConfigDict(frozen=True)

    id: str
    chat_id: str
    role: ChatRole
    content: str
    created_at: str
    #: Order inside a chat, starting at 1.
    ordering: int
    #: How the request that produced this assistant message spent the context window (§15).
    #: ``None`` for user messages.
    context_report: dict[str, Any] | None = None
    #: Which provider answered this turn (RAG-016 §7): ``"ollama"`` or ``"deepseek"``.
    #: ``None`` for user messages and for turns stored before RAG-016, which keep loading.
    provider: str | None = None
    #: The model id that answered, as the provider reported it.
    model: str | None = None


class ChatAttachment(BaseModel):
    """One working document attached to a chat.

    ``stored_path`` is the path this application chose (a chat id and an attachment id under
    the attachment directory); ``filename`` is what the user called the file, kept only to
    display it and never used to build a path.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    chat_id: str
    filename: str
    stored_path: str
    content_type: str
    extension: str
    size_bytes: int
    text_status: AttachmentTextStatus
    char_count: int = 0
    token_estimate: int = 0
    parse_error: str | None = None
    created_at: str
    ordering: int


class Chat(BaseModel):
    """A conversation with its messages and attachments."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: tuple[ChatMessage, ...] = ()
    attachments: tuple[ChatAttachment, ...] = ()


class OmittedContextItem(BaseModel):
    """Material that was left out because the context window is finite (§6)."""

    model_config = ConfigDict(frozen=True)

    #: ``history``, ``attachment`` or ``corpus``.
    kind: str
    label: str
    reason: str
    detail: str = ""
    tokens: int = 0


class AttachmentUsage(BaseModel):
    """How much of one attached document reached the model (§15)."""

    model_config = ConfigDict(frozen=True)

    attachment_id: str
    filename: str
    tokens: int = 0
    included_chars: int = 0
    total_chars: int = 0
    #: Whether the document was cut short by the budget (always at its end, never mid-way).
    truncated: bool = False


class ContextReport(BaseModel):
    """How the context window was spent on one chat request (§15).

    Debug material: it is returned with the answer and stored with the assistant message so
    the Advanced panel can show it, but the normal chat interface does not display it.
    """

    model_config = ConfigDict(frozen=True)

    num_ctx: int
    input_budget_tokens: int
    output_headroom_tokens: int
    system_tokens: int = 0
    history_tokens: int = 0
    attachment_tokens: int = 0
    corpus_tokens: int = 0
    current_message_tokens: int = 0
    total_tokens: int = 0
    history_messages_included: int = 0
    history_messages_omitted: int = 0
    corpus_requested: bool = False
    corpus_invoked: bool = False
    corpus_sources: int = 0
    #: How many extra generation attempts Ollama's repeat guard forced (RAG-015 FIX1).
    #: ``0`` is the normal case; ``1`` means the turn succeeded (or failed) after one retry.
    generation_retries: int = 0
    #: Why the retry happened, in Ollama's own words.
    retry_reason: str = ""
    # --- which provider answered (RAG-016 §19) ---
    #: ``"ollama"`` or ``"deepseek"``.
    provider: str = ""
    #: The model id the answer came from, as the provider reported it.
    model: str = ""
    #: Token counts the *provider* reported for this turn. ``None`` means the provider said
    #: nothing; no number is ever estimated into these fields.
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    #: How long the provider took to answer.
    latency_seconds: float | None = None
    #: The provider's request id, for support and debugging.
    request_id: str = ""
    attachments: tuple[AttachmentUsage, ...] = ()
    omitted: tuple[OmittedContextItem, ...] = ()

    @property
    def has_omissions(self) -> bool:
        """Whether anything was left out for want of context."""
        return bool(self.omitted)


class ChatAnswer(BaseModel):
    """The outcome of one user turn.

    ``failure`` is set instead of ``assistant_message`` when the model could not answer: the
    user message is still saved, so nothing the user typed is ever lost.
    """

    model_config = ConfigDict(frozen=True)

    chat_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage | None = None
    report: ContextReport | None = None
    failure: dict[str, Any] | None = None
    #: Which provider answered (RAG-016 §7). Empty when the turn failed before a provider was
    #: reached, which only happens for a misconfiguration.
    provider: str = ""
    model: str = ""
    wall_seconds: float = 0.0
    prompt_tokens: int | None = None
    generated_tokens: int | None = None


class ChatError(Exception):
    """Base class for a chat failure the API turns into a readable error."""

    status_code = 400
    kind = "chat_error"
    title = "The chat request failed"

    def __init__(self, detail: str, *, title: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if title is not None:
            self.title = title


class ChatNotFoundError(ChatError):
    """The chat id does not exist (or was deleted)."""

    status_code = 404
    kind = "chat_not_found"
    title = "That chat does not exist"

    def __init__(self, chat_id: str) -> None:
        super().__init__(
            f"No saved chat has the id {chat_id}. It may have been deleted; start a new chat."
        )
        self.chat_id = chat_id


class AttachmentError(ChatError):
    """The attachment was rejected; the message says what to do instead."""

    status_code = 400
    kind = "attachment_rejected"
    title = "That file cannot be attached"


class UnsupportedAttachmentError(AttachmentError):
    """The extension is not one the chat can read."""

    kind = "unsupported_attachment"


class AttachmentTooLargeError(AttachmentError):
    """The file is larger than the configured limit."""

    status_code = 413
    kind = "attachment_too_large"
    title = "That file is too large to attach"


class AttachmentLimitError(AttachmentError):
    """The chat already holds as many attachments as it may."""

    kind = "attachment_limit"


class EmptyMessageError(ChatError):
    """The message was empty after trimming."""

    kind = "empty_message"
    title = "There is nothing to send"


class NothingToRetryError(ChatError):
    """There is no unanswered user turn to answer again (RAG-015 FIX1 §6)."""

    status_code = 409
    kind = "nothing_to_retry"
    title = "There is no unanswered message"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class UnsupportedProviderError(ChatError):
    """The requested provider is not one this build can answer with (RAG-016 §6)."""

    kind = "unsupported_provider"
    title = "That model choice is not available"

    def __init__(self, requested: str) -> None:
        super().__init__(
            f"'{requested}' is not a model this application can use. Choose the local model "
            "or DeepSeek."
        )
        self.requested = requested


class UnsupportedModelError(ChatError):
    """The requested model is not one the provider offers (RAG-016 FIX1)."""

    kind = "unsupported_model"
    title = "That model is not available"

    def __init__(self, provider: str, requested: str, offered: Sequence[str] = ()) -> None:
        choices = ", ".join(offered) if offered else "the configured models"
        super().__init__(
            f"'{requested}' is not a {provider} model this application offers. Select one of: "
            f"{choices}."
        )
        self.provider = provider
        self.requested = requested
        self.offered = tuple(offered)


class ProviderNotConfiguredError(ChatError):
    """The provider was selected but is unusable on this server (RAG-016 §4)."""

    status_code = 503
    kind = "provider_not_configured"
    title = "That model is not configured"

    def __init__(self, provider: str, detail: str) -> None:
        super().__init__(detail)
        self.provider = provider


class MessageTooLargeError(ChatError):
    """The mandatory part of the request does not fit the context window."""

    status_code = 413
    kind = "message_too_large"
    title = "That message does not fit the context window"


__all__ = [
    "AttachmentError",
    "AttachmentLimitError",
    "AttachmentTextStatus",
    "AttachmentTooLargeError",
    "AttachmentUsage",
    "Chat",
    "ChatAnswer",
    "ChatAttachment",
    "ChatError",
    "ChatMessage",
    "ChatNotFoundError",
    "ChatRole",
    "ChatSummary",
    "ContextReport",
    "EmptyMessageError",
    "MessageTooLargeError",
    "NothingToRetryError",
    "OmittedContextItem",
    "ProviderNotConfiguredError",
    "UnsupportedAttachmentError",
    "UnsupportedModelError",
    "UnsupportedProviderError",
    "utc_now",
]
