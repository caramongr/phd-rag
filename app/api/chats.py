"""The simple-chat API (RAG-015 §13).

Nine endpoints and no chat logic: every one delegates to
:class:`~app.chat.service.ChatService`, exactly as the RAG-014 mode API delegates to the mode
service. Nothing here touches the existing ``/api/mode`` endpoints, which keep working
unchanged.

Uploads are JSON with base64 content (``{"filename": ..., "content_base64": ...}``) rather than
multipart form data. That is a deliberate choice: multipart file uploads need an extra
dependency, base64 needs none, and the payload is the same bytes either way. Every failure
leaves as the same ``{"error": {...}}`` envelope the rest of the API uses, so the frontend has
one error path.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.chat.models import (
    AttachmentError,
    Chat,
    ChatAnswer,
    ChatAttachment,
    ChatError,
    ChatMessage,
    ChatSummary,
    ContextReport,
)
from app.chat.service import ChatService
from app.web.service import ChatRunner

router = APIRouter(prefix="/api", tags=["chat"])

#: Transport-sanity bound on a message body. The real limit is the configured
#: ``chat_max_message_chars``, checked by the service, which answers with a readable 413.
MAX_MESSAGE_CHARS = 50_000_000
#: Transport-sanity bound on an attachment body before it is decoded.
MAX_ATTACHMENT_CHARS = 200_000_000


# --- request bodies -----------------------------------------------------------


class NewChatRequest(BaseModel):
    """``{"title": "..."}`` — a title is optional; a chat may be named later."""

    model_config = {"extra": "forbid"}

    title: str | None = Field(default=None, max_length=200)


class RenameChatRequest(BaseModel):
    """``{"title": "..."}`` — renaming is explicit and never trims to nothing."""

    model_config = {"extra": "forbid"}

    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("a chat title cannot be empty")
        return stripped


class MessageRequest(BaseModel):
    """``{"message": "...", "consult_corpus": null}`` (§7, §8)."""

    model_config = {"extra": "forbid"}

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    #: ``True`` consults the permanent corpus, ``False`` never does, ``null`` decides from
    #: the message's own wording.
    consult_corpus: bool | None = None

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the message is empty")
        return value


class AttachmentRequest(BaseModel):
    """``{"filename": "chapter.docx", "content_base64": "..."}`` (§5)."""

    model_config = {"extra": "forbid"}

    filename: str = Field(min_length=1, max_length=500)
    content_base64: str = Field(min_length=1, max_length=MAX_ATTACHMENT_CHARS)


# --- serialisation ------------------------------------------------------------


def message_json(message: ChatMessage) -> dict[str, Any]:
    """One stored message, as the UI renders it."""
    return {
        "id": message.id,
        "chat_id": message.chat_id,
        "role": message.role.value,
        "content": message.content,
        "created_at": message.created_at,
        "ordering": message.ordering,
        "context_report": message.context_report,
    }


def attachment_json(attachment: ChatAttachment) -> dict[str, Any]:
    """One attachment's metadata, as the UI renders it (never a filesystem path)."""
    return {
        "id": attachment.id,
        "chat_id": attachment.chat_id,
        "filename": attachment.filename,
        "content_type": attachment.content_type,
        "extension": attachment.extension,
        "size_bytes": attachment.size_bytes,
        "text_status": attachment.text_status.value,
        "char_count": attachment.char_count,
        "token_estimate": attachment.token_estimate,
        "parse_error": attachment.parse_error,
        "created_at": attachment.created_at,
    }


def summary_json(summary: ChatSummary) -> dict[str, Any]:
    """One row of the saved-chat list."""
    return {
        "id": summary.id,
        "title": summary.title,
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "message_count": summary.message_count,
        "attachment_count": summary.attachment_count,
    }


def chat_json(chat: Chat, *, with_messages: bool = True) -> dict[str, Any]:
    """A chat with its messages and attachments."""
    payload: dict[str, Any] = {
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "attachments": [attachment_json(item) for item in chat.attachments],
        "message_count": len(chat.messages),
    }
    if with_messages:
        payload["messages"] = [message_json(item) for item in chat.messages]
    return payload


def report_json(report: ContextReport | None) -> dict[str, Any] | None:
    """The context summary (§15). Advanced and debug material, never the main interface."""
    if report is None:
        return None
    return {
        "num_ctx": report.num_ctx,
        "input_budget_tokens": report.input_budget_tokens,
        "output_headroom_tokens": report.output_headroom_tokens,
        "system_tokens": report.system_tokens,
        "history_tokens": report.history_tokens,
        "attachment_tokens": report.attachment_tokens,
        "corpus_tokens": report.corpus_tokens,
        "current_message_tokens": report.current_message_tokens,
        "total_tokens": report.total_tokens,
        "history_messages_included": report.history_messages_included,
        "history_messages_omitted": report.history_messages_omitted,
        "corpus_requested": report.corpus_requested,
        "corpus_invoked": report.corpus_invoked,
        "corpus_sources": report.corpus_sources,
        "attachments": [usage.model_dump(mode="json") for usage in report.attachments],
        "omitted": [item.model_dump(mode="json") for item in report.omitted],
        "generation_retries": report.generation_retries,
        "retry_reason": report.retry_reason,
    }


def answer_json(answer: ChatAnswer) -> dict[str, Any]:
    """One answered turn: the two messages, the model's own numbers, and the context report."""
    report = report_json(answer.report)
    return {
        "chat_id": answer.chat_id,
        "user_message": message_json(answer.user_message),
        "assistant_message": (
            message_json(answer.assistant_message) if answer.assistant_message else None
        ),
        "failure": answer.failure,
        "retried": bool(report and report["generation_retries"]),
        "model": answer.model,
        "wall_seconds": answer.wall_seconds,
        "prompt_tokens": answer.prompt_tokens,
        "generated_tokens": answer.generated_tokens,
        "context": report,
    }


# --- dependency ---------------------------------------------------------------


def get_chat_runner(request: Request) -> ChatRunner:
    """The app's chat runner, created with the application."""
    runner = getattr(request.app.state, "chat_runner", None)
    if runner is None:  # pragma: no cover - defensive: create_app always sets it
        raise ChatError("the chat service is not configured")
    return runner


def _service(runner: ChatRunner) -> ChatService:
    return runner.service()


# --- endpoints ----------------------------------------------------------------


@router.get("/chat/info")
def chat_info(runner: ChatRunner = Depends(get_chat_runner)) -> dict[str, Any]:
    """What the composer needs to know: model, window, accepted formats and limits."""
    service = _service(runner)
    settings = service.settings
    return {
        "model": service.model_name,
        "num_ctx": settings.chat_num_ctx,
        "context_tokens": settings.chat_context_tokens,
        "max_output_tokens": settings.chat_max_output_tokens,
        "max_message_chars": settings.chat_max_message_chars,
        "accepted_formats": list(settings.chat_allowed_extensions),
        "max_attachment_bytes": settings.chat_max_attachment_bytes,
        "max_attachments_per_chat": settings.chat_max_attachments_per_chat,
        "corpus_retrieval_available": service.corpus_available,
        "prompt_version": service.prompt_version,
    }


@router.get("/chats")
def list_chats(runner: ChatRunner = Depends(get_chat_runner)) -> dict[str, Any]:
    """The saved conversations, most recently used first."""
    return {"chats": [summary_json(item) for item in _service(runner).list_chats()]}


@router.post("/chats", status_code=201)
def create_chat(
    body: NewChatRequest | None = None,
    runner: ChatRunner = Depends(get_chat_runner),
) -> dict[str, Any]:
    """Start a new, empty conversation."""
    chat = _service(runner).new_chat(body.title if body else None)
    return chat_json(chat)


@router.get("/chats/{chat_id}")
def get_chat(chat_id: str, runner: ChatRunner = Depends(get_chat_runner)) -> dict[str, Any]:
    """Reopen a conversation: its messages and its attachments."""
    return chat_json(_service(runner).get_chat(chat_id))


@router.patch("/chats/{chat_id}")
def rename_chat(
    chat_id: str,
    body: RenameChatRequest,
    runner: ChatRunner = Depends(get_chat_runner),
) -> dict[str, Any]:
    """Rename a conversation."""
    return chat_json(_service(runner).rename_chat(chat_id, body.title), with_messages=False)


@router.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, runner: ChatRunner = Depends(get_chat_runner)) -> dict[str, Any]:
    """Delete a conversation, its messages and its stored attachments."""
    _service(runner).delete_chat(chat_id)
    return {"deleted": True, "id": chat_id}


@router.post("/chats/{chat_id}/messages")
def send_message(
    chat_id: str,
    body: MessageRequest,
    runner: ChatRunner = Depends(get_chat_runner),
) -> dict[str, Any]:
    """Answer one message in a conversation, saving both turns."""
    answer = _service(runner).send_message(
        chat_id, body.message, consult_corpus=body.consult_corpus
    )
    return answer_json(answer)


@router.post("/chats/{chat_id}/retry")
def retry_last_message(
    chat_id: str,
    runner: ChatRunner = Depends(get_chat_runner),
) -> dict[str, Any]:
    """Answer the last user turn again, without adding a second copy of it (FIX1 §6).

    This is the recovery the UI offers after a failed generation: the conversation keeps
    exactly one user message, and the answer is written for that existing turn.
    """
    return answer_json(_service(runner).retry_last(chat_id))


@router.post("/chats/{chat_id}/attachments", status_code=201)
def add_attachment(
    chat_id: str,
    body: AttachmentRequest,
    runner: ChatRunner = Depends(get_chat_runner),
) -> dict[str, Any]:
    """Attach a working document to a conversation (never to the permanent corpus)."""
    try:
        data = base64.b64decode(body.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError(
            f"'{body.filename}' could not be read: the uploaded content is not valid base64 "
            f"({type(exc).__name__})."
        ) from exc
    attachment = _service(runner).add_attachment(chat_id, body.filename, data)
    return attachment_json(attachment)


@router.delete("/chats/{chat_id}/attachments/{attachment_id}")
def delete_attachment(
    chat_id: str,
    attachment_id: str,
    runner: ChatRunner = Depends(get_chat_runner),
) -> dict[str, Any]:
    """Detach a document from a conversation."""
    service = _service(runner)
    service.get_chat(chat_id)  # a missing chat is a 404, not a silent success
    removed = service.delete_attachment(chat_id, attachment_id)
    if removed is None:
        raise AttachmentError(
            f"No attachment {attachment_id} is attached to this chat. It may have been "
            "removed already."
        )
    return {"deleted": True, "id": attachment_id}


def install_chat_error_handlers(application: Any) -> None:
    """Turn chat failures into the same readable JSON envelope the rest of the API uses."""

    @application.exception_handler(ChatError)
    async def _chat_error(_: Request, exc: ChatError) -> JSONResponse:
        return JSONResponse(
            status_code=getattr(type(exc), "status_code", 400),
            content={
                "error": {
                    "kind": getattr(type(exc), "kind", "chat_error"),
                    "title": getattr(type(exc), "title", "The chat request failed"),
                    "detail": getattr(exc, "detail", str(exc)),
                }
            },
        )


__all__ = [
    "AttachmentRequest",
    "MessageRequest",
    "NewChatRequest",
    "RenameChatRequest",
    "answer_json",
    "attachment_json",
    "chat_json",
    "create_chat",
    "delete_attachment",
    "delete_chat",
    "get_chat",
    "get_chat_runner",
    "install_chat_error_handlers",
    "list_chats",
    "message_json",
    "rename_chat",
    "report_json",
    "retry_last_message",
    "router",
    "send_message",
    "summary_json",
]
