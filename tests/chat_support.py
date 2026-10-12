"""Support for the RAG-015 simple-chat tests.

The chat is tested against the project's existing doubles: :class:`~tests.generation_support.FakeGenerationModel`
replaces Ollama (so no model is loaded and every prompt is inspectable), and RAG-009
``RetrievalResponse`` objects built by the generation helpers replace real retrieval. Settings
point at a temporary chat database and attachment directory, so no test touches a real
conversation.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Sequence

from fastapi.testclient import TestClient

from app.chat.service import ChatService
from app.config import Settings
from app.generation.models import ModelFailure
from app.generation.ollama import ChatResult
from app.main import create_app
from app.retrieval.models import Candidate, RetrievalResponse
from app.web.service import ChatRunner, ModeRunner, make_corpus_provider
from tests.generation_support import FakeGenerationModel, candidate, retrieval_response


def chat_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings whose only storage is inside ``tmp_path``."""
    values: dict[str, Any] = {
        "chat_db_path": str(tmp_path / "chat.sqlite3"),
        "chat_attachment_dir": str(tmp_path / "chat_attachments"),
        "generation_chars_per_token": 4.0,
    }
    values.update(overrides)
    return Settings(**values)


class FakeCorpusRetrieval:
    """A corpus provider that records its queries and returns scripted candidates."""

    def __init__(self, results: Sequence[Candidate] = (), error: Exception | None = None) -> None:
        self.results = list(results)
        self.error = error
        self.queries: list[str] = []

    def __call__(self, query: str) -> RetrievalResponse:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return retrieval_response(self.results, query=query)


#: Ollama's repeat-guard message, byte-for-byte as the server returns it (RAG-015 FIX1).
REPEAT_ABORT_MESSAGE = '{"error":"prediction aborted, token repeat limit reached"}'


def repeat_abort_failure() -> ModelFailure:
    """The failure Ollama returns when its runner repeats one token forever."""
    return ModelFailure(
        kind="server_error",
        message=REPEAT_ABORT_MESSAGE,
        status_code=500,
        endpoint="http://127.0.0.1:11434/api/chat",
        retryable=True,
    )


class ScriptedModel:
    """A model that returns a scripted sequence of results, and records every call.

    Used to test the one-retry recovery: the first result can be the repeat abort and the
    second a real answer, so "exactly one retry, one assistant message" is checkable.
    """

    def __init__(self, results: Sequence[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.unloads = 0

    @property
    def name(self) -> str:
        return "fake:scripted"

    def chat(self, messages: Sequence[Any], *, options: Any = None, think: bool | None = None) -> Any:
        self.calls.append({"messages": list(messages), "options": options, "think": think})
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]

    def unload(self) -> bool:
        self.unloads += 1
        return True

    def describe(self) -> dict[str, Any]:
        return {"model": self.name}


def answer_result(content: str = "A real answer.") -> Any:
    """A successful :class:`ChatResult`."""
    return ChatResult(
        model="fake:scripted",
        content=content,
        done_reason="stop",
        prompt_tokens=100,
        generated_tokens=20,
        wall_seconds=0.5,
        model_details={"model": "fake:scripted"},
    )


def failure_result(failure: ModelFailure | None = None) -> Any:
    """A failed :class:`ChatResult` (the repeat abort unless another failure is given)."""
    return ChatResult(
        model="fake:scripted",
        content="",
        wall_seconds=0.5,
        failure=failure or repeat_abort_failure(),
    )


def build_chat(
    tmp_path: Path,
    *,
    model: Any | None = None,
    corpus: Any | None = None,
    **setting_overrides: Any,
) -> tuple[ChatService, Any, Settings]:
    """A chat service over a temporary database, a scripted model and an optional corpus."""
    settings = chat_settings(tmp_path, **setting_overrides)
    scripted = model or FakeGenerationModel(content="Answer from the conversation.")
    service = ChatService(
        settings=settings,
        model=scripted,
        corpus_retrieval=corpus,
    )
    return service, scripted, settings


def build_client(
    tmp_path: Path,
    *,
    model: FakeGenerationModel | None = None,
    corpus: Any | None = None,
    mode_harness: Any | None = None,
    **setting_overrides: Any,
) -> tuple[TestClient, ChatService, FakeGenerationModel, Settings]:
    """A web client whose chat runner owns a real chat service over temporary storage.

    ``mode_harness`` (from :mod:`tests.modes_support`) also wires the RAG-014 mode API to the
    RAG-013 fakes, so one client can exercise the simple chat and Advanced together — and the
    chat's corpus retrieval then runs through the *same* mode runner, exactly as it does in the
    application. Without a harness the chat gets an offline corpus double, so no test ever
    loads BGE-M3 or opens a Qdrant store.
    """
    settings = chat_settings(tmp_path, **setting_overrides)
    scripted = model or FakeGenerationModel(content="Answer from the conversation.")
    if mode_harness is None:
        mode_runner = ModeRunner(settings=settings)
        provider = corpus if corpus is not None else FakeCorpusRetrieval()
    else:
        mode_runner = ModeRunner(
            settings=settings, service_factory=lambda *_, **__: mode_harness.service
        )
        provider = corpus if corpus is not None else make_corpus_provider(mode_runner)
    service = ChatService(settings=settings, model=scripted, corpus_retrieval=provider)
    runner = ChatRunner(settings=settings, service_factory=lambda **_: service)
    application = create_app(
        settings=settings, mode_runner=mode_runner, chat_runner=runner
    )
    return TestClient(application), service, scripted, settings


def attachment_body(filename: str, data: bytes) -> dict[str, str]:
    """A ready-to-post attachment body (RAG-015 §13: base64 in JSON, no multipart)."""
    return {"filename": filename, "content_base64": base64.b64encode(data).decode("ascii")}


def docx_bytes(paragraphs: Sequence[str], *, heading: str | None = None) -> bytes:
    """A real DOCX in memory, so the DOCX parser is exercised for real."""
    import io

    from docx import Document as DocxDocument

    document = DocxDocument()
    if heading:
        document.add_heading(heading, level=1)
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def markdown_bytes(text: str) -> bytes:
    """A markdown file as bytes."""
    return text.encode("utf-8")


def text_bytes(text: str) -> bytes:
    """A plain-text file as bytes."""
    return text.encode("utf-8")


def persisted_messages(service: ChatService, chat_id: str) -> list[tuple[str, str]]:
    """``(role, content)`` for every stored message in a chat."""
    return [
        (message.role.value, message.content)
        for message in service.store.list_messages(chat_id)
    ]


def last_messages(model: FakeGenerationModel) -> list[tuple[str, str]]:
    """``(role, content)`` for the messages of the most recent model call."""
    if not model.calls:
        return []
    return [(item.role, item.content) for item in model.calls[-1]["messages"]]


__all__ = [
    "FakeCorpusRetrieval",
    "REPEAT_ABORT_MESSAGE",
    "ScriptedModel",
    "answer_result",
    "attachment_body",
    "build_chat",
    "build_client",
    "candidate",
    "chat_settings",
    "docx_bytes",
    "failure_result",
    "last_messages",
    "markdown_bytes",
    "persisted_messages",
    "repeat_abort_failure",
    "retrieval_response",
    "text_bytes",
]
