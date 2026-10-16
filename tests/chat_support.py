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
    cloud_model: Any | None = None,
    corpus: Any | None = None,
    **setting_overrides: Any,
) -> tuple[ChatService, Any, Settings]:
    """A chat service over a temporary database, a scripted model and an optional corpus."""
    settings = chat_settings(tmp_path, **setting_overrides)
    scripted = model or FakeGenerationModel(content="Answer from the conversation.")
    service = ChatService(
        settings=settings,
        model=scripted,
        cloud_model=cloud_model,
        corpus_retrieval=corpus,
    )
    return service, scripted, settings


def build_client(
    tmp_path: Path,
    *,
    model: FakeGenerationModel | None = None,
    cloud_model: Any | None = None,
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

    ``cloud_model`` replaces the DeepSeek client, so no test ever reaches the network: it is
    the only way a cloud turn is exercised, and a test asserts on it to prove which provider
    answered.
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
    service = ChatService(
        settings=settings,
        model=scripted,
        cloud_model=cloud_model,
        corpus_retrieval=provider,
    )
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


# --- the cloud provider double (RAG-016) -------------------------------------

#: A DeepSeek key that only ever exists in a test. Never a real one.
FAKE_DEEPSEEK_KEY = "sk-test-not-a-real-key"


class FakeCloudModel:
    """A scripted DeepSeek client: no network, no key, fully deterministic.

    It records its calls exactly like :class:`~tests.generation_support.FakeGenerationModel`
    does, so a test can assert *which* provider received a turn by looking at which double was
    called — which is how "no automatic failover" is checked (RAG-016 §8, §23).
    """

    def __init__(
        self,
        *,
        content: str = "Answer from the cloud.",
        failure: ModelFailure | None = None,
        name: str = "deepseek-flash",
        models: Sequence[str] = ("deepseek-flash", "deepseek-v4-pro"),
        prompt_tokens: int = 1200,
        generated_tokens: int = 80,
        cached_input_tokens: int | None = 900,
        reasoning_tokens: int | None = None,
        request_id: str = "req-fake-1",
        wall_seconds: float = 1.25,
        available_models_error: Exception | None = None,
    ) -> None:
        self.provider = "deepseek"
        self.name = name
        self.content = content
        self.failure = failure
        self.models = tuple(models)
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.cached_input_tokens = cached_input_tokens
        self.reasoning_tokens = reasoning_tokens
        self.request_id = request_id
        self.wall_seconds = wall_seconds
        self.available_models_error = available_models_error
        self.calls: list[dict[str, Any]] = []
        self.listed = 0

    def chat(
        self,
        messages: Sequence[Any],
        *,
        options: Any = None,
        think: bool | None = None,
    ) -> Any:
        """Record the call and return the scripted result."""
        self.calls.append(
            {"messages": list(messages), "options": options, "think": think, "model": self.name}
        )
        return ChatResult(
            model=self.name,
            content="" if self.failure else self.content,
            done_reason="stop",
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            cached_input_tokens=self.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens,
            request_id=self.request_id,
            wall_seconds=self.wall_seconds,
            model_details={"provider": "deepseek", "model": self.name},
            failure=self.failure,
            provider="deepseek",
        )

    def available_models(self, *, timeout: float | None = None) -> tuple[str, ...]:
        """The ids a real client would have read from ``/models``."""
        self.listed += 1
        if self.available_models_error is not None:
            raise self.available_models_error
        return self.models

    def use_model(self, model: str) -> None:
        """Point this client at another model id, as the real client does (RAG-016 FIX1)."""
        clean = str(model or "").strip()
        if clean:
            self.name = clean

    def models_requested(self) -> list[str]:
        """The model id each call was made with, in order."""
        return [str(call.get("model") or "") for call in self.calls]

    def describe(self) -> dict[str, Any]:
        return {"provider": "deepseek", "model": self.name}

    @property
    def last_prompt(self) -> str:
        """The last message of the most recent call."""
        if not self.calls:
            return ""
        return self.calls[-1]["messages"][-1].content


def cloud_failure(kind: str = "auth_error", message: str = "DeepSeek authentication failed.") -> Any:
    """A failed cloud :class:`ChatResult` with the provider's own classification."""
    return ModelFailure(kind=kind, message=message, status_code=401, retryable=False)


__all__ = [
    "FAKE_DEEPSEEK_KEY",
    "FakeCloudModel",
    "FakeCorpusRetrieval",
    "REPEAT_ABORT_MESSAGE",
    "ScriptedModel",
    "answer_result",
    "attachment_body",
    "build_chat",
    "build_client",
    "candidate",
    "chat_settings",
    "cloud_failure",
    "docx_bytes",
    "failure_result",
    "last_messages",
    "markdown_bytes",
    "persisted_messages",
    "repeat_abort_failure",
    "retrieval_response",
    "text_bytes",
]
