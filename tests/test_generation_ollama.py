"""Generation-model tests (RAG-011 §31 items 6-10).

The Ollama client is exercised through an injected session, so the suite proves the
request shape and the failure mapping without a running server. Nothing here needs
Ollama or the Internet.
"""

from __future__ import annotations

import json
from typing import Any

from app.generation.models import ModelFailure
from app.generation.ollama import (
    ChatMessage,
    GenerationOptions,
    OllamaClient,
    model_summary,
)


class FakeResponse:
    """A minimal stand-in for a ``requests`` response."""

    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        *,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})
        self.headers = headers or {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeSession:
    """Records requests and replays scripted responses."""

    def __init__(
        self,
        *,
        post: list[FakeResponse | Exception] | None = None,
        get: dict[str, FakeResponse | Exception] | None = None,
    ) -> None:
        self._post = list(post or [])
        self._get = dict(get or {})
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        if not self._post:
            raise AssertionError(f"unexpected POST {url}")
        item = self._post.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.gets.append(url)
        item = self._get.get(url)
        if item is None:
            raise AssertionError(f"unexpected GET {url}")
        if isinstance(item, Exception):
            raise item
        return item


def chat_payload(content: str = "An answer [S1].", **overrides: Any) -> dict[str, Any]:
    """A normal Ollama ``/api/chat`` body."""
    payload: dict[str, Any] = {
        "model": "qwen3.5:35b",
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "prompt_eval_count": 1200,
        "eval_count": 150,
        "prompt_eval_duration": 200_000_000,
        "eval_duration": 1_000_000_000,
        "load_duration": 50_000_000,
        "total_duration": 1_300_000_000,
    }
    payload.update(overrides)
    return payload


def client_for(session: FakeSession, **kwargs: Any) -> OllamaClient:
    """A client wired to the fake session with a short timeout."""
    return OllamaClient(session=session, timeout=5.0, **kwargs)


# --- 6. the request is built correctly --------------------------------------


def test_request_contains_messages_options_and_model() -> None:
    """The POST body carries the model, both messages and the sampling options."""
    session = FakeSession(post=[FakeResponse(payload=chat_payload())])
    client = client_for(session, model="qwen3.5:35b", keep_alive="30m")

    client.chat(
        [ChatMessage("system", "rules"), ChatMessage("user", "question")],
        options=GenerationOptions(temperature=0.1, num_ctx=8192, max_output_tokens=900),
    )

    body = session.posts[0]["json"]
    assert session.posts[0]["url"].endswith("/api/chat")
    assert body["model"] == "qwen3.5:35b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    assert body["options"]["temperature"] == 0.1
    assert body["options"]["num_ctx"] == 8192
    assert body["options"]["num_predict"] == 900
    assert body["keep_alive"] == "30m"
    assert body["options"]["seed"] == 0


def test_result_reports_real_token_counts_and_throughput() -> None:
    """Ollama's own counters are surfaced, not estimated."""
    session = FakeSession(post=[FakeResponse(payload=chat_payload())])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.ok is True
    assert result.content == "An answer [S1]."
    assert result.prompt_tokens == 1200
    assert result.generated_tokens == 150
    assert result.prompt_eval_seconds == 0.2
    assert result.generation_seconds == 1.0
    assert result.load_seconds == 0.05
    assert result.tokens_per_second == 150.0
    assert result.done_reason == "stop"
    assert isinstance(result.wall_seconds, float)
    assert result.wall_seconds >= 0.0


def test_thinking_output_is_captured_but_not_returned_as_the_answer() -> None:
    """A reasoning trace is recorded separately, never mixed into the answer."""
    payload = chat_payload(content="Final answer [S1].")
    payload["message"]["thinking"] = "Let me consider the sources..."
    session = FakeSession(post=[FakeResponse(payload=payload)])

    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.content == "Final answer [S1]."
    assert result.thinking == "Let me consider the sources..."


# --- 7. configurable model ---------------------------------------------------


def test_configured_model_name_is_used() -> None:
    """The model comes from configuration, not from a literal in the call."""
    session = FakeSession(post=[FakeResponse(payload=chat_payload())])
    client = client_for(session, model="some-other-model:latest")

    assert client.name == "some-other-model:latest"
    client.chat([ChatMessage("user", "q")])
    assert session.posts[0]["json"]["model"] == "some-other-model:latest"


def test_model_metadata_is_read_from_the_server_and_cached() -> None:
    """``describe`` reports what Ollama says about the model, once."""
    session = FakeSession(
        post=[
            FakeResponse(
                payload={
                    "details": {
                        "family": "qwen35moe",
                        "parameter_size": "34.7B",
                        "quantization_level": "Q4_K_M",
                    },
                    "model_info": {"qwen35moe.context_length": 262144},
                    "capabilities": ["completion", "thinking"],
                }
            )
        ],
        get={
            "http://127.0.0.1:11434/api/version": FakeResponse(payload={"version": "0.35.1"})
        },
    )
    client = client_for(session)

    described = client.describe()
    again = client.describe()

    assert described == again
    assert described["family"] == "qwen35moe"
    assert described["parameter_size"] == "34.7B"
    assert described["quantization_level"] == "Q4_K_M"
    assert described["context_length"] == 262144
    assert described["ollama_version"] == "0.35.1"
    assert described["capabilities"] == ["completion", "thinking"]
    assert len(session.posts) == 1  # cached, not re-requested


def test_available_models_lists_the_server_catalogue() -> None:
    """``available_models`` reads ``/api/tags``."""
    session = FakeSession(
        get={
            "http://127.0.0.1:11434/api/tags": FakeResponse(
                payload={"models": [{"name": "qwen3.5:35b"}, {"name": "other:1b"}]}
            )
        }
    )
    assert client_for(session).available_models() == ("qwen3.5:35b", "other:1b")


# --- 8. timeout --------------------------------------------------------------


def test_timeout_is_reported_as_a_timeout_failure() -> None:
    """A read timeout becomes a typed failure, not an exception into retrieval."""
    import requests

    session = FakeSession(post=[requests.exceptions.Timeout("read timed out")])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.ok is False
    assert result.content == ""
    assert result.failure is not None
    assert result.failure.kind == "timeout"
    assert result.failure.retryable is True
    assert "5s timeout" in result.failure.message


def test_connection_error_is_reported_as_unavailable() -> None:
    """A refused connection says the server could not be reached."""
    import requests

    session = FakeSession(post=[requests.exceptions.ConnectionError("refused")])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.failure is not None
    assert result.failure.kind == "connection_error"
    assert result.failure.is_unavailable is True
    assert "could not reach the model server" in result.failure.message


def test_interrupted_generation_is_not_reported_as_an_empty_success() -> None:
    """A transport error mid-call is a failure with the exception preserved."""
    session = FakeSession(post=[RuntimeError("connection reset by peer")])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.kind in ("connection_error", "interrupted")
    assert "connection reset by peer" in result.failure.message


# --- 9. missing model --------------------------------------------------------


def test_missing_model_is_reported_in_the_servers_own_words() -> None:
    """A 404 that mentions a missing model is classified as ``model_not_found``."""
    body = json.dumps({"error": "model 'qwen9:1b' not found, try pulling it first"})
    session = FakeSession(post=[FakeResponse(status_code=404, text=body)])
    result = client_for(session, model="qwen9:1b").chat([ChatMessage("user", "q")])

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.kind == "model_not_found"
    assert result.failure.status_code == 404
    assert "not found" in result.failure.message
    assert result.failure.retryable is False


def test_server_error_is_marked_retryable() -> None:
    """A 500 is a server problem and is reported as retryable."""
    session = FakeSession(post=[FakeResponse(500, text="internal error")])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.failure is not None
    assert result.failure.kind == "server_error"
    assert result.failure.retryable is True


# --- 10. malformed response --------------------------------------------------


def test_non_json_body_is_reported_as_malformed() -> None:
    """An HTML error page instead of JSON is a malformed response."""
    session = FakeSession(post=[FakeResponse(200, None, text="<html>nope</html>")])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.kind == "malformed_response"
    assert "not JSON" in result.failure.message


def test_response_without_a_message_field_is_reported() -> None:
    """A JSON body with the wrong shape is a malformed response, not an empty answer."""
    session = FakeSession(post=[FakeResponse(200, {"done": True})])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.failure is not None
    assert result.failure.kind == "malformed_response"
    assert "message.content" in result.failure.message


def test_non_text_content_is_reported() -> None:
    """Content that is not a string is a malformed response."""
    session = FakeSession(post=[FakeResponse(200, {"message": {"content": ["a", "b"]}})])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.failure is not None
    assert result.failure.kind == "malformed_response"
    assert "not text" in result.failure.message


def test_empty_answer_is_not_counted_as_success() -> None:
    """Whitespace-only output is a failure: there is no answer to ground."""
    session = FakeSession(post=[FakeResponse(payload=chat_payload(content="   \n"))])
    result = client_for(session).chat([ChatMessage("user", "q")])

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.kind == "empty_response"


def test_generation_is_never_retried_automatically() -> None:
    """One call, one answer: a silent retry would produce a second, different answer."""
    session = FakeSession(post=[FakeResponse(500, text="boom")])
    client_for(session).chat([ChatMessage("user", "q")])

    assert len(session.posts) == 1


def test_model_summary_survives_an_unreachable_server() -> None:
    """A report helper never fails because the server is down."""
    session = FakeSession(post=[], get={})
    summary = model_summary(client_for(session, model="qwen3.5:35b"))

    assert summary["model"] == "qwen3.5:35b"
    assert "checked_at" in summary


def test_failure_object_is_serialisable_for_the_audit() -> None:
    """The failure travels into the audit as data, not as an exception."""
    failure = ModelFailure(kind="timeout", message="slow", status_code=None, retryable=True)
    assert failure.model_dump()["kind"] == "timeout"
    assert failure.is_unavailable is False
