"""DeepSeek provider tests (RAG-016 §4, §5, §15, §16, §17).

The client is exercised through an injected session, so the request shape, the normalised
result and every failure sentence are proven without a key and without the Internet. The two
rules that matter most are checked directly: the key travels in a header and never in an error,
and reasoning output is never read.
"""

from __future__ import annotations

import json
from typing import Any

from app.generation.deepseek import DeepSeekClient, redact
from app.generation.ollama import ChatMessage, GenerationOptions

#: A key that only exists in this file. Deliberately shaped like a real one.
KEY = "sk-0123456789abcdef0123456789abcdef"


class FakeResponse:
    """A minimal stand-in for a ``requests`` response."""

    def __init__(self, status_code: int = 200, payload: Any = None, *, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})

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
        get: list[FakeResponse | Exception] | None = None,
    ) -> None:
        self._post = list(post or [])
        self._get = list(get or [])
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self._post.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        item = self._get.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def completion(content: str = "An answer.", **overrides: Any) -> dict[str, Any]:
    """A normal OpenAI-compatible completion body, as DeepSeek returns it."""
    payload: dict[str, Any] = {
        "id": "req-abc-123",
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 21284,
            "completion_tokens": 864,
            "total_tokens": 22148,
            "prompt_cache_hit_tokens": 14500,
            "prompt_cache_miss_tokens": 6784,
        },
    }
    payload.update(overrides)
    return payload


def client_for(session: FakeSession, **kwargs: Any) -> DeepSeekClient:
    """A client wired to the fake session, with a key and a short timeout."""
    return DeepSeekClient(api_key=KEY, session=session, timeout=5.0, **kwargs)


# --- 16. the request and the normalised result -------------------------------


def test_request_is_openai_shaped_and_authenticated_by_header() -> None:
    """The model, the messages and the sampling options go in the body; the key in a header."""
    session = FakeSession(post=[FakeResponse(payload=completion())])
    client = client_for(session, model="deepseek-flash")

    client.chat(
        [ChatMessage("system", "rules"), ChatMessage("user", "question")],
        options=GenerationOptions(temperature=0.1, top_p=0.9, max_output_tokens=900),
    )

    request = session.posts[0]
    assert request["url"] == "https://api.deepseek.com/chat/completions"
    assert request["json"]["model"] == "deepseek-flash"
    assert request["json"]["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    assert request["json"]["temperature"] == 0.1
    assert request["json"]["max_tokens"] == 900
    assert request["json"]["stream"] is False
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert KEY not in json.dumps(request["json"])


def test_result_is_normalised_into_the_projects_chat_result() -> None:
    """Every number the provider reports is carried through; nothing is invented (§16, §19)."""
    session = FakeSession(
        post=[
            FakeResponse(
                payload=completion(
                    usage={
                        "prompt_tokens": 21284,
                        "completion_tokens": 864,
                        "prompt_cache_hit_tokens": 14500,
                        "completion_tokens_details": {"reasoning_tokens": 1100},
                    }
                )
            )
        ]
    )

    result = client_for(session).chat([ChatMessage("user", "hi")])

    assert result.ok is True
    assert result.provider == "deepseek"
    assert result.model == "deepseek-flash"
    assert result.content == "An answer."
    assert result.prompt_tokens == 21284
    assert result.generated_tokens == 864
    assert result.cached_input_tokens == 14500
    assert result.reasoning_tokens == 1100
    assert result.request_id == "req-abc-123"
    # A latency is measured around the request, however fast an injected session returns.
    assert result.wall_seconds >= 0.0


def test_unreported_usage_stays_none() -> None:
    """A provider that reports no numbers produces no numbers, rather than zeros."""
    session = FakeSession(
        post=[FakeResponse(payload={"model": "deepseek-flash", "choices": [{"message": {"content": "ok"}}]})]
    )

    result = client_for(session).chat([ChatMessage("user", "hi")])

    assert result.prompt_tokens is None
    assert result.generated_tokens is None
    assert result.cached_input_tokens is None
    assert result.reasoning_tokens is None


def test_reasoning_content_is_never_read() -> None:
    """A reasoning model's private trace is not the answer and is not carried anywhere (§15)."""
    session = FakeSession(
        post=[
            FakeResponse(
                payload=completion(
                    content="The final answer.",
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The final answer.",
                                "reasoning_content": "SECRET CHAIN OF THOUGHT",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                )
            )
        ]
    )

    result = client_for(session).chat([ChatMessage("user", "hi")], think=True)

    assert result.content == "The final answer."
    assert result.thinking is None
    assert "SECRET" not in json.dumps(
        {"content": result.content, "thinking": result.thinking, "details": result.model_details}
    )


def test_the_same_client_answers_with_either_configured_model() -> None:
    """One client serves both DeepSeek models, so switching one does not rebuild anything."""
    session = FakeSession(post=[FakeResponse(payload=completion()), FakeResponse(payload=completion())])
    client = client_for(session)
    assert client.name == "deepseek-flash"

    client.chat([ChatMessage("user", "hi")])
    client.use_model("deepseek-v4-pro")
    client.chat([ChatMessage("user", "hi")])
    client.use_model("")  # an empty id changes nothing

    assert [item["json"]["model"] for item in session.posts] == [
        "deepseek-flash",
        "deepseek-v4-pro",
    ]
    assert client.name == "deepseek-v4-pro"


def test_a_missing_key_fails_before_any_request_is_made() -> None:
    """With no key configured the client explains itself instead of calling out anonymously."""
    session = FakeSession()
    client = DeepSeekClient(api_key=None, session=session)

    result = client.chat([ChatMessage("user", "hi")])

    assert session.posts == []
    assert result.failure is not None
    assert result.failure.kind == "not_configured"
    assert "DEEPSEEK_API_KEY" in result.failure.message
    assert client.configured is False


def test_model_ids_come_from_the_provider() -> None:
    """``/models`` is what the selector offers; a failure to read it is not an error (§5)."""
    session = FakeSession(
        get=[
            FakeResponse(
                payload={
                    "object": "list",
                    "data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}],
                }
            )
        ]
    )
    client = client_for(session)

    assert client.available_models() == ("deepseek-flash", "deepseek-v4-pro")
    assert session.gets[0]["url"] == "https://api.deepseek.com/models"
    assert session.gets[0]["headers"]["Authorization"] == f"Bearer {KEY}"

    offline = DeepSeekClient(api_key=KEY, session=FakeSession(get=[RuntimeError("offline")]))
    assert offline.available_models() == ()


def test_a_custom_base_url_is_honoured() -> None:
    """The endpoint is configuration, never a constant."""
    session = FakeSession(post=[FakeResponse(payload=completion())])
    client = client_for(session, base_url="https://proxy.example/v1/")

    client.chat([ChatMessage("user", "hi")])

    assert session.posts[0]["url"] == "https://proxy.example/v1/chat/completions"


# --- 17. every useful failure, in a readable sentence ------------------------


def failure_for(status: int, body: Any = None, *, text: str | None = None) -> Any:
    """Run one failing request and return its typed failure."""
    session = FakeSession(post=[FakeResponse(status_code=status, payload=body, text=text)])
    result = client_for(session).chat([ChatMessage("user", "hi")])
    assert result.failure is not None
    assert result.content == ""
    return result.failure


def test_authentication_failure_is_a_sentence_not_a_trace() -> None:
    failure = failure_for(401, {"error": {"message": "Authentication Fails"}})

    assert failure.kind == "auth_error"
    assert failure.message.startswith("DeepSeek authentication failed")
    assert "DEEPSEEK_API_KEY" in failure.message
    assert failure.retryable is False


def test_insufficient_balance_is_named() -> None:
    failure = failure_for(402, {"error": {"message": "Insufficient Balance"}})

    assert failure.kind == "insufficient_balance"
    assert "balance" in failure.message
    assert failure.retryable is False


def test_rate_limit_tells_the_user_to_retry_themselves() -> None:
    failure = failure_for(429, {"error": {"message": "Rate limit reached"}})

    assert failure.kind == "rate_limit"
    assert "rate limit" in failure.message.lower()
    assert failure.retryable is True


def test_an_outage_is_retryable_but_never_automatic() -> None:
    failure = failure_for(503, text="<html>gateway</html>")

    assert failure.kind == "server_error"
    assert "temporarily unavailable" in failure.message
    assert failure.retryable is True


def test_an_unknown_model_is_reported_with_the_configured_id() -> None:
    failure = failure_for(400, {"error": {"message": "Model Not Exist"}})

    assert failure.kind == "model_not_found"
    assert "deepseek-flash" in failure.message
    assert "DEEPSEEK_MODEL" in failure.message


def test_a_timeout_and_a_connection_failure_are_distinguished() -> None:
    import requests

    timed_out = client_for(
        FakeSession(post=[requests.exceptions.Timeout("too slow")])
    ).chat([ChatMessage("user", "hi")])
    offline = client_for(
        FakeSession(post=[requests.exceptions.ConnectionError("refused")])
    ).chat([ChatMessage("user", "hi")])

    assert timed_out.failure.kind == "timeout"
    assert "timed out" in timed_out.failure.message
    assert offline.failure.kind == "connection_error"
    assert "Could not reach the DeepSeek API" in offline.failure.message


def test_an_empty_answer_is_a_failure_not_an_empty_message() -> None:
    session = FakeSession(post=[FakeResponse(payload=completion(content="   "))])

    result = client_for(session).chat([ChatMessage("user", "hi")])

    assert result.ok is False
    assert result.failure.kind == "empty_response"


def test_a_malformed_body_is_reported_rather_than_crashing() -> None:
    session = FakeSession(post=[FakeResponse(payload={"choices": []})])

    result = client_for(session).chat([ChatMessage("user", "hi")])

    assert result.failure.kind == "malformed_response"


# --- 4. the key can never appear in an error ---------------------------------


def test_the_key_is_redacted_from_every_error_path() -> None:
    """A proxy or the provider itself may echo the header back; it still must not reach a log."""
    echoed = failure_for(
        500, text=f'{{"error": {{"message": "Bad header Bearer {KEY} for org"}}}}'
    )
    described = failure_for(400, {"error": {"message": f"key {KEY} is invalid"}})

    for failure in (echoed, described):
        assert KEY not in json.dumps(failure.model_dump())
        assert "[redacted]" in failure.detail


def test_redact_removes_the_configured_key_and_other_bearer_tokens() -> None:
    text = f"sent Authorization: Bearer {KEY} and Bearer sk-other-key-999"

    cleaned = redact(text, secret=KEY)

    assert KEY not in cleaned
    assert "sk-other-key-999" not in cleaned
    assert cleaned.count("[redacted]") == 2
