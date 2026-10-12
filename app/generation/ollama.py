"""The generation-model abstraction and its Ollama implementation (RAG-011 §2, §3).

The rest of the application never talks to Ollama directly. It calls a
:class:`GenerationModel`, which returns a :class:`ChatResult` that always carries either
content or a typed :class:`~app.generation.models.ModelFailure`. That is the whole point
of the boundary: "the model answered nothing useful" and "the server is not running"
must not look the same, and a provider problem must never damage the retrieval state.

Generation is deliberately **not retried**. A retry would silently produce a second,
different answer for the same question, which is worse for grounding and for
reproducibility than reporting the failure once.
"""

from __future__ import annotations

import datetime as _datetime
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from app.generation.models import ModelFailure

#: Ollama error substrings mapped to a clearer failure kind.
_MODEL_NOT_FOUND_HINTS = ("model not found", "no such model", "pull model", "not found")


@dataclass(frozen=True)
class ChatMessage:
    """One chat turn."""

    role: str
    content: str

    def as_payload(self) -> dict[str, str]:
        """The wire form Ollama expects."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationOptions:
    """The sampling and window settings for one call."""

    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 8192
    max_output_tokens: int = 900
    seed: int | None = 0
    stop: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        """The ``options`` object Ollama expects."""
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "num_predict": self.max_output_tokens,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop:
            options["stop"] = list(self.stop)
        return options


@dataclass
class ChatResult:
    """What one generation call produced (or why it produced nothing)."""

    model: str
    content: str = ""
    thinking: str | None = None
    done_reason: str | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_eval_seconds: float | None = None
    generation_seconds: float | None = None
    load_seconds: float | None = None
    total_seconds: float | None = None
    wall_seconds: float = 0.0
    model_details: dict[str, Any] = field(default_factory=dict)
    failure: ModelFailure | None = None

    @property
    def ok(self) -> bool:
        """Whether the model returned usable content."""
        return self.failure is None and bool(self.content.strip())

    @property
    def tokens_per_second(self) -> float | None:
        """Real output throughput, when Ollama reported both numbers."""
        if self.generated_tokens and self.generation_seconds:
            return round(self.generated_tokens / self.generation_seconds, 2)
        return None


@runtime_checkable
class GenerationModel(Protocol):
    """A text-generation backend."""

    @property
    def name(self) -> str:
        """The configured model name."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """Generate one answer."""

    def describe(self) -> dict[str, Any]:
        """Model metadata (family, quantization, context length) where available."""


def _nanoseconds_to_seconds(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds."""
    if isinstance(value, (int, float)) and value > 0:
        return round(float(value) / 1e9, 6)
    return None


def _failure_from_response(status: int, body: str, endpoint: str) -> ModelFailure:
    """Map an HTTP error to a typed failure, keeping the server's own words."""
    text = (body or "").strip()
    lowered = text.lower()
    if status == 404 or any(hint in lowered for hint in _MODEL_NOT_FOUND_HINTS):
        return ModelFailure(
            kind="model_not_found",
            message=text[:400] or f"HTTP {status}",
            status_code=status,
            endpoint=endpoint,
            retryable=False,
        )
    if status == 400:
        return ModelFailure(
            kind="invalid_request",
            message=text[:400] or "the model server rejected the request",
            status_code=status,
            endpoint=endpoint,
        )
    if status >= 500:
        return ModelFailure(
            kind="server_error",
            message=text[:400] or f"HTTP {status}",
            status_code=status,
            endpoint=endpoint,
            retryable=True,
        )
    return ModelFailure(
        kind="http_error",
        message=text[:400] or f"HTTP {status}",
        status_code=status,
        endpoint=endpoint,
    )


class OllamaClient:
    """A minimal, injectable Ollama client.

    ``session`` can be replaced with any object exposing ``post``/``get`` returning a
    ``requests``-like response, which is how the tests stay offline.
    """

    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:35b",
        timeout: float = 300.0,
        keep_alive: str | None = "30m",
        think: bool = False,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.timeout = float(timeout)
        self.keep_alive = keep_alive
        self.think = think
        self._session = session
        self._details: dict[str, Any] | None = None
        self._version: str | None = None

    # --- plumbing -------------------------------------------------------------
    @property
    def session(self) -> Any:
        """The HTTP session, created on first use."""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    @property
    def name(self) -> str:
        """The configured model name."""
        return self._model

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any, str]:
        """POST JSON, returning ``(status, parsed_json_or_None, raw_text)``."""
        endpoint = f"{self.base_url}{path}"
        response = self.session.post(endpoint, json=payload, timeout=self.timeout)
        status = int(getattr(response, "status_code", 0))
        text = getattr(response, "text", "") or ""
        try:
            data = response.json()
        except Exception:
            data = None
        return status, data, text

    def _get_json(self, path: str, *, timeout: float | None = None) -> Any | None:
        """GET JSON, returning ``None`` when the server cannot answer."""
        endpoint = f"{self.base_url}{path}"
        try:
            response = self.session.get(endpoint, timeout=timeout or min(self.timeout, 15.0))
            if int(getattr(response, "status_code", 0)) != 200:
                return None
            return response.json()
        except Exception:
            return None

    # --- introspection --------------------------------------------------------
    def version(self) -> str | None:
        """The Ollama server version, when it can be read."""
        if self._version is None:
            payload = self._get_json("/api/version")
            if isinstance(payload, dict):
                self._version = str(payload.get("version") or "") or None
        return self._version

    def available_models(self, *, timeout: float | None = None) -> tuple[str, ...]:
        """Model names the server currently has."""
        payload = self._get_json("/api/tags", timeout=timeout or 30.0)
        if not isinstance(payload, dict):
            return ()
        names = [
            str(item.get("name"))
            for item in payload.get("models") or []
            if isinstance(item, dict) and item.get("name")
        ]
        return tuple(names)

    def describe(self) -> dict[str, Any]:
        """Model metadata from ``/api/show`` (cached), never fatal."""
        if self._details is not None:
            return self._details
        details: dict[str, Any] = {"model": self._model}
        payload = None
        try:
            status, payload, _text = self._post("/api/show", {"model": self._model})
            if status != 200:
                payload = None
        except Exception:
            payload = None
        if isinstance(payload, dict):
            raw_details = payload.get("details") or {}
            if isinstance(raw_details, dict):
                details.update(
                    {
                        key: raw_details.get(key)
                        for key in (
                            "family",
                            "families",
                            "parameter_size",
                            "quantization_level",
                            "format",
                            "parent_model",
                        )
                    }
                )
            model_info = payload.get("model_info") or {}
            if isinstance(model_info, dict):
                context = [
                    value
                    for key, value in model_info.items()
                    if key.endswith("context_length")
                ]
                if context:
                    details["context_length"] = max(
                        value for value in context if isinstance(value, int)
                    )
            capabilities = payload.get("capabilities")
            if isinstance(capabilities, list):
                details["capabilities"] = [str(item) for item in capabilities]
        details["ollama_version"] = self.version()
        self._details = details
        return details

    # --- generation -----------------------------------------------------------
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """Generate one answer, mapping every failure to a typed result."""
        options = options or GenerationOptions()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.as_payload() for message in messages],
            "stream": False,
            "options": options.as_payload(),
            "think": self.think if think is None else think,
            "tools": [],
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        endpoint = f"{self.base_url}/api/chat"

        started = time.perf_counter()
        try:
            status, data, text = self._post("/api/chat", payload)
        except Exception as exc:  # connection refused, timeout, reset mid-stream
            return ChatResult(
                model=self._model,
                wall_seconds=round(time.perf_counter() - started, 4),
                failure=self._failure_from_exception(exc, endpoint),
            )
        wall = round(time.perf_counter() - started, 4)

        if status != 200:
            return ChatResult(
                model=self._model,
                wall_seconds=wall,
                failure=_failure_from_response(status, text, endpoint),
            )
        if not isinstance(data, dict):
            return ChatResult(
                model=self._model,
                wall_seconds=wall,
                failure=ModelFailure(
                    kind="malformed_response",
                    message="the model server returned a body that is not JSON",
                    status_code=status,
                    endpoint=endpoint,
                ),
            )
        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            return ChatResult(
                model=self._model,
                wall_seconds=wall,
                failure=ModelFailure(
                    kind="malformed_response",
                    message="the response has no message.content field",
                    status_code=status,
                    endpoint=endpoint,
                ),
            )
        content = message.get("content")
        if not isinstance(content, str):
            return ChatResult(
                model=self._model,
                wall_seconds=wall,
                failure=ModelFailure(
                    kind="malformed_response",
                    message="the response content is not text",
                    status_code=status,
                    endpoint=endpoint,
                ),
            )
        thinking = message.get("thinking")
        result = ChatResult(
            model=str(data.get("model") or self._model),
            content=content,
            thinking=str(thinking) if isinstance(thinking, str) and thinking.strip() else None,
            done_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            generated_tokens=data.get("eval_count"),
            prompt_eval_seconds=_nanoseconds_to_seconds(data.get("prompt_eval_duration")),
            generation_seconds=_nanoseconds_to_seconds(data.get("eval_duration")),
            load_seconds=_nanoseconds_to_seconds(data.get("load_duration")),
            total_seconds=_nanoseconds_to_seconds(data.get("total_duration")),
            wall_seconds=wall,
            model_details=self.describe(),
        )
        if not result.content.strip():
            result.failure = ModelFailure(
                kind="empty_response",
                message=(
                    "the model returned no text"
                    + (f" (done_reason={result.done_reason})" if result.done_reason else "")
                ),
                status_code=status,
                endpoint=endpoint,
            )
        return result

    def unload(self) -> bool:
        """Ask Ollama to drop this model, so the next call starts a fresh runner.

        Used by the simple chat's one-retry recovery (RAG-015 FIX1): when Ollama's runner has
        ended up emitting one token forever, only a fresh runner produces sane output, and
        ``keep_alive: 0`` is the documented way to ask for that. Best effort: a failure here
        is reported, never raised, because the retry itself is still worth attempting.
        """
        try:
            status, _data, _text = self._post(
                "/api/generate", {"model": self._model, "keep_alive": 0, "stream": False}
            )
        except Exception:
            return False
        self._details = None
        return status == 200

    def _failure_from_exception(self, exc: Exception, endpoint: str) -> ModelFailure:
        """Classify a transport exception without losing its message."""
        name = type(exc).__name__
        message = f"{name}: {exc}"
        try:  # requests is a project dependency, but never assume it is installed
            import requests

            if isinstance(exc, requests.exceptions.Timeout):
                return ModelFailure(
                    kind="timeout",
                    message=f"generation exceeded the {self.timeout:g}s timeout ({message})",
                    endpoint=endpoint,
                    retryable=True,
                )
            if isinstance(exc, requests.exceptions.ConnectionError):
                return ModelFailure(
                    kind="connection_error",
                    message=f"could not reach the model server at {self.base_url}: {message}",
                    endpoint=endpoint,
                    retryable=True,
                )
        except Exception:  # pragma: no cover - requests always present in practice
            pass
        if "timeout" in name.lower():
            return ModelFailure(
                kind="timeout",
                message=f"generation exceeded the {self.timeout:g}s timeout ({message})",
                endpoint=endpoint,
                retryable=True,
            )
        if "connection" in name.lower() or "refused" in message.lower():
            return ModelFailure(
                kind="connection_error",
                message=f"could not reach the model server at {self.base_url}: {message}",
                endpoint=endpoint,
                retryable=True,
            )
        return ModelFailure(kind="interrupted", message=message, endpoint=endpoint)


def model_summary(client: GenerationModel) -> dict[str, Any]:
    """A small, report-friendly description of the configured model."""
    try:
        described = client.describe()
    except Exception as exc:  # pragma: no cover - defensive
        described = {"model": client.name, "error": f"{type(exc).__name__}: {exc}"}
    described.setdefault("model", client.name)
    described["checked_at"] = _datetime.datetime.now(_datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    return described
