"""The DeepSeek chat provider (RAG-016 §4, §5, §16, §17).

This is the *one* place the application talks to a cloud model, and it exists for exactly one
purpose: when local Ollama is unreliable, the user can select DeepSeek for the next message of
a simple chat and keep working. It is a generation provider and nothing else — it stores
nothing, is asked nothing about the corpus, and is never reached automatically.

The client is deliberately small and model-agnostic. It speaks the provider's OpenAI-compatible
``/chat/completions`` endpoint with ``requests`` (already a project dependency, no SDK), takes
the same :class:`~app.generation.ollama.ChatMessage` sequence the local client takes, and
normalises the answer into the same :class:`~app.generation.ollama.ChatResult` the rest of the
application already understands. Model ids come from configuration and from the provider's own
``/models`` answer, never from a code constant.

Two rules are enforced here rather than trusted downstream:

* the API key is sent as a header only, and every failure message passes through
  :func:`redact` before it leaves this module, so a key can never appear in an error, a log or
  the UI;
* ``reasoning_content`` is never read. The number of reasoning tokens may be reported in usage
  metadata, but the reasoning text is not part of the answer and is not stored (RAG-016 §15).
"""

from __future__ import annotations

import re
import time
from typing import Any, Sequence

from app.generation.models import ModelFailure
from app.generation.ollama import ChatMessage, ChatResult, GenerationOptions

#: The stored/selected name of this provider.
PROVIDER = "deepseek"

#: Anything that looks like a bearer token, so a key pasted into an error by a proxy or by the
#: provider itself is still not echoed back.
_SECRET_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{6,}", re.IGNORECASE)

#: Words the provider uses when the model id in the request does not exist.
_MODEL_HINTS = (
    "model not found",
    "model_not_found",
    "not exist",
    "does not exist",
    "invalid model",
)


def redact(text: str, *, secret: str | None = None) -> str:
    """Remove credentials from a string that came from the network.

    The configured key is replaced first (it is the only string known to be a secret), then any
    other bearer token. Errors are the one place a key is most likely to leak, so this is
    applied to every failure path, not only to the ones that look risky.
    """
    cleaned = text or ""
    if secret:
        cleaned = cleaned.replace(secret, "[redacted]")
    return _SECRET_PATTERN.sub(r"\1[redacted]", cleaned)


def _short(text: str, *, limit: int = 300) -> str:
    """Collapse whitespace and clip, so a failure sentence stays readable."""
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


class DeepSeekClient:
    """A minimal DeepSeek chat client.

    ``session`` can be replaced with any object exposing ``post``/``get`` and returning a
    ``requests``-like response, which is how the tests stay offline. ``api_key`` is never
    logged, serialised or included in an error: when it is missing the client fails with a
    configuration sentence instead of calling out anonymously.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout: float = 300.0,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.timeout = float(timeout)
        self._session = session
        self._api_key = (api_key or "").strip()

    def use_model(self, model: str) -> None:
        """Point this client at another model id, for the next call (RAG-016 FIX1).

        One provider answers with several models, and the user switches between them per
        message, so the client is reused rather than rebuilt. An empty id changes nothing.
        """
        clean = str(model or "").strip()
        if clean:
            self._model = clean

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
        """The configured model id."""
        return self._model

    @property
    def configured(self) -> bool:
        """Whether a key is configured (never *what* the key is)."""
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any, str]:
        """POST JSON, returning ``(status, parsed_json_or_None, raw_text)``."""
        response = self.session.post(
            f"{self.base_url}{path}", json=payload, headers=self._headers(), timeout=self.timeout
        )
        status = int(getattr(response, "status_code", 0))
        text = getattr(response, "text", "") or ""
        try:
            data = response.json()
        except Exception:
            data = None
        return status, data, text

    def _get_json(self, path: str, *, timeout: float | None = None) -> Any | None:
        """GET JSON, returning ``None`` when the provider cannot answer."""
        try:
            response = self.session.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                timeout=timeout or min(self.timeout, 15.0),
            )
            if int(getattr(response, "status_code", 0)) != 200:
                return None
            return response.json()
        except Exception:
            return None

    @staticmethod
    def _error_text(data: Any, raw: str) -> str:
        """The provider's own error sentence, whichever shape it arrived in."""
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("type") or "")
            if isinstance(error, str):
                return error
            if isinstance(data.get("message"), str):
                return str(data["message"])
        return raw

    # --- introspection --------------------------------------------------------
    def available_models(self, *, timeout: float | None = None) -> tuple[str, ...]:
        """Model ids the provider currently offers (empty when it cannot be asked).

        Used to show the user what is actually available instead of a name hard-coded here
        (RAG-016 §5). Never fatal, and asked at most once per process: a listing is not worth
        making a page wait, so the timeout is short and an unreachable endpoint answers with
        nothing.
        """
        if not self._api_key:
            return ()
        payload = self._get_json("/models", timeout=timeout or min(self.timeout, 5.0))
        if not isinstance(payload, dict):
            return ()
        return tuple(
            str(item.get("id"))
            for item in payload.get("data") or []
            if isinstance(item, dict) and item.get("id")
        )

    def describe(self) -> dict[str, Any]:
        """Provider metadata for the debug panel. Never includes the key."""
        return {
            "provider": PROVIDER,
            "model": self._model,
            "base_url": self.base_url,
            "api_key_configured": self.configured,
        }

    # --- generation -----------------------------------------------------------
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """Generate one answer, mapping every failure to a typed result.

        ``think`` is accepted for interface compatibility and deliberately ignored: the
        reasoning channel is not requested, and ``reasoning_content`` is never read even when a
        model returns it (RAG-016 §15).
        """
        options = options or GenerationOptions()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "stream": False,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_output_tokens,
        }
        if options.stop:
            payload["stop"] = list(options.stop)
        endpoint = f"{self.base_url}/chat/completions"

        if not self._api_key:
            return ChatResult(
                model=self._model,
                provider=PROVIDER,
                failure=ModelFailure(
                    kind="not_configured",
                    message=(
                        "DeepSeek is not configured: set DEEPSEEK_API_KEY (or "
                        "PHD_RAG_DEEPSEEK_API_KEY) on the server and restart it."
                    ),
                    endpoint=endpoint,
                    retryable=False,
                ),
            )

        started = time.perf_counter()
        try:
            status, data, text = self._post("/chat/completions", payload)
        except Exception as exc:  # timeout, connection refused, reset mid-request
            wall = round(time.perf_counter() - started, 4)
            return ChatResult(
                model=self._model,
                provider=PROVIDER,
                wall_seconds=wall,
                failure=self._failure_from_exception(exc, endpoint),
            )
        wall = round(time.perf_counter() - started, 4)

        if status != 200:
            return ChatResult(
                model=self._model,
                provider=PROVIDER,
                wall_seconds=wall,
                failure=self._failure_from_response(status, self._error_text(data, text), endpoint),
            )

        choices = data.get("choices") if isinstance(data, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        if not isinstance(message, dict):
            return ChatResult(
                model=self._model,
                provider=PROVIDER,
                wall_seconds=wall,
                failure=ModelFailure(
                    kind="malformed_response",
                    message=(
                        "DeepSeek returned a body with no choice, so there is no answer to "
                        "store. Your message is saved; Retry this turn sends it again."
                    ),
                    status_code=status,
                    endpoint=endpoint,
                ),
            )

        content = message.get("content")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        reasoning_tokens = (
            details.get("reasoning_tokens") if isinstance(details, dict) else None
        )
        # ``prompt_cache_hit_tokens`` is the provider's name for cached input tokens.
        cached = usage.get("prompt_cache_hit_tokens") if isinstance(usage, dict) else None
        result = ChatResult(
            model=str(data.get("model") or self._model),
            content=content if isinstance(content, str) else "",
            done_reason=(
                choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
            ),
            prompt_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            generated_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
            cached_input_tokens=cached if isinstance(cached, int) else None,
            reasoning_tokens=reasoning_tokens if isinstance(reasoning_tokens, int) else None,
            wall_seconds=wall,
            model_details=self.describe(),
            provider=PROVIDER,
            request_id=str(data.get("id") or ""),
        )
        if not result.content.strip():
            result.failure = ModelFailure(
                kind="empty_response",
                message=(
                    "DeepSeek returned no text"
                    + (
                        f" (finish_reason={result.done_reason})"
                        if result.done_reason
                        else ""
                    )
                    + ". Your message is saved; Retry this turn sends it again."
                ),
                status_code=status,
                endpoint=endpoint,
            )
        return result

    # --- failure classification ----------------------------------------------
    def _failure_from_response(self, status: int, body: str, endpoint: str) -> ModelFailure:
        """Map an HTTP status to a typed failure with one useful sentence (RAG-016 §17)."""
        detail = redact(_short(body), secret=self._api_key)
        lowered = detail.lower()
        if status in (401, 403):
            return ModelFailure(
                kind="auth_error",
                message=(
                    "DeepSeek authentication failed. Check DEEPSEEK_API_KEY on the server "
                    "(the key is never read from the browser)."
                ),
                status_code=status,
                endpoint=endpoint,
                detail=detail,
            )
        if status == 402 or "insufficient balance" in lowered:
            return ModelFailure(
                kind="insufficient_balance",
                message=(
                    "DeepSeek refused the request because the account balance is "
                    "insufficient. Top up, or switch back to the local model."
                ),
                status_code=status,
                endpoint=endpoint,
                detail=detail,
            )
        if status == 429:
            return ModelFailure(
                kind="rate_limit",
                message=(
                    "DeepSeek rate limit reached. Wait a moment, then use Retry this turn — "
                    "the message is saved and is not sent twice."
                ),
                status_code=status,
                endpoint=endpoint,
                retryable=True,
                detail=detail,
            )
        if status == 404 or any(hint in lowered for hint in _MODEL_HINTS):
            return ModelFailure(
                kind="model_not_found",
                message=(
                    f"DeepSeek does not recognise the model '{self._model}'. Set "
                    "DEEPSEEK_MODEL to an id DeepSeek offers and restart the server."
                ),
                status_code=status,
                endpoint=endpoint,
                detail=detail,
            )
        if status >= 500:
            return ModelFailure(
                kind="server_error",
                message=(
                    f"DeepSeek service is temporarily unavailable (HTTP {status}). Your "
                    "message is saved; Retry this turn sends it again."
                ),
                status_code=status,
                endpoint=endpoint,
                retryable=True,
                detail=detail,
            )
        if status == 400:
            return ModelFailure(
                kind="invalid_request",
                message=(
                    "DeepSeek rejected the request as invalid. The selected context may be "
                    f"larger than the model accepts (HTTP {status})."
                ),
                status_code=status,
                endpoint=endpoint,
                detail=detail,
            )
        return ModelFailure(
            kind="http_error",
            message=f"DeepSeek returned HTTP {status} and no answer.",
            status_code=status,
            endpoint=endpoint,
            detail=detail,
        )

    def _failure_from_exception(self, exc: Exception, endpoint: str) -> ModelFailure:
        """Classify a transport exception without losing its message."""
        name = type(exc).__name__
        detail = redact(f"{name}: {exc}", secret=self._api_key)
        try:  # requests is a project dependency, but never assume it is installed
            import requests

            if isinstance(exc, requests.exceptions.Timeout):
                return ModelFailure(
                    kind="timeout",
                    message=(
                        f"DeepSeek request timed out after {self.timeout:g}s. Your message is "
                        "saved; Retry this turn sends it again."
                    ),
                    endpoint=endpoint,
                    retryable=True,
                    detail=detail,
                )
            if isinstance(exc, requests.exceptions.ConnectionError):
                return ModelFailure(
                    kind="connection_error",
                    message=(
                        f"Could not reach the DeepSeek API at {self.base_url}. Check the "
                        "network or DEEPSEEK_BASE_URL, or switch back to the local model."
                    ),
                    endpoint=endpoint,
                    retryable=True,
                    detail=detail,
                )
        except Exception:  # pragma: no cover - requests always present in practice
            pass
        if "timeout" in name.lower():
            return ModelFailure(
                kind="timeout",
                message=f"DeepSeek request timed out after {self.timeout:g}s.",
                endpoint=endpoint,
                retryable=True,
                detail=detail,
            )
        if "connection" in name.lower() or "refused" in detail.lower():
            return ModelFailure(
                kind="connection_error",
                message=f"Could not reach the DeepSeek API at {self.base_url}.",
                endpoint=endpoint,
                retryable=True,
                detail=detail,
            )
        return ModelFailure(kind="interrupted", message=detail, endpoint=endpoint, detail=detail)


__all__ = ["DeepSeekClient", "PROVIDER", "redact"]
