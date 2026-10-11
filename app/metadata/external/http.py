"""Minimal HTTP client with timeout, bounded retry, and rate limiting.

The transport is injectable so tests never touch the network: a fake transport
that records calls can be substituted for :class:`RequestsTransport`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.parse import urlparse


class HttpResponse(Protocol):
    """The subset of a response object this client relies on."""

    status_code: int
    text: str

    def json(self) -> Any: ...


class Transport(Protocol):
    """A synchronous HTTP GET transport."""

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HttpResponse: ...


class RequestsTransport:
    """A :class:`Transport` backed by ``requests`` (imported lazily)."""

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HttpResponse:
        import requests

        return requests.get(url, params=params, headers=headers, timeout=timeout)


class RateLimiter:
    """A trivial monotonic-clock rate limiter enforcing a minimum interval."""

    def __init__(self, min_interval: float, *, sleep: Callable[[float], None] = time.sleep):
        self._min_interval = max(0.0, min_interval)
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Sleep if needed so consecutive calls are spaced by the interval."""
        now = time.monotonic()
        if self._last is not None:
            elapsed = now - self._last
            if elapsed < self._min_interval:
                self._sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


@dataclass
class HttpResult:
    """The outcome of one HTTP JSON request.

    ``headers`` and ``text_snippet`` are populated for error responses so a caller can
    report what the provider actually said (e.g. a rate-limit message and its
    ``Retry-After``) instead of inventing a reason. Both are additive: existing
    callers keep working unchanged.
    """

    ok: bool
    status_code: int | None
    data: Any = None
    error: str | None = None
    headers: dict[str, str] | None = None
    text_snippet: str | None = None

    @property
    def retry_after(self) -> float | None:
        """``Retry-After`` in seconds, when the provider sent one."""
        if not self.headers:
            return None
        raw = None
        for key, value in self.headers.items():
            if key.lower() == "retry-after":
                raw = value
                break
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpClient:
    """A GET-JSON client with timeout, retry with backoff, and rate limiting."""

    def __init__(
        self,
        transport: Transport,
        *,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._rate_limiter = rate_limiter or RateLimiter(0.0, sleep=sleep)
        self._sleep = sleep

        # Lightweight, cumulative instrumentation for the enrichment reports. A
        # "request" is one actual transport call (retries included).
        self.request_count = 0
        self.retry_count = 0
        self.network_error_count = 0
        self.http_error_count = 0
        #: Non-404 error statuses seen, so a report can say *why* a lookup failed.
        self.http_error_statuses: dict[str, int] = {}
        self.requests_by_host: dict[str, int] = {}

    def _note_request(self, url: str) -> None:
        self.request_count += 1
        host = urlparse(url).netloc or "unknown"
        self.requests_by_host[host] = self.requests_by_host.get(host, 0) + 1

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResult:
        """Perform a GET expecting JSON, retrying transient failures."""
        attempt = 0
        while True:
            self._rate_limiter.wait()
            self._note_request(url)
            try:
                response = self._transport.get(
                    url, params=params, headers=headers, timeout=self._timeout
                )
            except Exception as exc:  # network errors, timeouts, DNS failures
                if attempt < self._max_retries:
                    self.retry_count += 1
                    self._sleep(self._backoff_base * (2**attempt))
                    attempt += 1
                    continue
                self.network_error_count += 1
                return HttpResult(False, None, None, f"{type(exc).__name__}: {exc}")

            status = getattr(response, "status_code", None)
            if status == 200:
                try:
                    return HttpResult(True, 200, response.json())
                except Exception as exc:
                    return HttpResult(False, status, None, f"invalid JSON: {exc}")
            if status in _RETRYABLE_STATUS and attempt < self._max_retries:
                self.retry_count += 1
                self._sleep(self._backoff_base * (2**attempt))
                attempt += 1
                continue
            if status != 404:
                # 404 is a normal "not found" for a bibliographic lookup; any
                # other non-200 status is a genuine error worth reporting.
                self.http_error_count += 1
                key = str(status)
                self.http_error_statuses[key] = self.http_error_statuses.get(key, 0) + 1
            return HttpResult(
                False,
                status,
                None,
                f"HTTP {status}",
                headers=_response_headers(response),
                text_snippet=_response_text(response),
            )


def _response_headers(response: Any) -> dict[str, str] | None:
    """Headers of a response object, when it exposes them as a mapping."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return {str(key): str(value) for key, value in dict(headers).items()}
    except Exception:  # pragma: no cover - defensive: odd header containers
        return None


def _response_text(response: Any, limit: int = 400) -> str | None:
    """A short snippet of a response body, for error reporting."""
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()[:limit]
