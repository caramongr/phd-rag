"""The provider interface every discovery source implements (RAG-010).

The service never imports a provider's response schema: providers adapt their own
payloads into :class:`~app.discovery.models.ExternalWork` and report failure instead of
raising. That is what keeps a provider outage from breaking discovery (§11, §28) and
keeps provider-specific logic out of the rest of the application (§3).

Conceptually the brief's interface is ``search(query, filters)``,
``lookup_doi(doi)`` and ``provider_name``; the only difference here is that both
operations return a :class:`ProviderResponse` so the *reason* a provider produced
nothing (no results vs. HTTP 429) travels with the records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.discovery.models import DiscoveryFilters, ExternalWork, ProviderFailure


@dataclass
class ProviderResponse:
    """What one provider call produced."""

    provider: str
    operation: str
    query: str
    records: list[ExternalWork] = field(default_factory=list)
    cache_hit: bool = False
    http_requests: int = 0
    seconds: float = 0.0
    failure: ProviderFailure | None = None

    @property
    def ok(self) -> bool:
        """Whether the provider answered (even with zero results)."""
        return self.failure is None

    @property
    def count(self) -> int:
        """Number of records returned."""
        return len(self.records)


@runtime_checkable
class AcademicDiscoveryProvider(Protocol):
    """A source of external scholarly records."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier used in cache keys and provenance."""

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: DiscoveryFilters | None = None,
    ) -> ProviderResponse:
        """Bibliographic search."""

    def lookup_doi(self, doi: str) -> ProviderResponse:
        """Look up one DOI."""

    def lookup_title(self, title: str, *, limit: int = 5) -> ProviderResponse:
        """Look up one title."""


#: Retryable statuses already handled by the shared HTTP client; 429 is included
#: because backoff may recover, but the final outcome is still reported as a failure.
QUOTA_STATUS = 429


def failure_from_http(provider: str, operation: str, result: Any) -> ProviderFailure:
    """Build a :class:`ProviderFailure` from an ``HttpResult``.

    The provider's own words are preserved (a rate-limit message, a ``Retry-After``),
    and the wording stays factual: a 429 is the provider's request limit or budget for
    this client, never a charge to the user.
    """
    status = getattr(result, "status_code", None)
    retry_after = None
    if hasattr(result, "retry_after"):
        retry_after = result.retry_after
    snippet = getattr(result, "text_snippet", None) or ""
    if status == QUOTA_STATUS:
        message = (
            "provider rate limit reached (HTTP 429): "
            + (snippet[:280] or "the provider refused the request")
            + " | no results were fabricated"
        )
        return ProviderFailure(
            provider=provider,
            kind="rate_limited",
            status_code=status,
            message=message,
            retry_after=retry_after,
            counted_as_quota=True,
        )
    message = snippet[:280] or str(getattr(result, "error", "") or "request failed")
    return ProviderFailure(
        provider=provider,
        kind="http_error" if status else "network_error",
        status_code=status,
        message=message,
        retry_after=retry_after,
    )


def record_missing_fields(work: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    """Field names absent from a provider record (never guessed)."""
    return tuple(name for name in fields if work.get(name) in (None, "", [], {}))
