"""TTL-aware cache for external discovery (RAG-010).

RAG-006's :class:`~app.metadata.external.cache.JsonCache` stores raw provider
responses in one JSON file; RAG-010 reuses it and adds what discovery needs:

* a **TTL**, so a repeated academic search does not unnecessarily hit the Internet
  again, while an explicit ``refresh`` deliberately bypasses a still-valid entry;
* a **cache key** that includes provider, normalized query/DOI, filters and a provider
  request version, so a change in how a request is built invalidates old entries;
* **hit/miss counters** for the discovery audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.metadata.doi import doi_cache_key, normalize_doi
from app.metadata.external.cache import JsonCache
from app.discovery.models import DiscoveryFilters

#: Bumped when a provider request or its normalization changes materially. This is the
#: single source of truth for cache invalidation (reported by ``DiscoveryCache.describe``).
PROVIDER_REQUEST_VERSION = "rag010-v2"


def normalize_query(query: str) -> str:
    """Normalize a discovery query for hashing and provider requests."""
    return " ".join((query or "").split()).strip()


def key_for_search(provider: str, query: str, limit: int, filters: DiscoveryFilters) -> str:
    """Cache key for one provider search."""
    return ":".join(
        (
            "discovery",
            provider,
            PROVIDER_REQUEST_VERSION,
            "search",
            normalize_query(query).casefold(),
            f"n={limit}",
            filters.cache_key_part(),
        )
    )


def key_for_doi(provider: str, doi: str) -> str:
    """Cache key for one provider DOI lookup."""
    canonical = normalize_doi(doi) or doi
    return ":".join(
        (
            "discovery",
            provider,
            PROVIDER_REQUEST_VERSION,
            "doi",
            doi_cache_key(canonical),
        )
    )


def key_for_title(provider: str, title: str, limit: int) -> str:
    """Cache key for one provider title lookup."""
    return ":".join(
        (
            "discovery",
            provider,
            PROVIDER_REQUEST_VERSION,
            "title",
            normalize_query(title).casefold(),
            f"n={limit}",
        )
    )


@dataclass
class CacheLookup:
    """The outcome of one cache read."""

    hit: bool
    value: Any = None
    age_seconds: float | None = None
    expired: bool = False


@dataclass
class DiscoveryCache:
    """A TTL cache over :class:`JsonCache`."""

    path: str | Path
    ttl_seconds: float = 7 * 24 * 3600
    enabled: bool = True
    _store: JsonCache | None = field(default=None, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)
    _now: datetime | None = field(default=None, init=False, repr=False)

    # --- introspection --------------------------------------------------------
    @property
    def store(self) -> JsonCache:
        """The underlying JSON store (created lazily)."""
        if self._store is None:
            self._store = JsonCache(self.path)
        return self._store

    @property
    def hits(self) -> int:
        """Number of cache hits (excluding refresh bypasses)."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @property
    def entry_count(self) -> int:
        """Number of stored entries."""
        return self.store.entry_count if self.enabled else 0

    def reset_counters(self) -> None:
        """Zero the hit/miss counters (called at the start of a request)."""
        self._hits = 0
        self._misses = 0

    # --- reads/writes ---------------------------------------------------------
    def get(self, key: str, *, refresh: bool = False) -> CacheLookup:
        """Read ``key``, honouring the TTL and an explicit refresh."""
        if not self.enabled:
            self._misses += 1
            return CacheLookup(False)
        if refresh:
            self._misses += 1
            return CacheLookup(False)
        age = self.store.age_seconds(key, now=self._now)
        value = self.store.get(key)
        if value is None or age is None:
            self._misses += 1
            return CacheLookup(False)
        if self.ttl_seconds >= 0 and age > self.ttl_seconds:
            self._misses += 1
            return CacheLookup(False, age_seconds=age, expired=True)
        self._hits += 1
        return CacheLookup(True, value=value, age_seconds=age)

    def set(self, key: str, value: Any) -> None:
        """Store a raw provider response."""
        if not self.enabled:
            return
        self.store.set(key, value)

    # --- reporting ------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        """Cache configuration for the report."""
        return {
            "path": str(self.path),
            "ttl_seconds": self.ttl_seconds,
            "enabled": self.enabled,
            "provider_request_version": PROVIDER_REQUEST_VERSION,
            "entries": self.entry_count,
        }
