"""Cache behaviour tests (RAG-010 §30 items 19-21, §14)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.discovery.cache import (
    DiscoveryCache,
    key_for_doi,
    key_for_search,
    key_for_title,
    normalize_query,
)
from app.discovery.models import DiscoveryFilters
from app.discovery.providers.crossref import CrossrefProvider
from tests.discovery_support import (
    FakeResponse,
    QueuedTransport,
    RoutingTransport,
    crossref_message,
    crossref_search_payload,
    make_cache,
    make_http,
)


# --- 19. cache hit avoids an HTTP request -----------------------------------


def test_cache_hit_avoids_a_second_http_request(tmp_path: Path) -> None:
    """The second identical search is answered from the cache."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(
            200, crossref_search_payload([crossref_message(doi="10.1234/a")])
        )
    )
    cache = make_cache(tmp_path)
    provider = CrossrefProvider(make_http(transport), cache=cache)

    first = provider.search("ai governance")
    second = provider.search("ai governance")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert transport.calls.__len__() == 1
    assert len(second.records) == 1
    assert second.records[0].doi == "10.1234/a"
    assert cache.hits >= 1


def test_cache_survives_a_new_provider_instance(tmp_path: Path) -> None:
    """The cache is on disk, so a later process also benefits."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, crossref_search_payload([crossref_message()]))
    )
    cache = make_cache(tmp_path)
    CrossrefProvider(make_http(transport), cache=cache).search("ai")

    second_cache = make_cache(tmp_path)
    provider = CrossrefProvider(make_http(QueuedTransport([])), cache=second_cache)
    response = provider.search("ai")

    assert response.cache_hit is True
    assert len(response.records) == 1


# --- 20. refresh bypasses the cache -----------------------------------------


def test_refresh_bypasses_a_valid_cache_entry(tmp_path: Path) -> None:
    """``refresh`` deliberately re-requests, even with a fresh entry."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, crossref_search_payload([crossref_message()]))
    )
    cache = make_cache(tmp_path)
    provider = CrossrefProvider(make_http(transport), cache=cache)
    provider.search("ai")

    key = key_for_search("crossref", "ai", 20, DiscoveryFilters())
    bypassed = cache.get(key, refresh=True)
    provider.search("ai", limit=20)

    assert bypassed.hit is False


def test_expired_entry_is_not_used(tmp_path: Path) -> None:
    """A TTL of zero means the next read misses."""
    cache = DiscoveryCache(tmp_path / "cache.json", ttl_seconds=0.0)
    cache.set("k", {"value": 1})
    time.sleep(0.01)

    assert cache.get("k").hit is False
    assert cache.get("k").expired is True


def test_disabled_cache_never_reads_or_writes(tmp_path: Path) -> None:
    """A disabled cache always misses and stores nothing."""
    cache = DiscoveryCache(tmp_path / "cache.json", enabled=False)
    cache.set("k", {"value": 1})

    assert cache.get("k").hit is False
    assert cache.entry_count == 0
    assert cache.describe()["enabled"] is False


# --- 21. provider/filter differences create different keys ------------------


def test_provider_differences_create_different_keys() -> None:
    """Two providers never share a cache entry."""
    filters = DiscoveryFilters()
    assert key_for_search("crossref", "ai", 20, filters) != key_for_search(
        "openalex", "ai", 20, filters
    )


def test_filter_differences_create_different_keys() -> None:
    """Different filters are different requests."""
    base = key_for_search("crossref", "ai", 20, DiscoveryFilters())
    year = key_for_search("crossref", "ai", 20, DiscoveryFilters(year_min=2025))
    author = key_for_search("crossref", "ai", 20, DiscoveryFilters(author="Doe"))
    limit = key_for_search("crossref", "ai", 5, DiscoveryFilters())
    sort = key_for_search("crossref", "ai", 20, DiscoveryFilters(sort="recent"))

    assert len({base, year, author, limit, sort}) == 5


def test_query_whitespace_and_case_do_not_fragment_the_cache() -> None:
    """A re-typed query with different spacing is the same cache entry."""
    filters = DiscoveryFilters()
    assert key_for_search("crossref", "  AI   Governance ", 20, filters) == key_for_search(
        "crossref", "ai governance", 20, filters
    )
    assert normalize_query("  AI   Governance ") == "AI Governance"


def test_doi_keys_ignore_wrapper_and_case() -> None:
    """Equivalent DOI spellings share one cache key."""
    assert key_for_doi("crossref", "10.1234/ABC") == key_for_doi(
        "crossref", "https://doi.org/10.1234/abc"
    )


def test_title_keys_are_stable() -> None:
    """Title lookups have their own key space."""
    assert key_for_title("crossref", " A Title ", 5) == key_for_title(
        "crossref", "a title", 5
    )


def test_request_version_is_part_of_every_key() -> None:
    """A provider-request-version bump invalidates old entries."""
    from app.discovery import cache as cache_module

    key = key_for_search("crossref", "ai", 20, DiscoveryFilters())
    assert cache_module.PROVIDER_REQUEST_VERSION in key


def test_cache_file_is_readable_json(tmp_path: Path) -> None:
    """The cache remains an inspectable JSON file (no opaque store)."""
    cache = make_cache(tmp_path)
    cache.set("k", {"message": {"items": []}})

    payload = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert payload, "cache file should not be empty"


def test_counters_reset_per_request(tmp_path: Path) -> None:
    """The audit reports this request's hits and misses, not a lifetime total."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, crossref_search_payload([]))
    )
    cache = make_cache(tmp_path)
    provider = CrossrefProvider(make_http(transport), cache=cache)
    provider.search("ai")

    cache.reset_counters()

    assert cache.hits == 0
    assert cache.misses == 0
