"""Shared helpers for the RAG-010 discovery tests.

Everything here is offline: a queue-based fake transport answers Crossref/OpenAlex
requests, a fake binary fetcher answers full-text verification, and a synthetic local
corpus index stands in for the real Qdrant payloads. No test in the discovery suite
touches the Internet or the real index.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from app.discovery.cache import DiscoveryCache
from app.discovery.local import LocalCorpusIndex, build_index
from app.discovery.models import DiscoveryFilters, ExternalWork
from app.discovery.service import DiscoveryService
from app.discovery.fulltext import FetchResult
from app.metadata.external.http import HttpClient, RateLimiter


class FakeResponse:
    """A minimal HTTP response stand-in."""

    def __init__(
        self,
        status_code: int,
        data: Any = None,
        *,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._data = data
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(data if data is not None else {})

    def json(self) -> Any:
        if self._data is None:
            raise ValueError("no JSON body")
        return self._data


class QueuedTransport:
    """A transport that answers from a queue of responses (per call order)."""

    def __init__(self, responses: Sequence[FakeResponse | Exception]) -> None:
        self.queue: list[FakeResponse | Exception] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params=None, headers=None, timeout: float = 10.0) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self.queue:
            raise AssertionError(f"unexpected HTTP call to {url} (queue empty)")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def call_count(self) -> int:
        """Number of transport calls made."""
        return len(self.calls)


class RoutingTransport:
    """A transport that dispatches on a substring of the URL."""

    def __init__(self, handler: Callable[[str, dict | None], FakeResponse]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params=None, headers=None, timeout: float = 10.0) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.handler(url, params)


def make_http(transport: Any, *, max_retries: int = 0) -> HttpClient:
    """An :class:`HttpClient` that never sleeps and does not rate limit."""
    return HttpClient(
        transport,
        max_retries=max_retries,
        backoff_base=0.0,
        rate_limiter=RateLimiter(0.0, sleep=lambda _seconds: None),
        sleep=lambda _seconds: None,
    )


# --- provider payloads -------------------------------------------------------


def crossref_message(
    *,
    doi: str = "10.1000/xyz",
    title: str = "A Verified Title",
    authors: Sequence[tuple[str, str]] | None = None,
    year: int = 2019,
    type_: str = "journal-article",
    venue: str = "Journal of Testing",
    publisher: str = "Test Press",
    abstract: str | None = None,
    license_: str | None = None,
    pdf_link: str | None = None,
) -> dict[str, Any]:
    """Build a Crossref ``message`` payload."""
    author_list = list(authors) if authors is not None else [("Jane", "Doe")]
    message: dict[str, Any] = {
        "DOI": doi,
        "title": [title],
        "author": [{"given": given, "family": family} for given, family in author_list],
        "issued": {"date-parts": [[year, 1, 1]]},
        "type": type_,
        "container-title": [venue],
        "publisher": publisher,
        "URL": f"https://doi.org/{doi}",
    }
    if abstract is not None:
        message["abstract"] = abstract
    if license_ is not None:
        message["license"] = [{"URL": license_}]
    if pdf_link is not None:
        message["link"] = [{"URL": pdf_link, "content-type": "application/pdf"}]
    return message


def crossref_search_payload(messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """A Crossref ``/works`` search body."""
    return {"status": "ok", "message": {"items": list(messages)}}


def openalex_work(
    *,
    doi: str | None = "10.1000/xyz",
    title: str = "A Verified Title",
    authors: Sequence[str] | None = None,
    year: int = 2019,
    type_: str = "article",
    venue: str = "Journal of Testing",
    publisher: str = "Test Press",
    abstract_words: Sequence[str] | None = None,
    is_oa: bool | None = True,
    pdf_url: str | None = "https://example.org/paper.pdf",
    landing_page: str | None = "https://example.org/landing",
    cited_by: int | None = 12,
    work_id: str = "https://openalex.org/W123",
) -> dict[str, Any]:
    """Build an OpenAlex work payload."""
    inverted: dict[str, list[int]] | None = None
    if abstract_words is not None:
        inverted = {word: [index] for index, word in enumerate(abstract_words)}
    return {
        "id": work_id,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title,
        "display_name": title,
        "publication_year": year,
        "type": type_,
        "cited_by_count": cited_by,
        "authorships": [
            {"author": {"display_name": name}} for name in (authors or ["Jane Doe"])
        ],
        "primary_location": {
            "source": {
                "display_name": venue,
                "host_organization_name": publisher,
            },
            "landing_page_url": landing_page,
            "pdf_url": None,
        },
        "best_oa_location": {"pdf_url": pdf_url, "landing_page_url": landing_page}
        if is_oa
        else None,
        "open_access": {"is_oa": is_oa, "oa_status": "gold" if is_oa else "closed"},
        "abstract_inverted_index": inverted,
    }


def openalex_search_payload(works: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """An OpenAlex ``/works`` search body."""
    items = list(works)
    return {"meta": {"count": len(items)}, "results": items}


# --- local corpus ------------------------------------------------------------


def local_payload(
    *,
    document_id: str = "doc-1",
    filename: str = "paper.pdf",
    file_hash: str = "hash-1",
    title: str | None = "A Verified Title",
    doi: str | None = None,
    year: int | None = 2019,
    first_author: str | None = "Jane Doe",
    corpus_role: str = "literature",
    evidence_eligible: bool = True,
) -> dict[str, Any]:
    """One document-level payload, shaped like the RAG-008 index payload."""
    return {
        "document_id": document_id,
        "filename": filename,
        "file_hash": file_hash,
        "title": title,
        "doi": doi,
        "year": year,
        "first_author": first_author,
        "corpus_role": corpus_role,
        "evidence_eligible": evidence_eligible,
    }


def make_local_index(payloads: Sequence[dict[str, Any]]) -> LocalCorpusIndex:
    """Build a local corpus index from synthetic payloads."""
    return build_index(payloads)


def local_index_with(**kwargs: Any) -> LocalCorpusIndex:
    """A one-document local corpus index built from keyword payload fields."""
    return build_index([local_payload(**kwargs)])


# --- service wiring ----------------------------------------------------------


def disabled_cache() -> DiscoveryCache:
    """A cache that touches no disk, so tests never read or write a real cache file."""
    return DiscoveryCache(
        path=Path(tempfile.gettempdir()) / "phd-rag-discovery-tests-unused" / "cache.json",
        enabled=False,
    )


def build_service(
    transport: Any,
    *,
    cache: DiscoveryCache | None = None,
    local_index: LocalCorpusIndex | None = None,
    provider_names: Sequence[str] = ("crossref", "openalex"),
    settings: Any = None,
    max_retries: int = 0,
) -> DiscoveryService:
    """Build a discovery service wired to a fake transport and a synthetic corpus.

    When no cache is supplied a *disabled* one is used, so a stray entry from a real run
    can never leak into a test (and no test writes to the repository).
    """
    http = make_http(transport, max_retries=max_retries)
    return DiscoveryService(
        providers=None,
        http=http,
        cache=cache if cache is not None else disabled_cache(),
        local_index=local_index if local_index is not None else build_index([]),
        settings=settings,
        provider_names=provider_names,
    )


def make_cache(tmp_path: Path, *, ttl: float = 3600.0, enabled: bool = True) -> DiscoveryCache:
    """A file-backed discovery cache in a temporary directory."""
    return DiscoveryCache(tmp_path / "cache.json", ttl_seconds=ttl, enabled=enabled)


class FakeFetcher:
    """A binary fetcher answering from a URL -> ``FetchResult`` map."""

    def __init__(self, responses: dict[str, FetchResult | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def fetch(
        self, url: str, *, headers: dict[str, str], timeout: float, max_bytes: int
    ) -> FetchResult:
        self.calls.append(
            {"url": url, "headers": headers, "timeout": timeout, "max_bytes": max_bytes}
        )
        item = self.responses.get(url)
        if item is None:
            raise AssertionError(f"unexpected fetch of {url}")
        if isinstance(item, Exception):
            raise item
        return item


def fetch_result(
    *,
    url: str = "https://example.org/paper.pdf",
    body: bytes = b"%PDF-1.4 test",
    status_code: int = 200,
    content_type: str = "application/pdf",
    headers: dict[str, str] | None = None,
) -> FetchResult:
    """A successful download."""
    merged = {"Content-Type": content_type}
    merged.update(headers or {})
    return FetchResult(
        status_code=status_code,
        url=url,
        headers=merged,
        content=body,
        content_type=content_type,
    )


def work_with_local(**kwargs: Any) -> ExternalWork:
    """An external record built from keyword fields (test convenience)."""
    return ExternalWork(**kwargs)


#: A stable filters object for cache-key tests.
DEFAULT_FILTERS = DiscoveryFilters()
