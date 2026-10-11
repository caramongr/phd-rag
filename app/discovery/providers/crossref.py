"""Crossref discovery adapter (RAG-010).

Crossref is the DOI registration agency, so it is authoritative for DOI identity and
for the published metadata it holds. Search uses ``query.bibliographic`` (relevance) or
``sort=published`` (recent), all requests go through the shared RAG-006 HTTP client
(timeout, bounded retry with backoff, rate limiting) and every raw response is cached
before it is normalized.

Fields Crossref does not state are left empty: there is no open-access flag and no
citation count in this API, so those stay ``None`` rather than being inferred.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from app.discovery.cache import DiscoveryCache, key_for_doi, key_for_search, key_for_title, normalize_query
from app.discovery.models import (
    DiscoveryFilters,
    ExternalWork,
    FulltextStatus,
    ProviderFailure,
)
from app.discovery.providers.base import ProviderResponse, failure_from_http
from app.metadata.doi import normalize_doi
from app.metadata.external.http import HttpClient
from app.metadata.external.hygiene import clean_external_text
from app.metadata.text import collapse_whitespace

PROVIDER_NAME = "crossref"
BASE_URL = "https://api.crossref.org"

#: Fields requested from Crossref; keeps responses small and deterministic.
_SELECT = (
    "DOI,title,author,issued,published-print,published-online,container-title,event,"
    "publisher,type,URL,abstract,is-referenced-by-count,license,link,subject"
)

#: Crossref type -> the coarse publication type used across discovery.
_TYPE_MAP = {
    "journal-article": "journal-article",
    "proceedings-article": "conference-paper",
    "book": "book",
    "monograph": "book",
    "edited-book": "book",
    "book-chapter": "book-chapter",
    "reference-entry": "book-chapter",
    "dissertation": "thesis",
    "report": "report",
    "standard": "report",
    "posted-content": "preprint",
    "dataset": "dataset",
    "peer-review": "peer-review",
    "component": "unknown",
    "proceedings": "conference-paper",
}

#: License URLs that clearly indicate an open licence.
_OPEN_LICENCE_HINTS = ("creativecommons.org", "openaccess", "cc-by", "cc0")


def _default_headers(mailto: str | None) -> dict[str, str]:
    """Identify this client to Crossref's polite pool when a contact exists."""
    agent = "phd-rag/0.1 (academic discovery)"
    if mailto:
        agent = f"{agent} mailto:{mailto}"
    return {"User-Agent": agent, "Accept": "application/json"}


def _year_from_message(message: dict[str, Any]) -> int | None:
    for field_name in ("issued", "published-print", "published-online", "created"):
        parts = (message.get(field_name) or {}).get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            year = parts[0][0]
            if isinstance(year, int):
                return year
    return None


def _authors_from_message(message: dict[str, Any]) -> tuple[str, ...]:
    authors: list[str] = []
    for item in message.get("author") or []:
        if not isinstance(item, dict):
            continue
        given = clean_external_text(item.get("given"))
        family = clean_external_text(item.get("family"))
        name = clean_external_text(item.get("name"))
        full = collapse_whitespace(" ".join(part for part in (given, family) if part))
        if full or name:
            authors.append(full or str(name))
    return tuple(authors)


def _venue_from_message(message: dict[str, Any]) -> str | None:
    container = message.get("container-title") or []
    if isinstance(container, list) and container:
        cleaned = clean_external_text(container[0])
        if cleaned:
            return cleaned
    event = message.get("event") or {}
    if isinstance(event, dict):
        return clean_external_text(event.get("name"))
    return None


def _is_pdf_link(item: dict[str, Any]) -> bool:
    """Whether a Crossref ``link`` entry points at a PDF.

    Publishers are inconsistent: some declare ``application/pdf``, others leave the
    content type as ``unspecified`` while the URL plainly points at a PDF file
    (``…/paper.pdf``) or at a PDF route (``…/296/pdf``, the MDPI convention). Both
    count; an XML/plain-text TDM endpoint does not.
    """
    if "pdf" in str(item.get("content-type") or "").lower():
        return True
    path = urlparse(str(item.get("URL") or "")).path.lower()
    last_segment = path.rsplit("/", 1)[-1]
    return last_segment.endswith(".pdf") or last_segment == "pdf"


def _fulltext_from_message(message: dict[str, Any]) -> tuple[str | None, FulltextStatus]:
    """Return a full-text URL only when Crossref marks it as openly licensed.

    Crossref ``link`` entries point at the publisher file; without an open licence we
    cannot claim the full text is available, so the record stays landing-page only.
    """
    licences = [
        str(item.get("URL", ""))
        for item in (message.get("license") or [])
        if isinstance(item, dict)
    ]
    is_open = any(hint in licence.lower() for licence in licences for hint in _OPEN_LICENCE_HINTS)
    links = [item for item in (message.get("link") or []) if isinstance(item, dict)]
    pdf_links = [
        collapse_whitespace(str(item.get("URL") or ""))
        for item in links
        if _is_pdf_link(item)
    ]
    if is_open and pdf_links:
        return pdf_links[0] or None, FulltextStatus.FULLTEXT_OPEN
    if licences or links:
        return pdf_links[0] if pdf_links else None, FulltextStatus.LANDING_PAGE_ONLY
    return None, FulltextStatus.UNKNOWN


def work_from_message(message: dict[str, Any]) -> ExternalWork:
    """Normalize one Crossref work into an :class:`ExternalWork`."""
    titles = message.get("title") or []
    raw_title = titles[0] if isinstance(titles, list) and titles else None
    title = clean_external_text(raw_title)
    venue = _venue_from_message(message)
    publisher = clean_external_text(message.get("publisher"))
    doi = normalize_doi(str(message.get("DOI") or ""))
    abstract = clean_external_text(message.get("abstract"))
    fulltext_url, fulltext_status = _fulltext_from_message(message)
    year = _year_from_message(message)
    authors = _authors_from_message(message)
    cited = message.get("is-referenced-by-count")
    cited_by = int(cited) if isinstance(cited, int) else None

    missing = tuple(
        name
        for name, value in (
            ("title", title),
            ("authors", authors),
            ("year", year),
            ("doi", doi),
            ("venue", venue),
            ("abstract", abstract),
        )
        if value in (None, "", ())
    )
    provenance = {
        field: (PROVIDER_NAME,)
        for field, value in (
            ("title", title),
            ("authors", authors),
            ("year", year),
            ("doi", doi),
            ("venue", venue),
            ("publisher", publisher),
            ("abstract", abstract),
            ("cited_by_count", cited_by),
            ("publication_type", _TYPE_MAP.get(str(message.get("type") or "").casefold())),
        )
        if value not in (None, "", ())
    }
    return ExternalWork(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=collapse_whitespace(str(message.get("URL") or "")) or None,
        venue=venue,
        publisher=publisher,
        publication_type=_TYPE_MAP.get(str(message.get("type") or "").casefold()),
        abstract=abstract,
        cited_by_count=cited_by,
        open_access=None,
        open_access_status=None,
        fulltext_url=fulltext_url,
        fulltext_status=fulltext_status,
        providers=(PROVIDER_NAME,),
        raw_provider_ids=({PROVIDER_NAME: doi} if doi else {}),
        field_provenance=provenance,
        source_provenance=(f"crossref:{message.get('type') or 'unknown'}",),
        missing_fields=missing,
    )


class CrossrefProvider:
    """Crossref search and DOI/title lookup."""

    provider_name = PROVIDER_NAME

    def __init__(
        self,
        http: HttpClient,
        *,
        cache: DiscoveryCache | None = None,
        base_url: str = BASE_URL,
        mailto: str | None = None,
    ) -> None:
        self._http = http
        self._cache = cache
        self._base_url = base_url.rstrip("/")
        self._headers = _default_headers(mailto)
        self.mailto = mailto

    # --- helpers --------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> tuple[Any | None, ProviderFailure | None, int]:
        result = self._http.get_json(f"{self._base_url}/{path}", params=params, headers=self._headers)
        if not result.ok:
            return None, failure_from_http(PROVIDER_NAME, path, result), 1
        return result.data, None, 1

    def _search_params(
        self, query: str, limit: int, filters: DiscoveryFilters | None
    ) -> dict[str, Any]:
        filters = filters or DiscoveryFilters()
        params: dict[str, Any] = {
            "query.bibliographic": normalize_query(query),
            "rows": max(1, min(limit, 100)),
            "select": _SELECT,
        }
        if filters.author:
            params["query.author"] = filters.author
        filter_parts: list[str] = []
        if filters.year_min is not None:
            filter_parts.append(f"from-pub-date:{filters.year_min}-01-01")
        if filters.year_max is not None:
            filter_parts.append(f"until-pub-date:{filters.year_max}-12-31")
        if filters.publication_types:
            filter_parts.append(f"type:{','.join(filters.publication_types)}")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        if filters.sort == "recent":
            params["sort"] = "published"
            params["order"] = "desc"
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    # --- interface ------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: DiscoveryFilters | None = None,
    ) -> ProviderResponse:
        """Bibliographic search over Crossref ``/works``."""
        started = time.perf_counter()
        normalized = normalize_query(query)
        if not normalized:
            return ProviderResponse(
                PROVIDER_NAME, "search", query, failure=ProviderFailure(
                    provider=PROVIDER_NAME, kind="invalid_request", message="empty query"
                )
            )
        cache_key = key_for_search(PROVIDER_NAME, normalized, limit, filters or DiscoveryFilters())
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None and cached.hit:
            records = [work_from_message(item) for item in (cached.value or {}).get("message", {}).get("items", [])]
            return ProviderResponse(
                PROVIDER_NAME,
                "search",
                normalized,
                records=records,
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        data, failure, requests = self._get("works", self._search_params(normalized, limit, filters))
        if failure is not None:
            return ProviderResponse(
                PROVIDER_NAME,
                "search",
                normalized,
                http_requests=requests,
                seconds=time.perf_counter() - started,
                failure=failure,
            )
        items = ((data or {}).get("message") or {}).get("items") or []
        records = [work_from_message(item) for item in items if isinstance(item, dict)]
        if self._cache:
            self._cache.set(cache_key, data)
        return ProviderResponse(
            PROVIDER_NAME,
            "search",
            normalized,
            records=records,
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )

    def lookup_doi(self, doi: str) -> ProviderResponse:
        """Look up a single DOI."""
        started = time.perf_counter()
        canonical = normalize_doi(doi)
        if canonical is None:
            return ProviderResponse(
                PROVIDER_NAME, "doi", doi, failure=ProviderFailure(
                    provider=PROVIDER_NAME, kind="invalid_request", message=f"not a DOI: {doi!r}"
                )
            )
        cache_key = key_for_doi(PROVIDER_NAME, canonical)
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None and cached.hit:
            message = (cached.value or {}).get("message")
            records = [work_from_message(message)] if isinstance(message, dict) else []
            return ProviderResponse(
                PROVIDER_NAME,
                "doi",
                canonical,
                records=records,
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        # Crossref's single-work route rejects ``select`` ("this route does not support
        # select"), so the parameter list stays empty here.
        data, failure, requests = self._get(f"works/{canonical}", {})
        if failure is not None:
            if failure.status_code == 404:
                failure = ProviderFailure(
                    provider=PROVIDER_NAME,
                    kind="not_found",
                    status_code=404,
                    message=f"Crossref has no work with DOI {canonical}",
                )
            return ProviderResponse(
                PROVIDER_NAME,
                "doi",
                canonical,
                http_requests=requests,
                seconds=time.perf_counter() - started,
                failure=failure,
            )
        message = (data or {}).get("message")
        records = [work_from_message(message)] if isinstance(message, dict) else []
        if self._cache:
            self._cache.set(cache_key, data)
        return ProviderResponse(
            PROVIDER_NAME,
            "doi",
            canonical,
            records=records,
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )

    def lookup_title(self, title: str, *, limit: int = 5) -> ProviderResponse:
        """Look up one title bibliographically."""
        started = time.perf_counter()
        normalized = normalize_query(title)
        if not normalized:
            return ProviderResponse(
                PROVIDER_NAME, "title", title, failure=ProviderFailure(
                    provider=PROVIDER_NAME, kind="invalid_request", message="empty title"
                )
            )
        cache_key = key_for_title(PROVIDER_NAME, normalized, limit)
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None and cached.hit:
            items = ((cached.value or {}).get("message") or {}).get("items") or []
            return ProviderResponse(
                PROVIDER_NAME,
                "title",
                normalized,
                records=[work_from_message(item) for item in items if isinstance(item, dict)],
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        data, failure, requests = self._get(
            "works",
            {
                "query.bibliographic": normalized,
                "rows": max(1, min(limit, 100)),
                "select": _SELECT,
            },
        )
        if failure is not None:
            return ProviderResponse(
                PROVIDER_NAME,
                "title",
                normalized,
                http_requests=requests,
                seconds=time.perf_counter() - started,
                failure=failure,
            )
        items = ((data or {}).get("message") or {}).get("items") or []
        if self._cache:
            self._cache.set(cache_key, data)
        return ProviderResponse(
            PROVIDER_NAME,
            "title",
            normalized,
            records=[work_from_message(item) for item in items if isinstance(item, dict)],
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )
