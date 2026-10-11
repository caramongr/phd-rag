"""Semantic Scholar adapter (optional, RAG-010).

Semantic Scholar's public API now requires a key for most traffic and rate-limits
anonymous callers aggressively, so this adapter is **optional**: it is only built when
explicitly requested, and a 429 or a missing key degrades to a normal provider failure
that the rest of discovery reports and continues past. No credentials are invented and
none are required for RAG-010 to work.

Reference data for the endpoint shapes used here:
``GET /paper/search?query=&limit=&year=&fields=`` and ``GET /paper/DOI:<doi>``.
"""

from __future__ import annotations

import time
from typing import Any

from app.discovery.cache import DiscoveryCache, key_for_doi, key_for_search, normalize_query
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

PROVIDER_NAME = "semanticscholar"
BASE_URL = "https://api.semanticscholar.org/graph/v1"

_FIELDS = (
    "title,abstract,year,venue,publicationVenue,externalIds,authors,"
    "citationCount,openAccessPdf,publicationTypes,url"
)

_TYPE_MAP = {
    "journalarticle": "journal-article",
    "review": "journal-article",
    "conference": "conference-paper",
    "book": "book",
    "booksection": "book-chapter",
    "thesis": "thesis",
    "report": "report",
    "dataset": "dataset",
}

#: The public endpoint answers 429 for most anonymous traffic; the adapter reports
#: that as a failure rather than pretending there are no results.
_QUOTA_HINT = "rate limit"


def work_from_paper(paper: dict[str, Any]) -> ExternalWork:
    """Normalize one Semantic Scholar paper into an :class:`ExternalWork`."""
    title = clean_external_text(paper.get("title"))
    external = paper.get("externalIds") or {}
    external = external if isinstance(external, dict) else {}
    doi = normalize_doi(str(external.get("DOI") or ""))
    year = paper.get("year")
    year = int(year) if isinstance(year, int) else None
    authors: list[str] = []
    for item in paper.get("authors") or []:
        if not isinstance(item, dict):
            continue
        name = clean_external_text(item.get("name"))
        if name:
            authors.append(name)
    venue = clean_external_text(paper.get("venue"))
    if not venue:
        publication_venue = paper.get("publicationVenue") or {}
        if isinstance(publication_venue, dict):
            venue = clean_external_text(publication_venue.get("name"))
    pdf = paper.get("openAccessPdf") or {}
    pdf_url = None
    if isinstance(pdf, dict):
        pdf_url = collapse_whitespace(str(pdf.get("url") or "")) or None
    cited = paper.get("citationCount")
    cited_by = int(cited) if isinstance(cited, int) else None
    types = paper.get("publicationTypes") or []
    publication_type = None
    if isinstance(types, list) and types:
        publication_type = _TYPE_MAP.get(str(types[0]).replace(" ", "").casefold())

    if pdf_url:
        fulltext_url, fulltext_status = pdf_url, FulltextStatus.FULLTEXT_OPEN
    else:
        fulltext_url, fulltext_status = None, FulltextStatus.LANDING_PAGE_ONLY

    missing = tuple(
        name
        for name, value in (
            ("title", title),
            ("authors", tuple(authors)),
            ("year", year),
            ("doi", doi),
            ("venue", venue),
            ("abstract", clean_external_text(paper.get("abstract"))),
        )
        if value in (None, "", ())
    )
    provenance = {
        field: (PROVIDER_NAME,)
        for field, value in (
            ("title", title),
            ("authors", tuple(authors)),
            ("year", year),
            ("doi", doi),
            ("venue", venue),
            ("abstract", clean_external_text(paper.get("abstract"))),
            ("cited_by_count", cited_by),
            ("fulltext_url", fulltext_url),
            ("publication_type", publication_type),
        )
        if value not in (None, "", ())
    }
    raw_ids: dict[str, str] = {}
    paper_id = paper.get("paperId")
    if isinstance(paper_id, str) and paper_id:
        raw_ids[PROVIDER_NAME] = paper_id
    return ExternalWork(
        title=title,
        authors=tuple(authors),
        year=year,
        doi=doi,
        url=collapse_whitespace(str(paper.get("url") or "")) or None,
        venue=venue,
        publisher=None,
        publication_type=publication_type,
        abstract=clean_external_text(paper.get("abstract")),
        cited_by_count=cited_by,
        open_access=True if pdf_url else None,
        fulltext_url=fulltext_url,
        fulltext_status=fulltext_status,
        providers=(PROVIDER_NAME,),
        raw_provider_ids=raw_ids,
        field_provenance=provenance,
        source_provenance=(f"semanticscholar:{publication_type or 'unknown'}",),
        missing_fields=missing,
    )


class SemanticScholarProvider:
    """Optional Semantic Scholar search and DOI lookup."""

    provider_name = PROVIDER_NAME

    def __init__(
        self,
        http: HttpClient,
        *,
        cache: DiscoveryCache | None = None,
        base_url: str = BASE_URL,
        api_key: str | None = None,
    ) -> None:
        self._http = http
        self._cache = cache
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._headers = {"User-Agent": "phd-rag/0.1 (academic discovery)"}

    def _get(self, path: str, params: dict[str, Any]) -> tuple[Any | None, ProviderFailure | None, int]:
        result = self._http.get_json(f"{self._base_url}/{path}", params=params, headers=self._headers)
        if not result.ok:
            failure = failure_from_http(PROVIDER_NAME, path, result)
            if failure.status_code == 404:
                failure = ProviderFailure(
                    provider=PROVIDER_NAME,
                    kind="not_found",
                    status_code=404,
                    message="Semantic Scholar has no record for this identifier",
                )
            return None, failure, 1
        return result.data, None, 1

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: DiscoveryFilters | None = None,
    ) -> ProviderResponse:
        """Search Semantic Scholar."""
        started = time.perf_counter()
        normalized = normalize_query(query)
        if not normalized:
            return ProviderResponse(
                PROVIDER_NAME, "search", query, failure=ProviderFailure(
                    provider=PROVIDER_NAME, kind="invalid_request", message="empty query"
                )
            )
        filters = filters or DiscoveryFilters()
        cache_key = key_for_search(PROVIDER_NAME, normalized, limit, filters)
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None and cached.hit:
            return ProviderResponse(
                PROVIDER_NAME,
                "search",
                normalized,
                records=[
                    work_from_paper(item)
                    for item in ((cached.value or {}).get("data") or [])
                    if isinstance(item, dict)
                ],
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        params: dict[str, Any] = {"query": normalized, "limit": max(1, min(limit, 100)), "fields": _FIELDS}
        if filters.year_min is not None or filters.year_max is not None:
            params["year"] = f"{filters.year_min or ''}-{filters.year_max or ''}"
        data, failure, requests = self._get("paper/search", params)
        if failure is not None:
            return ProviderResponse(
                PROVIDER_NAME,
                "search",
                normalized,
                http_requests=requests,
                seconds=time.perf_counter() - started,
                failure=failure,
            )
        if self._cache:
            self._cache.set(cache_key, data)
        return ProviderResponse(
            PROVIDER_NAME,
            "search",
            normalized,
            records=[
                work_from_paper(item)
                for item in ((data or {}).get("data") or [])
                if isinstance(item, dict)
            ],
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )

    def lookup_doi(self, doi: str) -> ProviderResponse:
        """Look up one DOI."""
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
            paper = cached.value if isinstance(cached.value, dict) else None
            return ProviderResponse(
                PROVIDER_NAME,
                "doi",
                canonical,
                records=[work_from_paper(paper)] if paper else [],
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        data, failure, requests = self._get(f"paper/DOI:{canonical}", {"fields": _FIELDS})
        if failure is not None:
            return ProviderResponse(
                PROVIDER_NAME,
                "doi",
                canonical,
                http_requests=requests,
                seconds=time.perf_counter() - started,
                failure=failure,
            )
        if self._cache:
            self._cache.set(cache_key, data)
        paper = data if isinstance(data, dict) else None
        return ProviderResponse(
            PROVIDER_NAME,
            "doi",
            canonical,
            records=[work_from_paper(paper)] if paper else [],
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )

    def lookup_title(self, title: str, *, limit: int = 5) -> ProviderResponse:
        """Title lookup is a search with a small limit."""
        return self.search(title, limit=limit)
