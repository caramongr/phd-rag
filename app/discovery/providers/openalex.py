"""OpenAlex discovery adapter (RAG-010).

OpenAlex is the richer source here: it states open-access status, citation counts and
an abstract (as an inverted index), and it resolves DOIs to a work id. It is also the
provider that rate-limits hardest - RAG-006 already saw HTTP 429 on anonymous access -
so this adapter:

* sends ``mailto`` when a contact identity is configured, and ``api_key`` when
  ``PHD_RAG_OPENALEX_API_KEY`` is set (spec §11),
* reports a 429 as a :class:`ProviderFailure` with the provider's own message and its
  ``Retry-After``, and never fabricates a result,
* leaves fields OpenAlex does not state empty instead of inferring them.
"""

from __future__ import annotations

import time
from typing import Any

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

PROVIDER_NAME = "openalex"
BASE_URL = "https://api.openalex.org"

#: Coarse publication types shared across discovery.
_TYPE_MAP = {
    "article": "journal-article",
    "review": "journal-article",
    "editorial": "journal-article",
    "letter": "journal-article",
    "book": "book",
    "book-chapter": "book-chapter",
    "dissertation": "thesis",
    "report": "report",
    "preprint": "preprint",
    "proceedings-article": "conference-paper",
    "standard": "report",
    "paratext": "unknown",
    "erratum": "unknown",
    "dataset": "dataset",
    "peer-review": "peer-review",
}

_FIELDS = (
    "id,doi,title,display_name,publication_year,publication_date,authorships,type,"
    "primary_location,best_oa_location,open_access,cited_by_count,"
    "abstract_inverted_index,ids,language"
)


def _default_headers(mailto: str | None) -> dict[str, str]:
    agent = "phd-rag/0.1 (academic discovery)"
    if mailto:
        agent = f"{agent} mailto:{mailto}"
    return {"User-Agent": agent, "Accept": "application/json"}


def _abstract_from_inverted(inverted: Any) -> str | None:
    """Reconstruct an abstract from OpenAlex's inverted index."""
    if not isinstance(inverted, dict) or not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for position in indexes:
            if isinstance(position, int):
                positions.append((position, str(word)))
    if not positions:
        return None
    positions.sort(key=lambda pair: pair[0])
    return collapse_whitespace(" ".join(word for _position, word in positions)) or None


def _authors_from_work(work: dict[str, Any]) -> tuple[str, ...]:
    authors: list[str] = []
    for item in work.get("authorships") or []:
        if not isinstance(item, dict):
            continue
        author = item.get("author") or {}
        name = clean_external_text(author.get("display_name") if isinstance(author, dict) else None)
        if name:
            authors.append(name)
    return tuple(authors)


def _source_name(location: Any) -> str | None:
    if not isinstance(location, dict):
        return None
    source = location.get("source") or {}
    if not isinstance(source, dict):
        return None
    return clean_external_text(source.get("display_name"))


def _publisher_name(location: Any) -> str | None:
    if not isinstance(location, dict):
        return None
    source = location.get("source") or {}
    if not isinstance(source, dict):
        return None
    return clean_external_text(source.get("host_organization_name"))


def work_from_work(work: dict[str, Any]) -> ExternalWork:
    """Normalize one OpenAlex work into an :class:`ExternalWork`."""
    raw_title = work.get("title") or work.get("display_name")
    title = clean_external_text(raw_title)
    doi = normalize_doi(str(work.get("doi") or ""))
    year = work.get("publication_year")
    year = int(year) if isinstance(year, int) else None
    authors = _authors_from_work(work)
    primary = work.get("primary_location")
    best_oa = work.get("best_oa_location")
    oa = work.get("open_access") or {}
    oa = oa if isinstance(oa, dict) else {}
    is_oa = oa.get("is_oa")
    oa_status = oa.get("oa_status")
    pdf_url = None
    landing = None
    for location in (best_oa, primary):
        if not isinstance(location, dict):
            continue
        if pdf_url is None:
            pdf_url = collapse_whitespace(str(location.get("pdf_url") or "")) or None
        if landing is None:
            landing = collapse_whitespace(str(location.get("landing_page_url") or "")) or None
    cited = work.get("cited_by_count")
    cited_by = int(cited) if isinstance(cited, int) else None
    abstract = _abstract_from_inverted(work.get("abstract_inverted_index"))
    publication_type = _TYPE_MAP.get(str(work.get("type") or "").casefold())
    venue = _source_name(primary)
    publisher = _publisher_name(primary)
    url = collapse_whitespace(str(work.get("id") or "")) or landing

    if is_oa is True and pdf_url:
        fulltext_url, fulltext_status = pdf_url, FulltextStatus.FULLTEXT_OPEN
    elif landing or url:
        fulltext_url, fulltext_status = landing, FulltextStatus.LANDING_PAGE_ONLY
    else:
        fulltext_url, fulltext_status = None, FulltextStatus.UNKNOWN

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
            ("open_access", is_oa if isinstance(is_oa, bool) else None),
            ("open_access_status", clean_external_text(oa_status)),
            ("fulltext_url", fulltext_url),
            ("publication_type", publication_type),
        )
        if value not in (None, "", ())
    }
    raw_ids: dict[str, str] = {}
    work_id = work.get("id")
    if isinstance(work_id, str) and work_id:
        raw_ids[PROVIDER_NAME] = work_id.rsplit("/", 1)[-1]
    ids = work.get("ids") or {}
    if isinstance(ids, dict):
        for key in ("mag", "pmid", "pmcid"):
            value = ids.get(key)
            if isinstance(value, str) and value:
                raw_ids[key] = value.rsplit("/", 1)[-1]
    return ExternalWork(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        venue=venue,
        publisher=publisher,
        publication_type=publication_type,
        abstract=abstract,
        cited_by_count=cited_by,
        open_access=is_oa if isinstance(is_oa, bool) else None,
        open_access_status=clean_external_text(oa_status),
        fulltext_url=fulltext_url,
        fulltext_status=fulltext_status,
        providers=(PROVIDER_NAME,),
        raw_provider_ids=raw_ids,
        field_provenance=provenance,
        source_provenance=(f"openalex:{work.get('type') or 'unknown'}",),
        missing_fields=missing,
    )


class OpenAlexProvider:
    """OpenAlex search and DOI/title lookup."""

    provider_name = PROVIDER_NAME

    def __init__(
        self,
        http: HttpClient,
        *,
        cache: DiscoveryCache | None = None,
        base_url: str = BASE_URL,
        mailto: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._http = http
        self._cache = cache
        self._base_url = base_url.rstrip("/")
        self._mailto = mailto
        self._api_key = api_key
        self._headers = _default_headers(mailto)

    # --- helpers --------------------------------------------------------------
    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(extra)
        params["select"] = _FIELDS
        if self._mailto:
            params["mailto"] = self._mailto
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def _get(self, path: str, params: dict[str, Any]) -> tuple[Any | None, ProviderFailure | None, int]:
        result = self._http.get_json(f"{self._base_url}/{path}", params=params, headers=self._headers)
        if not result.ok:
            failure = failure_from_http(PROVIDER_NAME, path, result)
            if failure.status_code == 404:
                failure = ProviderFailure(
                    provider=PROVIDER_NAME,
                    kind="not_found",
                    status_code=404,
                    message=f"OpenAlex has no record at {path}",
                )
            return None, failure, 1
        return result.data, None, 1

    @staticmethod
    def _records_from(data: Any) -> list[ExternalWork]:
        results = (data or {}).get("results")
        if not isinstance(results, list):
            return []
        return [work_from_work(item) for item in results if isinstance(item, dict)]

    def _search_params(
        self, query: str, limit: int, filters: DiscoveryFilters | None
    ) -> dict[str, Any]:
        filters = filters or DiscoveryFilters()
        params: dict[str, Any] = {
            "search": normalize_query(query),
            "per-page": max(1, min(limit, 200)),
        }
        filter_parts: list[str] = []
        if filters.year_min is not None:
            filter_parts.append(f"from_publication_date:{filters.year_min}-01-01")
        if filters.year_max is not None:
            filter_parts.append(f"to_publication_date:{filters.year_max}-12-31")
        if filters.publication_types:
            filter_parts.append(f"type:{'|'.join(filters.publication_types)}")
        if filters.author:
            # OpenAlex filters raw author names with this search sub-filter.
            filter_parts.append(f"raw_author_name.search:{filters.author}")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        if filters.sort == "recent":
            params["sort"] = "publication_date:desc"
        return self._params(params)

    # --- interface ------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: DiscoveryFilters | None = None,
    ) -> ProviderResponse:
        """Bibliographic search over OpenAlex ``/works``."""
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
            return ProviderResponse(
                PROVIDER_NAME,
                "search",
                normalized,
                records=self._records_from(cached.value),
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
        if self._cache:
            self._cache.set(cache_key, data)
        return ProviderResponse(
            PROVIDER_NAME,
            "search",
            normalized,
            records=self._records_from(data),
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
            work = cached.value if isinstance(cached.value, dict) else None
            return ProviderResponse(
                PROVIDER_NAME,
                "doi",
                canonical,
                records=[work_from_work(work)] if work else [],
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        data, failure, requests = self._get(
            f"works/https://doi.org/{canonical}", self._params({})
        )
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
        work = data if isinstance(data, dict) else None
        return ProviderResponse(
            PROVIDER_NAME,
            "doi",
            canonical,
            records=[work_from_work(work)] if work else [],
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )

    def lookup_title(self, title: str, *, limit: int = 5) -> ProviderResponse:
        """Look up one title."""
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
            return ProviderResponse(
                PROVIDER_NAME,
                "title",
                normalized,
                records=self._records_from(cached.value),
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )
        data, failure, requests = self._get(
            "works",
            self._params({"search": normalized, "per-page": max(1, min(limit, 200))}),
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
        if self._cache:
            self._cache.set(cache_key, data)
        return ProviderResponse(
            PROVIDER_NAME,
            "title",
            normalized,
            records=self._records_from(data),
            http_requests=requests,
            seconds=time.perf_counter() - started,
        )
