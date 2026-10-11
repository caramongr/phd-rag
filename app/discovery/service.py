"""Academic discovery service (RAG-010).

The workflow is fixed and auditable:

```
query / DOI / title
  ↓  provider search (Crossref, OpenAlex, optional Semantic Scholar)
  ↓  normalize provider records
  ↓  merge duplicates + resolve field provenance/conflicts
  ↓  assign status and identity confidence
  ↓  compare with the local corpus
  ↓  rank deterministically
  ↓  result + audit trail
```

Two properties matter more than features:

* **isolation.** A failing provider never fails the request: its failure is recorded and
  the other providers continue (§11, §28).
* **no silent mixing.** External records are returned as :class:`ExternalWork` with an
  explicit status and a local-match verdict; they are never inserted into the corpus,
  the index, the embeddings or the RAG-009 evidence path (§16).
"""

from __future__ import annotations

import datetime as _datetime
import time
from typing import Iterable, Sequence

from app.config import Settings, get_settings
from app.discovery.cache import DiscoveryCache, normalize_query
from app.discovery.local import LocalCorpusIndex, build_index, payloads_from_collection
from app.discovery.merge import merge_conflicts
from app.discovery.models import (
    DiscoveryAudit,
    DiscoveryFilters,
    DiscoveryResult,
    ExternalWork,
    LocalMatchKind,
    ProviderFailure,
    ProviderQuery,
)
from app.discovery.providers import build_provider
from app.discovery.providers.base import AcademicDiscoveryProvider, ProviderResponse
from app.discovery.ranking import describe_weights, rank_records
from app.metadata.external.http import HttpClient, RateLimiter, RequestsTransport
from app.metadata.text import normalize_for_comparison, title_similarity

#: Providers that are searched by default. Semantic Scholar stays opt-in.
DEFAULT_PROVIDERS = ("crossref", "openalex")


class DiscoveryError(RuntimeError):
    """Base class for discovery failures."""


class DiscoveryService:
    """Discover external scholarly material, without ever touching the corpus."""

    def __init__(
        self,
        *,
        providers: Sequence[AcademicDiscoveryProvider] | None = None,
        http: HttpClient | None = None,
        cache: DiscoveryCache | None = None,
        local_index: LocalCorpusIndex | None = None,
        settings: Settings | None = None,
        provider_names: Sequence[str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or DiscoveryCache(
            self.settings.discovery_cache_path,
            ttl_seconds=self.settings.discovery_cache_ttl,
            enabled=self.settings.discovery_cache_enabled,
        )
        self._http = http or HttpClient(
            RequestsTransport(),
            timeout=self.settings.discovery_request_timeout,
            max_retries=self.settings.discovery_max_retries,
            backoff_base=self.settings.discovery_backoff_base,
            rate_limiter=RateLimiter(self.settings.discovery_min_request_interval),
        )
        names = (
            tuple(provider_names)
            if provider_names is not None
            else tuple(self.settings.discovery_providers or DEFAULT_PROVIDERS)
        )
        self.provider_names = names
        self._providers = (
            list(providers)
            if providers is not None
            else [
                build_provider(
                    name,
                    http=self._http,
                    cache=self.cache,
                    mailto=self.settings.metadata_crossref_mailto,
                    openalex_api_key=self.settings.openalex_api_key,
                    openalex_email=self.settings.openalex_email,
                )
                for name in names
            ]
        )
        self._local_index = local_index
        #: The most recent opt-in full-text verification, for audit/CLI reporting.
        self.last_verification: object | None = None

    # --- local corpus ---------------------------------------------------------
    @property
    def local_index(self) -> LocalCorpusIndex:
        """The local corpus index, built lazily from the Qdrant payloads."""
        if self._local_index is None:
            self._local_index = self._load_local_index()
        return self._local_index

    @property
    def has_local_index(self) -> bool:
        """Whether a local index is already available."""
        return self._local_index is not None

    def share_local_index(self, index: LocalCorpusIndex) -> None:
        """Use a local index built elsewhere.

        The combined service (RAG-010 §21) runs RAG-009 retrieval and discovery in one
        process. A local Qdrant store allows only one open client per process, so the
        index must come from the client the retrieval service already holds instead of
        opening a second one.
        """
        self._local_index = index

    def share_client(self, client: Any, collection: str | None = None) -> None:
        """Build the local index from an already-open Qdrant client (read-only)."""
        self._local_index = build_index(
            payloads_from_collection(client, collection or self.settings.qdrant_collection)
        )

    def _load_local_index(self) -> LocalCorpusIndex:
        from app.indexing.collection import build_client

        client = build_client(
            path=self.settings.qdrant_path or None,
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        try:
            payloads = payloads_from_collection(client, self.settings.qdrant_collection)
        finally:
            client.close()
        return build_index(payloads)

    @property
    def providers(self) -> tuple[AcademicDiscoveryProvider, ...]:
        """The configured providers."""
        return tuple(self._providers)

    # --- helpers --------------------------------------------------------------
    def _collect(
        self,
        responses: Iterable[ProviderResponse],
        *,
        operation: str,
        query: str,
        filters: DiscoveryFilters | None,
        refresh: bool,
        audit_queries: list[ProviderQuery],
        failures: list[ProviderFailure],
    ) -> list[ExternalWork]:
        """Gather records from provider responses, recording their fate."""
        collected: list[ExternalWork] = []
        for response in responses:
            audit_queries.append(
                ProviderQuery(
                    provider=response.provider,
                    operation=operation,
                    query=query,
                    ok=response.ok,
                    records=response.count,
                    cache_hit=response.cache_hit,
                    http_requests=response.http_requests,
                    seconds=round(response.seconds, 4),
                    failure=response.failure,
                )
            )
            if response.failure is not None:
                failures.append(response.failure)
                continue
            collected.extend(response.records)
        return collected

    def _apply_filters(self, records: Sequence[ExternalWork], filters: DiscoveryFilters) -> tuple[list[ExternalWork], int]:
        """Re-check provider filters locally (spec §20)."""
        kept: list[ExternalWork] = []
        dropped = 0
        for work in records:
            if filters.year_min is not None and (work.year is None or work.year < filters.year_min):
                dropped += 1
                continue
            if filters.year_max is not None and (work.year is None or work.year > filters.year_max):
                dropped += 1
                continue
            if filters.publication_types and (
                work.publication_type is None
                or work.publication_type not in filters.publication_types
            ):
                dropped += 1
                continue
            if filters.author and not _author_matches(work, filters.author):
                dropped += 1
                continue
            kept.append(work)
        return kept, dropped

    def _finish(
        self,
        *,
        raw_query: str,
        normalized: str,
        operation: str,
        records: Sequence[ExternalWork],
        filters: DiscoveryFilters,
        audit_queries: list[ProviderQuery],
        failures: list[ProviderFailure],
        refresh: bool,
        provider_records: int,
        duplicates_merged: int,
        local_matched: int,
        possible_matches: int,
        new_records: int,
        conflicts: int,
        requested: Sequence[str],
    ) -> DiscoveryResult:
        succeeded = tuple(dict.fromkeys(item.provider for item in audit_queries if item.ok))
        failed = tuple(dict.fromkeys(failure.provider for failure in failures))
        audit = DiscoveryAudit(
            raw_query=raw_query,
            normalized_query=normalized,
            operation=operation,
            providers_requested=tuple(requested),
            providers_succeeded=succeeded,
            providers_failed=failed,
            failures=tuple(failures),
            filters=filters.describe(),
            queries=tuple(audit_queries),
            provider_records=provider_records,
            merged_records=len(records),
            duplicates_merged=duplicates_merged,
            local_matches=local_matched,
            possible_matches=possible_matches,
            new_records=new_records,
            conflicts=conflicts,
            cache_hits=self.cache.hits,
            cache_misses=self.cache.misses,
            refresh=refresh,
            all_providers_failed=bool(requested) and not succeeded,
            timestamp=_datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
        )
        return DiscoveryResult(audit=audit, records=tuple(records))

    def _compare_locally(self, records: Sequence[ExternalWork]) -> list[ExternalWork]:
        """Attach the local-corpus verdict to every record (required, §7)."""
        index = self.local_index
        return [
            work.model_copy(update={"local_match": index.match(work)}) for work in records
        ]

    # --- public API -----------------------------------------------------------
    def discover(
        self,
        query: str,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
        max_results: int = 20,
        providers: Sequence[str] | None = None,
        filters: DiscoveryFilters | None = None,
        refresh: bool = False,
        author: str | None = None,
        sort: str = "relevance",
        publication_types: Sequence[str] = (),
    ) -> DiscoveryResult:
        """Search every provider, merge, compare with the corpus and rank."""
        filters = filters or DiscoveryFilters(
            year_min=year_min,
            year_max=year_max,
            author=author,
            publication_types=tuple(publication_types),
            sort=sort,
        )
        normalized = normalize_query(query)
        if not normalized:
            raise DiscoveryError("query is empty after normalization")
        self.cache.reset_counters()
        selected = self._select_providers(providers)
        started = time.perf_counter()
        audit_queries: list[ProviderQuery] = []
        failures: list[ProviderFailure] = []
        responses = [
            provider.search(normalized, limit=max_results, filters=filters)
            for provider in selected
        ]
        collected = self._collect(
            responses,
            operation="search",
            query=normalized,
            filters=filters,
            refresh=refresh,
            audit_queries=audit_queries,
            failures=failures,
        )
        provider_records = len(collected)
        filtered, _dropped = self._apply_filters(collected, filters)
        merged, merged_away = merge_conflicts(filtered)
        ranked = rank_records(merged, normalized, limit=max_results)
        with_local = self._compare_locally(ranked)
        return self._finish(
            raw_query=query,
            normalized=normalized,
            operation="search",
            records=with_local,
            filters=filters,
            audit_queries=audit_queries,
            failures=failures,
            refresh=refresh,
            provider_records=provider_records,
            duplicates_merged=merged_away,
            local_matched=sum(1 for work in with_local if work.local_match.kind is LocalMatchKind.ALREADY_IN_CORPUS),
            possible_matches=sum(1 for work in with_local if work.local_match.kind is LocalMatchKind.POSSIBLE_MATCH),
            new_records=sum(1 for work in with_local if not work.local_match.already_in_corpus),
            conflicts=sum(1 for work in with_local if work.conflicts),
            requested=tuple(provider.provider_name for provider in selected),
        )

    def lookup_doi(self, doi: str, *, providers: Sequence[str] | None = None, refresh: bool = False) -> DiscoveryResult:
        """Look up one DOI across providers and verify its identity."""
        self.cache.reset_counters()
        selected = self._select_providers(providers)
        started = time.perf_counter()
        audit_queries: list[ProviderQuery] = []
        failures: list[ProviderFailure] = []
        responses = [provider.lookup_doi(doi) for provider in selected]
        collected = self._collect(
            responses,
            operation="doi",
            query=doi,
            filters=None,
            refresh=refresh,
            audit_queries=audit_queries,
            failures=failures,
        )
        merged, merged_away = merge_conflicts(collected, doi_lookup=True)
        with_local = self._compare_locally(merged)
        return self._finish(
            raw_query=doi,
            normalized=normalize_query(doi),
            operation="doi",
            records=with_local,
            filters=DiscoveryFilters(),
            audit_queries=audit_queries,
            failures=failures,
            refresh=refresh,
            provider_records=len(collected),
            duplicates_merged=merged_away,
            local_matched=sum(1 for work in with_local if work.local_match.kind is LocalMatchKind.ALREADY_IN_CORPUS),
            possible_matches=sum(1 for work in with_local if work.local_match.kind is LocalMatchKind.POSSIBLE_MATCH),
            new_records=sum(1 for work in with_local if not work.local_match.already_in_corpus),
            conflicts=sum(1 for work in with_local if work.conflicts),
            requested=tuple(provider.provider_name for provider in selected),
        )

    def lookup_title(self, title: str, *, providers: Sequence[str] | None = None, refresh: bool = False) -> DiscoveryResult:
        """Look up one exact title across providers."""
        self.cache.reset_counters()
        selected = self._select_providers(providers)
        started = time.perf_counter()
        audit_queries: list[ProviderQuery] = []
        failures: list[ProviderFailure] = []
        responses = [provider.lookup_title(title, limit=5) for provider in selected]
        collected = self._collect(
            responses,
            operation="title",
            query=title,
            filters=None,
            refresh=refresh,
            audit_queries=audit_queries,
            failures=failures,
        )
        # Keep only candidates that really are this title.
        comparable = [
            work
            for work in collected
            if work.title and title_similarity(title, work.title) >= 0.90
        ]
        merged, merged_away = merge_conflicts(comparable)
        ranked = rank_records(merged, title)
        with_local = self._compare_locally(ranked)
        return self._finish(
            raw_query=title,
            normalized=normalize_query(title),
            operation="title",
            records=with_local,
            filters=DiscoveryFilters(),
            audit_queries=audit_queries,
            failures=failures,
            refresh=refresh,
            provider_records=len(collected),
            duplicates_merged=merged_away,
            local_matched=sum(1 for work in with_local if work.local_match.kind is LocalMatchKind.ALREADY_IN_CORPUS),
            possible_matches=sum(1 for work in with_local if work.local_match.kind is LocalMatchKind.POSSIBLE_MATCH),
            new_records=sum(1 for work in with_local if not work.local_match.already_in_corpus),
            conflicts=sum(1 for work in with_local if work.conflicts),
            requested=tuple(provider.provider_name for provider in selected),
        )

    def find_related(
        self,
        *,
        document_id: str | None = None,
        doi: str | None = None,
        title: str | None = None,
        since_year: int | None = None,
        max_results: int = 20,
        providers: Sequence[str] | None = None,
        refresh: bool = False,
    ) -> DiscoveryResult:
        """Find newer work related to one local paper (spec §23).

        The query is built deterministically from the local metadata - no model
        rewrites it - and the local document itself is always reported as already
        present, so "related" never masquerades as "new".
        """
        base_title = title
        base_doi = doi
        base_year = None
        local_document = None
        if document_id:
            for document in self.local_index.documents:
                if document.document_id == document_id:
                    local_document = document
                    base_title = base_title or document.title
                    base_doi = base_doi or document.doi
                    base_year = document.year
                    break
            if local_document is None and not base_title and not base_doi:
                raise DiscoveryError(f"local document {document_id!r} not found")
        if base_doi and not base_doi.strip():
            base_doi = None
        if not base_title and not base_doi:
            raise DiscoveryError("find_related needs a document_id, DOI or title")

        year_floor = since_year
        if year_floor is None and base_year is not None:
            year_floor = base_year + 1
        query = base_title or base_doi or ""
        result = self.discover(
            query,
            year_min=year_floor,
            max_results=max_results,
            providers=providers,
            refresh=refresh,
            sort="recent",
        )
        return result

    # --- internals ------------------------------------------------------------
    def _select_providers(
        self, names: Sequence[str] | None
    ) -> list[AcademicDiscoveryProvider]:
        if names is None:
            return list(self._providers)
        wanted = {name.strip().lower() for name in names}
        selected = [
            provider for provider in self._providers if provider.provider_name in wanted
        ]
        if not selected:
            raise DiscoveryError(
                f"no configured provider matches {sorted(wanted)}; configured: "
                f"{[provider.provider_name for provider in self._providers]}"
            )
        return selected

    def verify_fulltext(
        self,
        record: ExternalWork,
        *,
        follow_landing_pages: bool = False,
        fetcher: object | None = None,
        timeout: float | None = None,
        max_bytes: int | None = None,
    ) -> ExternalWork:
        """Opt-in full-text verification of one external record (spec §18).

        Fetches only a legitimate open-access URL, verifies that the bytes really are
        this paper, and returns a copy of the record with the verification evidence
        attached. The metadata is never replaced, and nothing is imported.
        """
        from app.discovery.fulltext import verify_fulltext as _verify

        result = _verify(
            record,
            fetcher=fetcher,
            timeout=timeout or self.settings.discovery_fulltext_timeout,
            max_bytes=max_bytes or self.settings.discovery_fulltext_max_bytes,
            mailto=self.settings.metadata_crossref_mailto or self.settings.openalex_email,
            follow_landing_pages=follow_landing_pages,
        )
        updated = result.apply(record)
        self.last_verification = result
        return updated

    # --- reporting ------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        """Configuration summary for reports."""
        return {
            "providers": [provider.provider_name for provider in self._providers],
            "cache": self.cache.describe(),
            "local_documents": self.local_index.document_count,
            "local_documents_with_doi": self.local_index.doi_count,
            "local_documents_with_title": self.local_index.title_count,
            "ranking_weights": describe_weights(),
            "crossref_mailto_configured": bool(self.settings.metadata_crossref_mailto),
            "openalex_email_configured": bool(self.settings.openalex_email),
            "openalex_api_key_configured": bool(self.settings.openalex_api_key),
            "polite_pool_identity_configured": bool(
                self.settings.metadata_crossref_mailto or self.settings.openalex_email
            ),
        }


def _author_matches(work: ExternalWork, author: str) -> bool:
    """Whether any author name looks like the requested author."""
    needle = normalize_for_comparison(author)
    if not needle:
        return True
    for name in work.authors:
        if needle in normalize_for_comparison(name):
            return True
    return False
