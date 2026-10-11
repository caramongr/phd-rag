"""External metadata enrichment orchestration (RAG-006).

Enrichment is opt-in, gated by corpus role, cached, and conservative: an
external record is merged only when the match is ``EXACT`` (DOI) or ``STRONG``
(title). Ambiguous or missing matches change nothing except the recorded
match status. Merged values are *added* as candidates, so the resolved value
gains a higher-priority source while every local candidate stays inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.metadata.corpus import is_enrichable
from app.metadata.doi import normalize_doi
from app.metadata.enums import (
    Confidence,
    ExternalMatchStatus,
    MetadataSource,
    PublicationType,
)
from app.metadata.external.cache import JsonCache
from app.metadata.external.clients import (
    CrossrefClient,
    ExternalRecord,
    OpenAlexClient,
)
from app.metadata.external.http import HttpClient, RateLimiter, RequestsTransport
from app.metadata.external.matching import (
    CONFLICT_TITLE_SIMILARITY,
    doi_match_status,
    title_match_status,
    title_similarity,
)
from app.metadata.models import AcademicMetadata, MetadataConflict, MetadataValue
from app.metadata.resolver import add_candidate, resolve_all

#: Local title sources strong enough to contradict an external record. A
#: filename token is deliberately excluded: it is too weak to raise a conflict.
_STRONG_LOCAL_TITLE_SOURCES = frozenset(
    {
        MetadataSource.PDF_METADATA,
        MetadataSource.DOCX_PROPERTIES,
        MetadataSource.DOCUMENT_TEXT,
    }
)


@dataclass
class EnrichmentComponents:
    """The wired external-enrichment components, exposed for reporting."""

    http: HttpClient
    cache: JsonCache
    crossref: CrossrefClient
    openalex: OpenAlexClient


@dataclass
class EnrichmentStats:
    """Cumulative counters describing what an enricher actually did.

    Populated as documents are enriched, so a stage runner can report exactly
    which lookups were made, accepted, conflicted or left unmatched.
    """

    eligible_doi: int = 0
    eligible_no_doi: int = 0
    skipped_not_enrichable: int = 0
    #: Crossref returned a record whose DOI matched the requested DOI.
    crossref_exact: int = 0
    #: OpenAlex returned a matching DOI record as the fallback.
    openalex_fallback_exact: int = 0
    #: Records merged (accepted).
    accepted: int = 0
    conflicts: int = 0
    no_match: int = 0
    #: Title-based searches attempted (0 in a DOI-only stage).
    title_lookups: int = 0
    accepted_by_source: dict[str, int] = field(default_factory=dict)
    conflicts_by_source: dict[str, int] = field(default_factory=dict)

    def _bump(self, mapping: dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1

    def note_accepted(self, source: MetadataSource) -> None:
        self.accepted += 1
        self._bump(self.accepted_by_source, source.value)

    def note_conflict(self, source: MetadataSource) -> None:
        self.conflicts += 1
        self._bump(self.conflicts_by_source, source.value)


class MetadataEnricher:
    """Applies external bibliographic enrichment to local metadata."""

    def __init__(
        self,
        *,
        crossref: CrossrefClient | None = None,
        openalex: OpenAlexClient | None = None,
        enabled: bool = True,
        title_lookup_enabled: bool = True,
    ) -> None:
        self.crossref = crossref
        self.openalex = openalex
        self.enabled = enabled
        #: When ``False`` the enricher performs DOI-only enrichment and never
        #: issues a title-based search (used by the DOI-only stage).
        self.title_lookup_enabled = title_lookup_enabled
        self.stats = EnrichmentStats()
        #: Outcome of the most recent enrichment, exposed so a stage runner can
        #: report exactly which external record was considered and why.
        self.last_record: ExternalRecord | None = None
        self.last_route: str | None = None
        self.last_status: ExternalMatchStatus | None = None
        #: Memoizes results per file hash so binary-identical duplicates make at
        #: most one external request set.
        self._by_hash: dict[str, AcademicMetadata] = {}

    def enrich(self, metadata: AcademicMetadata, *, force: bool = False) -> AcademicMetadata:
        """Return ``metadata`` enriched from external services where permitted.

        ``force`` bypasses the corpus-role gate (used only for explicit callers).
        """
        if not self.enabled:
            return metadata

        if metadata.file_hash and metadata.file_hash in self._by_hash:
            return self._by_hash[metadata.file_hash].model_copy(deep=True)

        if not force and not is_enrichable(
            metadata.corpus_role, has_doi=metadata.doi is not None
        ):
            self.stats.skipped_not_enrichable += 1
            return metadata

        if metadata.doi is None and not self.title_lookup_enabled:
            self.stats.eligible_no_doi += 1
            return metadata

        if metadata.doi is None and _title_search_source(metadata) is None:
            # No locally observed DOI and no title strong enough to search with:
            # a filename token must never initiate an external title search.
            self.stats.eligible_no_doi += 1
            return metadata

        if metadata.doi is not None:
            self.stats.eligible_doi += 1
        else:
            self.stats.eligible_no_doi += 1

        result = self._enrich(metadata)
        if metadata.file_hash:
            self._by_hash[metadata.file_hash] = result
        return result

    def _enrich(self, metadata: AcademicMetadata) -> AcademicMetadata:
        candidates: dict[str, list[MetadataValue]] = {
            name: list(values) for name, values in metadata.candidates.items()
        }
        conflicts: list[MetadataConflict] = list(metadata.conflicts)
        local_title = metadata.title.value if metadata.title else None
        local_year = metadata.year.value if metadata.year else None
        local_family = _local_first_family(metadata)

        record: ExternalRecord | None = None
        status = ExternalMatchStatus.NO_MATCH
        route: str | None = None

        if metadata.doi is not None:
            record, status, route = self._resolve_by_doi(metadata.doi.value)
        elif self.title_lookup_enabled and _title_search_source(metadata) is not None:
            self.stats.title_lookups += 1
            record, status, route = self._resolve_by_title(
                local_title, local_year, local_family
            )

        # An exact DOI is only as trustworthy as the DOI extracted locally, so a
        # DOI-exact record that contradicts a strong local title is withheld and
        # recorded as a conflict rather than silently made preferred.
        if record is not None and status is ExternalMatchStatus.EXACT:
            conflict = _doi_title_conflict(metadata, record, route or "DOI lookup")
            if conflict is not None:
                conflicts.append(conflict)
                self.stats.note_conflict(record.source)
                status = ExternalMatchStatus.CONFLICT
                record = None

        if record is not None and status in (
            ExternalMatchStatus.EXACT,
            ExternalMatchStatus.STRONG,
        ):
            confidence = (
                Confidence.HIGH
                if status is ExternalMatchStatus.EXACT
                else Confidence.MEDIUM
            )
            _merge_record(candidates, record, confidence, route or "external lookup")
            self.stats.note_accepted(record.source)
        elif status is ExternalMatchStatus.NO_MATCH:
            self.stats.no_match += 1

        self.last_record = record
        self.last_route = route
        self.last_status = status
        return _rebuild(metadata, candidates, status, conflicts)

    def _resolve_by_doi(
        self, doi: str
    ) -> tuple[ExternalRecord | None, ExternalMatchStatus, str | None]:
        normalized = normalize_doi(doi) or doi
        if self.crossref is not None:
            record = self.crossref.lookup_doi(normalized)
            if record is not None and doi_match_status(record, normalized) is ExternalMatchStatus.EXACT:
                self.stats.crossref_exact += 1
                return record, ExternalMatchStatus.EXACT, "Crossref DOI lookup"
        if self.openalex is not None:
            record = self.openalex.lookup_doi(normalized)
            if record is not None and doi_match_status(record, normalized) is ExternalMatchStatus.EXACT:
                self.stats.openalex_fallback_exact += 1
                return record, ExternalMatchStatus.EXACT, "OpenAlex DOI lookup"
        return None, ExternalMatchStatus.NO_MATCH, None

    def _resolve_by_title(
        self, title: str, year: int | None, family: str | None
    ) -> tuple[ExternalRecord | None, ExternalMatchStatus, str | None]:
        attempts = []
        if self.crossref is not None:
            attempts.append((self.crossref, "Crossref"))
        if self.openalex is not None:
            attempts.append((self.openalex, "OpenAlex"))

        best_record: ExternalRecord | None = None
        best_status = ExternalMatchStatus.NO_MATCH
        route: str | None = None
        for client, name in attempts:
            records = client.search_title(title)
            record, status = title_match_status(
                records, local_title=title, local_year=year, local_family=family
            )
            if status is ExternalMatchStatus.STRONG:
                return record, status, f"strong title-based external match ({name} title search)"
            if status is ExternalMatchStatus.AMBIGUOUS and best_status is not ExternalMatchStatus.AMBIGUOUS:
                best_record, best_status, route = record, status, f"{name} title search"
        return best_record, best_status, route


def local_first_author_family(metadata: AcademicMetadata) -> str | None:
    """Public wrapper: the local first author's family name, when derivable."""
    return _local_first_family(metadata)


def _local_first_family(metadata: AcademicMetadata) -> str | None:
    if metadata.authors is None or not metadata.authors.value:
        return None
    first = metadata.authors.value[0]
    if first.family_name:
        return first.family_name
    tokens = first.full_name.split()
    return tokens[-1] if len(tokens) >= 2 else None


def _strong_local_title(metadata: AcademicMetadata) -> MetadataValue | None:
    """Return the local title candidate strong enough to contradict a record."""
    title = metadata.title
    if title is None:
        return None
    if title.source not in _STRONG_LOCAL_TITLE_SOURCES:
        return None
    if title.confidence is Confidence.LOW:
        return None
    return title


def _doi_title_conflict(
    metadata: AcademicMetadata, record: ExternalRecord, route: str
) -> MetadataConflict | None:
    """Return a conflict when a DOI-exact record contradicts a strong local title."""
    local = _strong_local_title(metadata)
    if local is None or not record.title:
        return None
    similarity = title_similarity(local.value, record.title)
    if similarity >= CONFLICT_TITLE_SIMILARITY:
        return None
    return MetadataConflict(
        field="title",
        local_value=local.value,
        external_value=record.title,
        external_source=record.source,
        similarity=round(similarity, 4),
        route=route,
    )


def _merge_record(
    candidates: dict[str, list[MetadataValue]],
    record: ExternalRecord,
    confidence: Confidence,
    route: str,
) -> None:
    """Add an accepted external record's fields as candidates."""
    source: MetadataSource = record.source

    def evidence(field_name: str) -> str:
        base = f"{route}; external {field_name}"
        if field_name in record.normalised:
            return f"{base} (provider text normalised)"
        return base

    if record.title:
        add_candidate(
            candidates,
            "title",
            MetadataValue(value=record.title, source=source, confidence=confidence, evidence=evidence("title")),
        )
    if record.authors:
        add_candidate(
            candidates,
            "authors",
            MetadataValue(value=list(record.authors), source=source, confidence=confidence, evidence=evidence("authors")),
        )
    if record.year is not None:
        add_candidate(
            candidates,
            "year",
            MetadataValue(value=record.year, source=source, confidence=confidence, evidence=evidence("year")),
        )
    if record.doi:
        add_candidate(
            candidates,
            "doi",
            MetadataValue(value=record.doi, source=source, confidence=confidence, evidence=evidence("DOI")),
        )
        add_candidate(
            candidates,
            "url",
            MetadataValue(
                value=f"https://doi.org/{record.doi}",
                source=source,
                confidence=confidence,
                evidence=evidence("DOI resolver URL"),
            ),
        )
    if record.venue:
        add_candidate(
            candidates,
            "venue",
            MetadataValue(value=record.venue, source=source, confidence=confidence, evidence=evidence("venue")),
        )
    if record.publisher:
        add_candidate(
            candidates,
            "publisher",
            MetadataValue(value=record.publisher, source=source, confidence=confidence, evidence=evidence("publisher")),
        )
    if record.publication_type is not PublicationType.UNKNOWN:
        add_candidate(
            candidates,
            "publication_type",
            MetadataValue(
                value=record.publication_type,
                source=source,
                confidence=confidence,
                evidence=evidence("publication type"),
            ),
        )
    if record.url and not record.doi:
        add_candidate(
            candidates,
            "url",
            MetadataValue(value=record.url, source=source, confidence=confidence, evidence=evidence("URL")),
        )
    if record.abstract:
        add_candidate(
            candidates,
            "abstract",
            MetadataValue(value=record.abstract, source=source, confidence=confidence, evidence=evidence("abstract")),
        )


def _rebuild(
    metadata: AcademicMetadata,
    candidates: dict[str, list[MetadataValue]],
    status: ExternalMatchStatus,
    conflicts: list[MetadataConflict],
) -> AcademicMetadata:
    resolved = resolve_all(candidates)
    return AcademicMetadata(
        file_hash=metadata.file_hash,
        filename=metadata.filename,
        source_path=metadata.source_path,
        extension=metadata.extension,
        corpus_role=metadata.corpus_role,
        title=resolved.get("title"),
        authors=resolved.get("authors"),
        year=resolved.get("year"),
        doi=resolved.get("doi"),
        venue=resolved.get("venue"),
        publisher=resolved.get("publisher"),
        publication_type=resolved.get("publication_type"),
        url=resolved.get("url"),
        abstract=resolved.get("abstract"),
        arxiv_id=resolved.get("arxiv_id"),
        candidates=candidates,
        external_match_status=status,
        conflicts=conflicts,
    )


def has_strong_local_title(metadata: AcademicMetadata) -> bool:
    """Return whether ``metadata`` carries a title strong enough for cross-check.

    A DOI-exact record can only be cross-checked against the document's own
    identity when such a strong local title exists; otherwise no title-conflict
    check is possible and the caller should report that separately.
    """
    return _strong_local_title(metadata) is not None


def _title_search_source(metadata: AcademicMetadata) -> MetadataSource | None:
    """Return the title source if it may initiate an external title search.

    Only a title that was actually observed in the document (embedded PDF/DOCX
    metadata or page text) qualifies; a filename-derived token is too weak to
    establish a bibliographic identity and is never searched for.
    """
    strong = _strong_local_title(metadata)
    return strong.source if strong is not None else None


def strong_title_source(metadata: AcademicMetadata) -> MetadataSource | None:
    """Public wrapper around :func:`_title_search_source` for stage runners."""
    return _title_search_source(metadata)


def build_components(
    *,
    settings: Settings | None = None,
    cache_path: str | None = None,
) -> EnrichmentComponents:
    """Build the HTTP client, cache and API clients from application settings."""
    settings = settings or get_settings()
    http = HttpClient(
        RequestsTransport(),
        timeout=settings.metadata_request_timeout,
        max_retries=settings.metadata_max_retries,
        rate_limiter=RateLimiter(settings.metadata_min_request_interval),
    )
    cache = JsonCache(cache_path or settings.metadata_cache_path)
    crossref = CrossrefClient(http, cache, mailto=settings.metadata_crossref_mailto)
    openalex = OpenAlexClient(http, cache, mailto=settings.metadata_crossref_mailto)
    return EnrichmentComponents(http=http, cache=cache, crossref=crossref, openalex=openalex)


def build_default_enricher(
    *,
    enabled: bool | None = None,
    settings: Settings | None = None,
    cache_path: str | None = None,
    title_lookup_enabled: bool = True,
) -> MetadataEnricher:
    """Build an enricher wired from application settings.

    When ``enabled`` is ``False`` (offline mode) the returned enricher makes no
    requests at all. Set ``title_lookup_enabled=False`` for a DOI-only run.
    """
    settings = settings or get_settings()
    if enabled is None:
        enabled = settings.metadata_enrich_external

    components = build_components(settings=settings, cache_path=cache_path)
    return MetadataEnricher(
        crossref=components.crossref,
        openalex=components.openalex,
        enabled=enabled,
        title_lookup_enabled=title_lookup_enabled,
    )
