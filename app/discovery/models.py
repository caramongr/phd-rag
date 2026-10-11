"""Discovery-layer models (RAG-010).

RAG-010 adds a second knowledge domain next to the trusted local corpus: results
found on the Internet. The two must never be silently mixed, so this module makes the
distinction structural:

* :class:`ExternalWork` is a *normalized* external record. It carries the provider
  identifiers, the provenance of every field and an explicit
  :class:`DiscoveryStatus`; it is never a :class:`~app.retrieval.models.Candidate` and
  never becomes local evidence.
* :class:`LocalMatch` records the comparison against the local corpus, including
  *why* a match was accepted and which local document it is.
* :class:`DiscoveryAudit` records what was asked, which providers answered and how
  many results survived each stage, so a discovery run is reproducible.
"""

from __future__ import annotations

import datetime as _datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiscoveryStatus(str, Enum):
    """How far an external record has been verified."""

    #: Already present in the user's corpus.
    LOCAL_MATCH = "LOCAL_MATCH"
    #: Found online; identity not sufficiently verified.
    EXTERNAL_DISCOVERY = "EXTERNAL_DISCOVERY"
    #: Identity and bibliographic metadata confidently verified.
    EXTERNAL_METADATA_VERIFIED = "EXTERNAL_METADATA_VERIFIED"
    #: Content acquired and its identity verified.
    EXTERNAL_FULLTEXT_VERIFIED = "EXTERNAL_FULLTEXT_VERIFIED"
    #: Known result whose required metadata/content is unavailable.
    UNAVAILABLE = "UNAVAILABLE"
    #: External sources disagree about identity/metadata.
    CONFLICT = "CONFLICT"


class FulltextStatus(str, Enum):
    """What is known about the full text of an external record."""

    FULLTEXT_OPEN = "FULLTEXT_OPEN"
    LANDING_PAGE_ONLY = "LANDING_PAGE_ONLY"
    UNKNOWN = "UNKNOWN"


class LocalMatchKind(str, Enum):
    """How an external record relates to the local corpus."""

    NEW = "NEW"
    ALREADY_IN_CORPUS = "ALREADY_IN_CORPUS"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    UNAVAILABLE = "UNAVAILABLE"


class FieldConflict(BaseModel):
    """Two providers materially disagreeing about one field."""

    model_config = ConfigDict(frozen=True)

    field_name: str
    values: tuple[str, ...]
    providers: tuple[str, ...]
    detail: str = ""


class ProviderFailure(BaseModel):
    """A provider that could not be used, and why."""

    model_config = ConfigDict(frozen=True)

    provider: str
    kind: str
    status_code: int | None = None
    message: str = ""
    retry_after: float | None = None
    counted_as_quota: bool = False


class LocalMatch(BaseModel):
    """The result of comparing an external record with the local corpus."""

    model_config = ConfigDict(frozen=True)

    #: Whether the corpus contains this document (identity match).
    already_in_corpus: bool = False
    kind: LocalMatchKind = LocalMatchKind.NEW
    local_document_id: str | None = None
    local_filename: str | None = None
    local_file_hash: str | None = None
    local_title: str | None = None
    local_year: int | None = None
    #: Corpus role of the matched document (``literature``, ``my_work``, ...).
    local_corpus_role: str | None = None
    #: Whether the matched local document is scholarly evidence. A match against the
    #: author's own draft never makes an external record trusted.
    local_evidence_eligible: bool = False
    match_reason: str = ""
    title_similarity: float | None = None
    year_delta: int | None = None
    author_consistent: bool | None = None


class ExternalWork(BaseModel):
    """One normalized external record after merging all providers."""

    model_config = ConfigDict(frozen=True)

    # --- identity and bibliographic metadata ---
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    venue: str | None = None
    publisher: str | None = None
    publication_type: str | None = None
    abstract: str | None = None
    cited_by_count: int | None = None
    open_access: bool | None = None
    open_access_status: str | None = None
    #: A URL that genuinely serves the full text when the provider says so.
    fulltext_url: str | None = None
    fulltext_status: FulltextStatus = FulltextStatus.UNKNOWN

    # --- provenance and verification ---
    #: Providers that returned this work, in deterministic order.
    providers: tuple[str, ...] = ()
    #: Provider-specific identifiers, e.g. ``{"openalex": "W123", "crossref": "10.x/y"}``.
    raw_provider_ids: dict[str, str] = Field(default_factory=dict)
    #: Which provider supplied each field, e.g. ``{"doi": ("crossref",), ...}``.
    field_provenance: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    #: Free-form provenance notes (e.g. the URL a full text was fetched from).
    source_provenance: tuple[str, ...] = ()
    conflicts: tuple[FieldConflict, ...] = ()

    discovery_status: DiscoveryStatus = DiscoveryStatus.EXTERNAL_DISCOVERY
    identity_confidence: float = 0.0
    #: Deterministic relevance score for the query that produced this record.
    relevance_score: float = 0.0

    # --- full-text verification outcome (only after a real fetch) ---
    fulltext_verified: bool = False
    fulltext_sha256: str | None = None
    fulltext_bytes: int | None = None
    fulltext_content_type: str | None = None
    fulltext_fetched_at: str | None = None
    fulltext_source_url: str | None = None
    fulltext_note: str | None = None

    # --- local corpus comparison ---
    local_match: LocalMatch = Field(default_factory=LocalMatch)

    # --- provider-level missing-field honesty ---
    #: Field names the providers did not supply. Never filled in by guessing.
    missing_fields: tuple[str, ...] = ()

    @property
    def display_title(self) -> str:
        """Title, or a clear placeholder when no provider supplied one."""
        return self.title or "(untitled)"

    @property
    def first_author(self) -> str | None:
        """First author as a display string, when known."""
        return self.authors[0] if self.authors else None

    @property
    def provider_label(self) -> str:
        """Providers joined for display."""
        return ", ".join(self.providers) if self.providers else "unknown"

    @property
    def is_evidence(self) -> bool:
        """External records are never trusted evidence inside RAG-010."""
        return False


class DiscoveryFilters(BaseModel):
    """Filters sent to providers and re-checked locally."""

    model_config = ConfigDict(frozen=True)

    year_min: int | None = None
    year_max: int | None = None
    #: Author name filter, where the provider supports it.
    author: str | None = None
    #: Restrict to these publication types, where the provider supports it.
    publication_types: tuple[str, ...] = ()
    #: Ask providers to sort by recency instead of relevance.
    sort: str = "relevance"

    @model_validator(mode="after")
    def _check(self) -> "DiscoveryFilters":
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            raise ValueError("year_min must not be greater than year_max")
        if self.sort not in ("relevance", "recent"):
            raise ValueError("sort must be 'relevance' or 'recent'")
        return self

    @property
    def is_recent_only(self) -> bool:
        """Whether the search is explicitly restricted to recent work."""
        return self.sort == "recent" or self.year_min is not None

    def describe(self) -> dict[str, object]:
        """Audit-friendly description with empty values dropped."""
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in self.model_dump().items()
            if value not in (None, (), "relevance")
        }

    def cache_key_part(self) -> str:
        """Stable string used in cache keys."""
        return "|".join(
            (
                f"y={self.year_min or ''}",
                f"y2={self.year_max or ''}",
                f"a={self.author or ''}",
                f"t={','.join(self.publication_types)}",
                f"s={self.sort}",
            )
        )


class ProviderQuery(BaseModel):
    """One provider request in the audit trail."""

    model_config = ConfigDict(frozen=True)

    provider: str
    operation: str
    query: str
    ok: bool
    records: int = 0
    cache_hit: bool = False
    http_requests: int = 0
    seconds: float = 0.0
    failure: ProviderFailure | None = None


class DiscoveryAudit(BaseModel):
    """Everything needed to reproduce and explain a discovery request."""

    model_config = ConfigDict(frozen=True)

    raw_query: str
    normalized_query: str
    operation: str = "search"
    providers_requested: tuple[str, ...] = ()
    providers_succeeded: tuple[str, ...] = ()
    providers_failed: tuple[str, ...] = ()
    failures: tuple[ProviderFailure, ...] = ()
    filters: dict[str, object] = Field(default_factory=dict)
    queries: tuple[ProviderQuery, ...] = ()
    provider_records: int = 0
    merged_records: int = 0
    duplicates_merged: int = 0
    local_matches: int = 0
    possible_matches: int = 0
    new_records: int = 0
    conflicts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    refresh: bool = False
    #: True when every provider failed, so the caller knows this is not a "no results".
    all_providers_failed: bool = False
    timestamp: str = ""

    @property
    def any_provider_succeeded(self) -> bool:
        """Whether at least one provider produced a usable response."""
        return bool(self.providers_succeeded)


class DiscoveryResult(BaseModel):
    """The outcome of one discovery request."""

    model_config = ConfigDict(frozen=True)

    audit: DiscoveryAudit
    records: tuple[ExternalWork, ...] = ()

    @property
    def new_records(self) -> tuple[ExternalWork, ...]:
        """Records that are not in the local corpus."""
        return tuple(
            record for record in self.records if not record.local_match.already_in_corpus
        )

    @property
    def local_records(self) -> tuple[ExternalWork, ...]:
        """Records the local corpus already contains."""
        return tuple(
            record for record in self.records if record.local_match.already_in_corpus
        )

    @property
    def conflict_records(self) -> tuple[ExternalWork, ...]:
        """Records whose providers disagree materially."""
        return tuple(
            record
            for record in self.records
            if record.discovery_status is DiscoveryStatus.CONFLICT
        )

    @property
    def unverified_records(self) -> tuple[ExternalWork, ...]:
        """Discovery-only records, i.e. identity not verified."""
        return tuple(
            record
            for record in self.records
            if record.discovery_status is DiscoveryStatus.EXTERNAL_DISCOVERY
            and not record.local_match.already_in_corpus
        )


class LocalDiscoveryResult(BaseModel):
    """The local half of a combined discovery request (RAG-009 output only)."""

    model_config = ConfigDict(frozen=True)

    query: str
    collection: str
    mode: str
    candidate_k: int
    final_k: int
    chunks: tuple[dict[str, object], ...] = ()
    seconds: float = 0.0
    error: str | None = None


class CombinedDiscovery(BaseModel):
    """Local retrieval and external discovery, kept strictly separate."""

    model_config = ConfigDict(frozen=True)

    query: str
    local: LocalDiscoveryResult
    external: DiscoveryResult
    timestamp: str = _datetime.datetime.now(_datetime.timezone.utc).isoformat(
        timespec="seconds"
    )

    @property
    def external_available(self) -> bool:
        """Whether any external provider answered."""
        return bool(self.external.audit.any_provider_succeeded)
