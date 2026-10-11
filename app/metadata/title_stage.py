"""Title-based enrichment validation sample (RAG-006, stage 2A).

Enriches a *sample* of documents that have **no locally observed DOI** but do have
a strong local title, using a Crossref title search with OpenAlex as fallback.
Nothing is merged unless the title match is ``STRONG``; ``AMBIGUOUS`` and
``NO_MATCH`` results change nothing. A DOI supplied by a title match is recorded as
a Crossref/OpenAlex candidate (``MEDIUM`` confidence, evidence
``strong title-based external match``) so it can never be confused with a DOI that
was observed in the document itself.

Usage::

    python -m app.metadata.title_stage "data" [--size 30] [--out docs/rag006-stage2a-sample.txt]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.ingestion.scanner import scan_directory
from app.metadata.corpus import is_enrichable
from app.metadata.doi_stage import ConflictRow, FieldCoverage, field_coverage
from app.metadata.enrichment import (
    EnrichmentComponents,
    EnrichmentStats,
    MetadataEnricher,
    build_components,
    local_first_author_family,
    strong_title_source,
)
from app.metadata.enums import ExternalMatchStatus, MetadataSource, PublicationType
from app.metadata.external.matching import (
    REASON_AUTHOR,
    REASON_BELOW_MINIMUM,
    REASON_BELOW_STRONG,
    REASON_MULTIPLE_RECORDS,
    REASON_NO_CANDIDATES,
    REASON_YEAR,
    TitleMatchExplanation,
    explain_title_match,
)
from app.metadata.extractor import extract_metadata
from app.metadata.models import AcademicMetadata
from app.metadata.text import title_similarity
from app.parsing.exceptions import DocumentParseError, EncryptedDocumentError

#: Fields whose coverage is compared before/after within the Stage-2A sample.
_COVERAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("authors", "authors"),
    ("year", "year"),
    ("doi", "DOI"),
    ("venue", "venue"),
    ("publisher", "publisher"),
    ("publication_type", "publication type"),
    ("abstract", "abstract"),
)

#: Fields reported for the whole corpus in Stage 2B (the full field set).
_CORPUS_COVERAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", "title"),
    ("authors", "authors"),
    ("year", "year"),
    ("doi", "DOI"),
    ("venue", "venue"),
    ("publisher", "publisher"),
    ("publication_type", "publication type"),
    ("url", "URL"),
    ("abstract", "abstract"),
)

_BAND_SHORT = "short (<=6 words)"
_BAND_MEDIUM = "medium (7-12 words)"
_BAND_LONG = "long (>=13 words)"


def _length_band(word_count: int) -> str:
    if word_count <= 6:
        return _BAND_SHORT
    if word_count <= 12:
        return _BAND_MEDIUM
    return _BAND_LONG


def is_title_sample_eligible(metadata: AcademicMetadata) -> bool:
    """Return ``True`` for a DOI-less, enrichable document with a strong title."""
    if metadata.doi is not None:
        return False
    if not is_enrichable(metadata.corpus_role, has_doi=False):
        return False
    return strong_title_source(metadata) is not None


def _bucket(metadata: AcademicMetadata) -> tuple[str, str, str, str]:
    title = metadata.title
    source = title.source.value if title is not None else "none"
    band = _length_band(len(title.value.split())) if title is not None else "none"
    year = "year" if metadata.year is not None else "no-year"
    authors = "authors" if (metadata.authors and metadata.authors.value) else "no-authors"
    return (source, band, year, authors)


def _spread(values: list[AcademicMetadata]) -> list[AcademicMetadata]:
    """Order a bucket oldest/newest first so the sample spans publication years."""
    ordered = sorted(values, key=lambda item: (item.year.value if item.year else 0, item.filename))
    spread: list[AcademicMetadata] = []
    low, high = 0, len(ordered) - 1
    while low <= high:
        spread.append(ordered[low])
        if low != high:
            spread.append(ordered[high])
        low += 1
        high -= 1
    return spread


def select_title_sample(
    metadatas: list[AcademicMetadata], size: int | None = 30
) -> list[AcademicMetadata]:
    """Choose a deterministic, representative sample of DOI-less documents.

    Every eligible document is bucketed by (title source, title length band, year
    presence, author presence); buckets are then filled round-robin so short and
    long titles, embedded-metadata and first-page titles, documents with and
    without year/authors are all represented. Within a bucket, documents are taken
    oldest-first/newest-first alternately so both recent and older papers appear.
    ``size=None`` returns the whole eligible population in that same order.
    """
    if size is not None and size <= 0:
        return []

    grouped: dict[tuple[str, str, str, str], list[AcademicMetadata]] = {}
    for metadata in sorted(metadatas, key=lambda item: item.filename):
        if not is_title_sample_eligible(metadata):
            continue
        grouped.setdefault(_bucket(metadata), []).append(metadata)

    for key in grouped:
        grouped[key] = _spread(grouped[key])

    sample: list[AcademicMetadata] = []
    cursor = 0
    keys = sorted(grouped)
    while size is None or len(sample) < size:
        progressed = False
        for key in keys:
            values = grouped[key]
            if cursor < len(values):
                sample.append(values[cursor])
                progressed = True
                if size is not None and len(sample) >= size:
                    break
        if not progressed:
            break
        cursor += 1
    return sample


@dataclass
class TitleSampleRow:
    """One sampled document, before and after the title-based lookup."""

    filename: str
    corpus_role: str
    local_title: str | None
    local_title_source: str | None
    local_title_confidence: str | None
    local_year: int | None
    local_first_author: str | None
    provider: str | None
    external_title: str | None
    similarity: float | None
    external_year: int | None
    external_first_author: str | None
    status: str
    accepted: bool
    resolved_title: str | None
    external_doi: str | None
    doi_origin: str | None
    venue: str | None
    publication_type: str | None
    publisher: str | None
    has_abstract: bool


@dataclass
class SampleComposition:
    """How the selected sample is distributed across the requested axes."""

    by_title_source: dict[str, int] = field(default_factory=dict)
    by_length_band: dict[str, int] = field(default_factory=dict)
    by_year: dict[str, int] = field(default_factory=dict)
    by_authors: dict[str, int] = field(default_factory=dict)
    by_local_type: dict[str, int] = field(default_factory=dict)
    with_arxiv_id: int = 0
    year_min: int | None = None
    year_max: int | None = None


@dataclass
class TitleStageResult:
    unique_documents: int = 0
    eligible_population: int = 0
    sample_size: int = 0
    searches_attempted: int = 0
    stats: EnrichmentStats = field(default_factory=EnrichmentStats)
    strong: int = 0
    ambiguous: int = 0
    no_match: int = 0
    rows: list[TitleSampleRow] = field(default_factory=list)
    composition: SampleComposition = field(default_factory=SampleComposition)
    before: FieldCoverage = field(default_factory=FieldCoverage)
    after: FieldCoverage = field(default_factory=FieldCoverage)
    http_requests: int = 0
    http_requests_by_host: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    network_errors: int = 0
    http_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0


def _first_author(metadata: AcademicMetadata) -> str | None:
    if metadata.authors is None or not metadata.authors.value:
        return None
    return metadata.authors.value[0].full_name


def _external_first_author(record) -> str | None:
    if not record or not record.authors:
        return None
    return record.authors[0].full_name


def _external_doi_origin(metadata: AcademicMetadata, external_doi: str | None) -> str | None:
    """Return how the resolved DOI was obtained, distinguishing local vs external."""
    if metadata.doi is None:
        return None
    value = metadata.doi.value
    if external_doi and value.casefold() == external_doi.casefold():
        if metadata.doi.source in (MetadataSource.CROSSREF, MetadataSource.OPENALEX):
            return "title-based external match (not observed in the document)"
    return "observed locally in the document"


def _composition(sample: list[AcademicMetadata]) -> SampleComposition:
    composition = SampleComposition()
    years: list[int] = []
    for metadata in sample:
        title = metadata.title
        source = title.source.value if title is not None else "none"
        composition.by_title_source[source] = composition.by_title_source.get(source, 0) + 1
        if title is not None:
            band = _length_band(len(title.value.split()))
            composition.by_length_band[band] = composition.by_length_band.get(band, 0) + 1
        key = "year" if metadata.year is not None else "no year"
        composition.by_year[key] = composition.by_year.get(key, 0) + 1
        if metadata.year is not None:
            years.append(metadata.year.value)
        authors = "authors" if (metadata.authors and metadata.authors.value) else "no authors"
        composition.by_authors[authors] = composition.by_authors.get(authors, 0) + 1
        local_type = (
            metadata.publication_type.value.value
            if metadata.publication_type is not None
            else PublicationType.UNKNOWN.value
        )
        composition.by_local_type[local_type] = composition.by_local_type.get(local_type, 0) + 1
        if metadata.arxiv_id is not None:
            composition.with_arxiv_id += 1
    if years:
        composition.year_min = min(years)
        composition.year_max = max(years)
    return composition


def run_title_stage(
    root: str | Path,
    *,
    enricher: MetadataEnricher,
    components: EnrichmentComponents,
    size: int | None = 30,
) -> TitleStageResult:
    """Run the title-based validation sample over the corpus at ``root``."""
    scan = scan_directory(Path(root))
    result = TitleStageResult(unique_documents=scan.unique_documents)

    metadatas: list[AcademicMetadata] = []
    for document in scan.documents:
        try:
            metadatas.append(
                extract_metadata(
                    Path(document.canonical_path),
                    file_hash=document.file_hash,
                    physical_paths=[
                        location.path for location in document.physical_locations
                    ],
                )
            )
        except (DocumentParseError, EncryptedDocumentError):
            continue

    result.eligible_population = sum(
        1 for metadata in metadatas if is_title_sample_eligible(metadata)
    )
    sample = select_title_sample(metadatas, size)
    result.sample_size = len(sample)
    result.composition = _composition(sample)

    before_metadatas: list[AcademicMetadata] = []
    after_metadatas: list[AcademicMetadata] = []

    for metadata in sample:
        before = metadata
        after = enricher.enrich(before)
        record = enricher.last_record
        status = enricher.last_status or ExternalMatchStatus.NO_MATCH

        before_metadatas.append(before)
        after_metadatas.append(after)

        if metadata.doi is None and strong_title_source(before) is not None:
            result.searches_attempted += 1
        if status is ExternalMatchStatus.STRONG:
            result.strong += 1
        elif status is ExternalMatchStatus.AMBIGUOUS:
            result.ambiguous += 1
        else:
            result.no_match += 1

        external_title = record.title if record else None
        similarity = (
            title_similarity(before.title.value, external_title)
            if before.title is not None and external_title
            else None
        )
        external_doi = record.doi if record else None
        accepted = status is ExternalMatchStatus.STRONG and record is not None

        result.rows.append(
            TitleSampleRow(
                filename=after.filename,
                corpus_role=after.corpus_role.value,
                local_title=before.title.value if before.title else None,
                local_title_source=before.title.source.value if before.title else None,
                local_title_confidence=before.title.confidence.value if before.title else None,
                local_year=before.year.value if before.year else None,
                local_first_author=_first_author(before),
                provider=record.source.value if record and status is not ExternalMatchStatus.NO_MATCH else None,
                external_title=external_title,
                similarity=similarity,
                external_year=record.year if record else None,
                external_first_author=_external_first_author(record),
                status=status.value,
                accepted=accepted,
                resolved_title=after.title.value if after.title else None,
                external_doi=external_doi,
                doi_origin=_external_doi_origin(after, external_doi),
                venue=after.venue.value if after.venue else None,
                publication_type=(
                    after.publication_type.value.value
                    if after.publication_type is not None
                    else None
                ),
                publisher=after.publisher.value if after.publisher else None,
                has_abstract=after.abstract is not None,
            )
        )

    result.before = field_coverage(before_metadatas)
    result.after = field_coverage(after_metadatas)
    result.stats = enricher.stats
    result.http_requests = components.http.request_count
    result.http_requests_by_host = dict(components.http.requests_by_host)
    result.retries = components.http.retry_count
    result.network_errors = components.http.network_error_count
    result.http_errors = components.http.http_error_count
    result.cache_hits = components.cache.hits
    result.cache_misses = components.cache.misses
    result.cache_entries = components.cache.entry_count
    return result


#: Ambiguity/refusal reasons mapped to the reporting buckets used by Stage 2B.
_REASON_BUCKETS: dict[str, str] = {
    REASON_YEAR: "local year disagreement",
    REASON_AUTHOR: "local author disagreement",
    REASON_MULTIPLE_RECORDS: "multiple external records",
    REASON_BELOW_STRONG: "title similarity below STRONG",
    REASON_BELOW_MINIMUM: "title similarity below minimum",
    REASON_NO_CANDIDATES: "no external candidates",
}


def _file_state(root: Path) -> dict[str, tuple[int, int]]:
    """Return (size, mtime) for every file under ``root`` (integrity check)."""
    state: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            try:
                stat = path.stat()
            except OSError:
                continue
            state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return state


@dataclass
class AcceptedMatchRow:
    """An accepted title-based identity, with the evidence for auditing it."""

    filename: str
    corpus_role: str
    local_title: str
    local_title_source: str
    local_year: int | None
    local_first_author: str | None
    provider: str
    external_title: str
    similarity: float
    external_year: int | None
    external_first_author: str | None
    external_doi: str | None
    venue: str | None
    publisher: str | None
    publication_type: str | None
    has_abstract: bool


@dataclass
class WithheldRow:
    """A searched document whose match was refused, with the deciding reason."""

    filename: str
    corpus_role: str
    status: str
    reason: str
    bucket: str
    similarity: float | None
    local_title: str
    local_year: int | None
    local_first_author: str | None
    matched_title: str | None
    external_year: int | None


@dataclass
class Stage2bResult:
    unique_documents: int = 0
    eligible_population: int = 0
    searches_attempted: int = 0
    strong: int = 0
    ambiguous: int = 0
    no_match: int = 0
    accepted_by_source: dict[str, int] = field(default_factory=dict)
    doi_route_exact: int = 0
    doi_route_conflict: int = 0
    doi_route_no_match: int = 0
    conflicts: list[ConflictRow] = field(default_factory=list)
    local: FieldCoverage = field(default_factory=FieldCoverage)
    after_stage1: FieldCoverage = field(default_factory=FieldCoverage)
    after_stage2b: FieldCoverage = field(default_factory=FieldCoverage)
    accepted: list[AcceptedMatchRow] = field(default_factory=list)
    withheld: list[WithheldRow] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)
    stats: EnrichmentStats = field(default_factory=EnrichmentStats)
    http_requests: int = 0
    http_requests_by_host: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    network_errors: int = 0
    http_errors: int = 0
    http_error_statuses: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0
    files_modified: int = 0


def _explain_withheld(
    metadata: AcademicMetadata,
    *,
    enricher: MetadataEnricher,
    route: str | None,
) -> tuple[str, str, TitleMatchExplanation | None]:
    """Return (reason, bucket, explanation) for a withheld search result."""
    title = metadata.title.value if metadata.title else None
    if title is None:
        return ("no_local_title", "other", None)

    provider = (route or "").split(" ")[0].casefold()
    client = None
    if provider == "openalex":
        client = enricher.openalex
    elif provider == "crossref":
        client = enricher.crossref
    else:
        client = enricher.crossref or enricher.openalex
    if client is None:
        return ("no_provider", "other", None)

    records = client.search_title(title)
    explanation = explain_title_match(
        records,
        local_title=title,
        local_year=metadata.year.value if metadata.year else None,
        local_family=local_first_author_family(metadata),
    )
    reason = explanation.reason or REASON_BELOW_STRONG
    bucket = _REASON_BUCKETS.get(reason, "other")
    if (
        reason == REASON_NO_CANDIDATES
        and len(title.split()) <= 3
    ):
        bucket = "generic/ambiguous title"
    return (reason, bucket, explanation)


def run_stage2b(
    root: str | Path,
    *,
    enricher: MetadataEnricher,
    components: EnrichmentComponents,
) -> Stage2bResult:
    """Enrich the whole corpus: DOI route where a DOI exists, title route otherwise."""
    scan = scan_directory(Path(root))
    result = Stage2bResult(unique_documents=scan.unique_documents)
    root_path = Path(root)
    files_before = _file_state(root_path)

    metadatas: list[AcademicMetadata] = []
    for document in scan.documents:
        try:
            metadatas.append(
                extract_metadata(
                    Path(document.canonical_path),
                    file_hash=document.file_hash,
                    physical_paths=[
                        location.path for location in document.physical_locations
                    ],
                )
            )
        except (DocumentParseError, EncryptedDocumentError):
            continue

    result.eligible_population = sum(
        1 for metadata in metadatas if is_title_sample_eligible(metadata)
    )

    local_metadatas: list[AcademicMetadata] = []
    stage1_metadatas: list[AcademicMetadata] = []
    stage2b_metadatas: list[AcademicMetadata] = []

    for metadata in metadatas:
        has_doi = metadata.doi is not None
        searched = has_doi or is_title_sample_eligible(metadata)
        after = enricher.enrich(metadata)
        status = enricher.last_status

        local_metadatas.append(metadata)
        stage1_metadatas.append(after if has_doi else metadata)
        stage2b_metadatas.append(after)

        if has_doi:
            if status is ExternalMatchStatus.EXACT:
                result.doi_route_exact += 1
            elif status is ExternalMatchStatus.CONFLICT:
                result.doi_route_conflict += 1
            else:
                result.doi_route_no_match += 1
            for conflict in after.conflicts:
                result.conflicts.append(
                    ConflictRow(
                        filename=after.filename,
                        doi=metadata.doi.value if metadata.doi else "",
                        local_title=conflict.local_value,
                        external_title=conflict.external_value,
                        similarity=conflict.similarity,
                        external_source=conflict.external_source.value,
                    )
                )
            continue

        if not searched:
            continue

        result.searches_attempted += 1
        if status is ExternalMatchStatus.STRONG and enricher.last_record is not None:
            record = enricher.last_record
            result.strong += 1
            source = record.source.value
            result.accepted_by_source[source] = result.accepted_by_source.get(source, 0) + 1
            result.accepted.append(
                AcceptedMatchRow(
                    filename=after.filename,
                    corpus_role=after.corpus_role.value,
                    local_title=metadata.title.value if metadata.title else "",
                    local_title_source=metadata.title.source.value if metadata.title else "",
                    local_year=metadata.year.value if metadata.year else None,
                    local_first_author=_first_author(metadata),
                    provider=source,
                    external_title=record.title or "",
                    similarity=title_similarity(
                        metadata.title.value if metadata.title else "", record.title or ""
                    ),
                    external_year=record.year,
                    external_first_author=_external_first_author(record),
                    external_doi=record.doi,
                    venue=after.venue.value if after.venue else None,
                    publisher=after.publisher.value if after.publisher else None,
                    publication_type=(
                        after.publication_type.value.value
                        if after.publication_type is not None
                        else None
                    ),
                    has_abstract=after.abstract is not None,
                )
            )
            continue

        if status is ExternalMatchStatus.AMBIGUOUS:
            result.ambiguous += 1
        else:
            result.no_match += 1
        reason, bucket, explanation = _explain_withheld(
            metadata, enricher=enricher, route=enricher.last_route
        )
        result.reason_counts[bucket] = result.reason_counts.get(bucket, 0) + 1
        result.withheld.append(
            WithheldRow(
                filename=after.filename,
                corpus_role=after.corpus_role.value,
                status=status.value if status is not None else "NO_MATCH",
                reason=reason,
                bucket=bucket,
                similarity=explanation.best_similarity if explanation else None,
                local_title=metadata.title.value if metadata.title else "",
                local_year=metadata.year.value if metadata.year else None,
                local_first_author=_first_author(metadata),
                matched_title=explanation.matched_title if explanation else None,
                external_year=(
                    explanation.record.year
                    if explanation is not None and explanation.record is not None
                    else None
                ),
            )
        )

    result.local = field_coverage(local_metadatas)
    result.after_stage1 = field_coverage(stage1_metadatas)
    result.after_stage2b = field_coverage(stage2b_metadatas)
    result.stats = enricher.stats
    result.http_requests = components.http.request_count
    result.http_requests_by_host = dict(components.http.requests_by_host)
    result.retries = components.http.retry_count
    result.network_errors = components.http.network_error_count
    result.http_errors = components.http.http_error_count
    result.http_error_statuses = dict(components.http.http_error_statuses)
    result.cache_hits = components.cache.hits
    result.cache_misses = components.cache.misses
    result.cache_entries = components.cache.entry_count

    files_after = _file_state(root_path)
    result.files_modified = sum(
        1
        for path, state in files_before.items()
        if files_after.get(path) != state
    ) + len([path for path in files_after if path not in files_before])
    return result


def render_stage2b(result: Stage2bResult) -> str:
    """Render the corpus-wide Stage-2B report as plain text."""
    lines: list[str] = []
    lines.append("RAG-006 Stage 2B - corpus-wide title-based enrichment")
    lines.append("=" * 52)
    lines.append(f"unique documents examined:             {result.unique_documents}")
    lines.append(f"eligible population (DOI-less + strong local title): {result.eligible_population}")
    lines.append(f"title searches attempted:              {result.searches_attempted}")
    lines.append("")
    lines.append(f"STRONG (merged):                       {result.strong}")
    lines.append(f"AMBIGUOUS (nothing merged):            {result.ambiguous}")
    lines.append(f"NO_MATCH (nothing merged):             {result.no_match}")
    lines.append(f"API/network failures:                  {result.network_errors + result.http_errors}")
    lines.append(f"  network errors:                      {result.network_errors}")
    lines.append(f"  HTTP errors (excluding 404):         {result.http_errors}")
    for status, count in sorted(result.http_error_statuses.items()):
        lines.append(f"    HTTP {status}: {count}")
    lines.append(f"retries:                               {result.retries}")
    lines.append("")
    lines.append("accepted matches by external source:")
    for source, count in sorted(result.accepted_by_source.items()):
        lines.append(f"  {source}: {count}")
    lines.append("")
    lines.append("DOI-bearing documents re-checked through the DOI route (unchanged):")
    lines.append(f"  EXACT accepted:                      {result.doi_route_exact}")
    lines.append(f"  CONFLICT (withheld):                 {result.doi_route_conflict}")
    lines.append(f"  NO_MATCH:                            {result.doi_route_no_match}")
    lines.append("")
    lines.append(f"cached responses used:                 {result.cache_hits}")
    lines.append(f"cache misses:                          {result.cache_misses}")
    lines.append(f"cache entries now:                     {result.cache_entries}")
    lines.append(f"API requests made:                     {result.http_requests}")
    for host, count in sorted(result.http_requests_by_host.items()):
        lines.append(f"  {host}: {count}")
    lines.append(f"source files modified:                 {result.files_modified}")

    lines.append("")
    lines.append("Whole-corpus metadata coverage (documents with the field populated):")
    lines.append(f"  {'field':<18}{'local':>8}{'after DOI':>11}{'after title':>13}")
    local = result.local.as_dict()
    after_doi = result.after_stage1.as_dict()
    after_title = result.after_stage2b.as_dict()
    for name, label in _CORPUS_COVERAGE_FIELDS:
        lines.append(
            f"  {label:<18}{local.get(name, 0):>8}{after_doi.get(name, 0):>11}"
            f"{after_title.get(name, 0):>13}"
        )

    lines.append("")
    lines.append(f"withheld matches by reason - {len(result.withheld)}")
    for bucket, count in sorted(result.reason_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {bucket}: {count}")
    lines.append("")
    for row in result.withheld:
        lines.append(
            f"  {row.filename}  [{row.status}]  {row.reason} ({row.bucket})"
            f"  similarity={row.similarity if row.similarity is not None else 'n/a'}"
        )

    lines.append("")
    lines.append(f"accepted identities (audit list) - {len(result.accepted)}")
    for index, row in enumerate(result.accepted, start=1):
        lines.append(f"{index}. {row.filename}  [{row.corpus_role}]")
        lines.append(f"   local title:        {row.local_title!r}")
        lines.append(f"   local title source: {row.local_title_source}")
        lines.append(f"   local year:         {row.local_year}")
        lines.append(f"   local first author: {row.local_first_author}")
        lines.append(f"   external provider:  {row.provider}")
        lines.append(f"   external title:     {row.external_title!r}")
        lines.append(f"   similarity:         {row.similarity:.3f}")
        lines.append(f"   external year:      {row.external_year}")
        lines.append(f"   external 1st author:{row.external_first_author}")
        lines.append(f"   external DOI:       {row.external_doi}")
        lines.append(f"   venue:              {row.venue}")
        lines.append(f"   publication type:   {row.publication_type}")
        lines.append(f"   abstract present:   {'yes' if row.has_abstract else 'no'}")

    lines.append("")
    lines.append("external DOI provenance: every DOI above was recorded as source")
    lines.append("CROSSREF/OPENALEX with confidence MEDIUM and evidence")
    lines.append("'strong title-based external match (<provider> title search); external DOI'.")
    lines.append("No DOI in this stage is represented as observed in the source document.")
    return "\n".join(lines) + "\n"


def _render_composition(composition: SampleComposition) -> list[str]:
    lines = ["sample composition:"]
    for label, mapping in (
        ("title source", composition.by_title_source),
        ("title length", composition.by_length_band),
        ("year", composition.by_year),
        ("authors", composition.by_authors),
        ("local publication type", composition.by_local_type),
    ):
        rendered = ", ".join(f"{key} {value}" for key, value in sorted(mapping.items()))
        lines.append(f"  {label:<24}{rendered}")
    span = "n/a"
    if composition.year_min is not None and composition.year_max is not None:
        span = f"{composition.year_min}-{composition.year_max}"
    lines.append(f"  {'year range':<24}{span}")
    lines.append(f"  {'with arXiv id':<24}{composition.with_arxiv_id}")
    return lines


def render(result: TitleStageResult) -> str:
    """Render the stage-2A sample report as plain text."""
    lines: list[str] = []
    lines.append("RAG-006 Stage 2A - title-based enrichment validation sample")
    lines.append("=" * 60)
    lines.append(f"unique documents examined:        {result.unique_documents}")
    lines.append(f"eligible population (DOI-less + strong title): {result.eligible_population}")
    lines.append(f"sample size:                      {result.sample_size}")
    lines.append(f"title searches attempted:         {result.searches_attempted}")
    lines.append(f"STRONG:                           {result.strong}")
    lines.append(f"AMBIGUOUS:                        {result.ambiguous}")
    lines.append(f"NO_MATCH:                         {result.no_match}")
    lines.append(f"API/network failures:             {result.network_errors + result.http_errors}")
    lines.append(f"  network errors:                 {result.network_errors}")
    lines.append(f"  HTTP errors (excluding 404):    {result.http_errors}")
    lines.append(f"accepted (merged):                {result.stats.accepted}")
    lines.append(f"cached responses used:            {result.cache_hits}")
    lines.append(f"cache misses:                     {result.cache_misses}")
    lines.append(f"API requests made:                {result.http_requests}")
    for host, count in sorted(result.http_requests_by_host.items()):
        lines.append(f"  {host}: {count}")
    lines.append("")
    lines.extend(_render_composition(result.composition))

    lines.append("")
    lines.append("Metadata coverage within the sample (documents with the field populated):")
    lines.append(f"  {'field':<18}{'before':>8}{'after':>8}{'delta':>8}")
    before = result.before.as_dict()
    after = result.after.as_dict()
    for name, label in _COVERAGE_FIELDS:
        value_before = before.get(name, 0)
        value_after = after.get(name, 0)
        lines.append(
            f"  {label:<18}{value_before:>8}{value_after:>8}{value_after - value_before:>+8}"
        )

    lines.append("")
    lines.append("Per-document results")
    lines.append("-" * 20)
    for index, row in enumerate(result.rows, start=1):
        lines.append(f"{index}. {row.filename}  [{row.corpus_role}]")
        lines.append(f"   local title:        {row.local_title!r}")
        lines.append(f"   local title source: {row.local_title_source} ({row.local_title_confidence})")
        lines.append(f"   local year:         {row.local_year}")
        lines.append(f"   local first author: {row.local_first_author}")
        lines.append(f"   external provider:  {row.provider}")
        lines.append(f"   external title:     {row.external_title!r}")
        lines.append(
            "   similarity:         "
            + (f"{row.similarity:.3f}" if row.similarity is not None else "n/a")
        )
        lines.append(f"   external year:      {row.external_year}")
        lines.append(f"   external 1st author:{row.external_first_author}")
        lines.append(f"   match status:       {row.status}")
        lines.append(f"   accepted:           {'yes' if row.accepted else 'no'}")
        lines.append(f"   resolved title:     {row.resolved_title!r}")
        lines.append(f"   external DOI:       {row.external_doi}")
        lines.append(f"   DOI origin:         {row.doi_origin}")
        lines.append(f"   venue:              {row.venue}")
        lines.append(f"   publication type:   {row.publication_type}")
        lines.append(f"   publisher:          {row.publisher}")
        lines.append(f"   abstract present:   {'yes' if row.has_abstract else 'no'}")

    lines.append("")
    lines.append("Unresolved (AMBIGUOUS / NO_MATCH - nothing merged)")
    for row in result.rows:
        if not row.accepted:
            lines.append(f"  {row.filename}  [{row.status}]  similarity={row.similarity}")

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.metadata.title_stage",
        description="Title-based enrichment validation sample (no DOI lookups).",
    )
    parser.add_argument("root", type=Path, help="Corpus root to scan.")
    parser.add_argument("--out", type=Path, default=None, help="Write the report here.")
    parser.add_argument("--size", type=int, default=30, help="Sample size (default 30).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Stage 2B: enrich the whole eligible population instead of a sample.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage-2A sample (or stage 2B corpus run) and print/write it."""
    args = _build_parser().parse_args(argv)

    components = build_components()
    enricher = MetadataEnricher(
        crossref=components.crossref,
        openalex=components.openalex,
        enabled=True,
        title_lookup_enabled=True,
    )

    if args.all:
        result = run_stage2b(args.root, enricher=enricher, components=components)
        report = render_stage2b(result)
    else:
        result = run_title_stage(
            args.root, enricher=enricher, components=components, size=args.size
        )
        report = render(result)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
