"""DOI-only corpus enrichment stage (RAG-006, stage 1).

Performs exact DOI lookups (Crossref, with OpenAlex as the existing DOI fallback)
for every logical unique document that carries a local DOI, applies the
strong-local-title conflict safeguard, and reports the outcome. It never issues a
title-based search and never modifies a source file.

Usage::

    python -m app.metadata.doi_stage "data" [--out docs/rag006-doi-stage.txt]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.ingestion.scanner import scan_directory
from app.metadata.corpus import is_enrichable
from app.metadata.enrichment import (
    EnrichmentComponents,
    EnrichmentStats,
    MetadataEnricher,
    build_components,
    has_strong_local_title,
)
from app.metadata.enums import ExternalMatchStatus, PublicationType
from app.metadata.extractor import extract_metadata
from app.metadata.models import AcademicMetadata
from app.parsing.exceptions import DocumentParseError, EncryptedDocumentError

_COVERAGE_FIELDS: tuple[tuple[str, str], ...] = (
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


@dataclass
class FieldCoverage:
    """How many documents have each field populated (or known)."""

    title: int = 0
    authors: int = 0
    year: int = 0
    doi: int = 0
    venue: int = 0
    publisher: int = 0
    publication_type: int = 0
    url: int = 0
    abstract: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name, _label in _COVERAGE_FIELDS}


@dataclass
class ConflictRow:
    filename: str
    doi: str
    local_title: str
    external_title: str
    similarity: float
    external_source: str


@dataclass
class UncertainRow:
    filename: str
    doi: str


@dataclass
class UnresolvedRow:
    filename: str
    doi: str
    status: str


@dataclass
class DoiStageResult:
    unique_documents: int = 0
    with_local_doi: int = 0
    eligible_for_lookup: int = 0
    skipped_not_enrichable: int = 0
    stats: EnrichmentStats = field(default_factory=EnrichmentStats)
    before: FieldCoverage = field(default_factory=FieldCoverage)
    after: FieldCoverage = field(default_factory=FieldCoverage)
    conflicts: list[ConflictRow] = field(default_factory=list)
    accepted_without_strong_title: list[UncertainRow] = field(default_factory=list)
    unresolved: list[UnresolvedRow] = field(default_factory=list)
    http_requests: int = 0
    http_requests_by_host: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    network_errors: int = 0
    http_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0


def field_coverage(metadatas: list[AcademicMetadata]) -> FieldCoverage:
    coverage = FieldCoverage()
    for metadata in metadatas:
        if metadata.title is not None:
            coverage.title += 1
        if metadata.authors is not None and metadata.authors.value:
            coverage.authors += 1
        if metadata.year is not None:
            coverage.year += 1
        if metadata.doi is not None:
            coverage.doi += 1
        if metadata.venue is not None:
            coverage.venue += 1
        if metadata.publisher is not None:
            coverage.publisher += 1
        if (
            metadata.publication_type is not None
            and metadata.publication_type.value is not PublicationType.UNKNOWN
        ):
            coverage.publication_type += 1
        if metadata.url is not None:
            coverage.url += 1
        if metadata.abstract is not None:
            coverage.abstract += 1
    return coverage


def run_doi_stage(
    root: str | Path,
    *,
    enricher: MetadataEnricher,
    components: EnrichmentComponents,
    limit: int = 0,
) -> DoiStageResult:
    """Run DOI-only enrichment across the corpus at ``root`` (read-only)."""
    scan = scan_directory(Path(root))
    result = DoiStageResult(unique_documents=scan.unique_documents)

    before_metadatas: list[AcademicMetadata] = []
    after_metadatas: list[AcademicMetadata] = []
    documents = scan.documents[:limit] if limit > 0 else scan.documents

    for document in documents:
        path = Path(document.canonical_path)
        try:
            before = extract_metadata(
                path,
                file_hash=document.file_hash,
                physical_paths=[location.path for location in document.physical_locations],
            )
        except (DocumentParseError, EncryptedDocumentError):
            continue

        enrichable = is_enrichable(
            before.corpus_role, has_doi=before.doi is not None
        )
        after = enricher.enrich(before)

        before_metadatas.append(before)
        after_metadatas.append(after)

        if before.doi is None:
            continue

        result.with_local_doi += 1
        if not enrichable:
            result.skipped_not_enrichable += 1
            result.unresolved.append(
                UnresolvedRow(after.filename, before.doi.value, "ROLE_NOT_ENRICHABLE")
            )
            continue

        result.eligible_for_lookup += 1
        status = after.external_match_status
        status_value = status.value if status is not None else "NO_MATCH"

        for conflict in after.conflicts:
            result.conflicts.append(
                ConflictRow(
                    filename=after.filename,
                    doi=before.doi.value,
                    local_title=conflict.local_value,
                    external_title=conflict.external_value,
                    similarity=conflict.similarity,
                    external_source=conflict.external_source.value,
                )
            )

        if status is ExternalMatchStatus.EXACT:
            if not has_strong_local_title(before):
                result.accepted_without_strong_title.append(
                    UncertainRow(after.filename, before.doi.value)
                )
        else:
            result.unresolved.append(
                UnresolvedRow(after.filename, before.doi.value, status_value)
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


def render(result: DoiStageResult) -> str:
    """Render the DOI-stage report as plain text."""
    lines: list[str] = []
    stats = result.stats

    lines.append("RAG-006 Stage 1 - DOI-only corpus enrichment")
    lines.append("=" * 44)
    lines.append(f"unique documents examined:              {result.unique_documents}")
    lines.append(f"documents with local DOI:               {result.with_local_doi}")
    lines.append(f"eligible for external DOI lookup:       {result.eligible_for_lookup}")
    lines.append(f"  skipped (corpus role not enrichable): {result.skipped_not_enrichable}")
    lines.append(f"  enrichable without a DOI:             {stats.eligible_no_doi}")
    lines.append("")
    lines.append(f"Crossref exact matches:                 {stats.crossref_exact}")
    lines.append(f"OpenAlex DOI fallback matches:          {stats.openalex_fallback_exact}")
    lines.append(f"EXACT accepted:                         {stats.accepted}")
    lines.append(f"CONFLICT:                               {stats.conflicts}")
    lines.append(f"NO_MATCH:                               {stats.no_match}")
    lines.append(f"title-based lookups performed:          {stats.title_lookups}")
    lines.append("")
    lines.append(f"lookup/network failures:                {result.network_errors + result.http_errors}")
    lines.append(f"  network errors:                       {result.network_errors}")
    lines.append(f"  HTTP errors (excluding 404):          {result.http_errors}")
    lines.append(f"retries:                                {result.retries}")
    lines.append(f"cached responses used:                  {result.cache_hits}")
    lines.append(f"cache misses:                           {result.cache_misses}")
    lines.append(f"cache entries now:                      {result.cache_entries}")
    lines.append(f"API requests made:                      {result.http_requests}")
    for host, count in sorted(result.http_requests_by_host.items()):
        lines.append(f"  {host}: {count}")
    lines.append("")
    lines.append("accepted records by external source:")
    for source, count in sorted(stats.accepted_by_source.items()):
        lines.append(f"  {source}: {count}")
    if stats.conflicts_by_source:
        lines.append("conflicts by external source:")
        for source, count in sorted(stats.conflicts_by_source.items()):
            lines.append(f"  {source}: {count}")

    lines.append("")
    lines.append("Metadata coverage (documents with the field populated):")
    lines.append(f"  {'field':<18}{'before':>8}{'after':>8}{'delta':>8}")
    before = result.before.as_dict()
    after = result.after.as_dict()
    for _name, label in _COVERAGE_FIELDS:
        key = _name
        value_before = before.get(key, 0)
        value_after = after.get(key, 0)
        lines.append(
            f"  {label:<18}{value_before:>8}{value_after:>8}{value_after - value_before:>+8}"
        )

    lines.append("")
    lines.append(f"CONFLICT (external DOI record withheld) - {len(result.conflicts)}")
    for row in result.conflicts:
        lines.append(f"  {row.filename}")
        lines.append(f"    DOI:             {row.doi}")
        lines.append(f"    local title:     {row.local_title}")
        lines.append(f"    external title:  {row.external_title}")
        lines.append(f"    similarity:      {row.similarity}")
        lines.append(f"    external source: {row.external_source}")

    lines.append("")
    lines.append(
        "DOI-exact accepted WITHOUT a strong local title (no cross-check possible) "
        f"- {len(result.accepted_without_strong_title)}"
    )
    for row in result.accepted_without_strong_title:
        lines.append(f"  {row.filename}  ({row.doi})")

    lines.append("")
    lines.append(f"DOI-bearing documents that remain unresolved - {len(result.unresolved)}")
    for row in result.unresolved:
        lines.append(f"  {row.filename}  ({row.doi})  [{row.status}]")

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.metadata.doi_stage",
        description="DOI-only corpus enrichment (no title-based searches).",
    )
    parser.add_argument("root", type=Path, help="Corpus root to scan.")
    parser.add_argument("--out", type=Path, default=None, help="Write the report here.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N unique documents (0 = all).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DOI-only stage and print/write the report."""
    args = _build_parser().parse_args(argv)

    components = build_components()
    enricher = MetadataEnricher(
        crossref=components.crossref,
        openalex=components.openalex,
        enabled=True,
        title_lookup_enabled=False,
    )

    result = run_doi_stage(
        args.root, enricher=enricher, components=components, limit=args.limit
    )
    report = render(result)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
