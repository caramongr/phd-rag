"""Read-only corpus metadata reporting (RAG-006, validation phases 1 and 2).

Phase 1 (offline) extracts local metadata across every unique document and
reports coverage and source counts. Phase 2 runs external enrichment on a small
representative sample only. Neither phase modifies any source file.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.scanner import scan_directory
from app.metadata.enrichment import MetadataEnricher, build_default_enricher
from app.metadata.enums import (
    CorpusRole,
    ExternalMatchStatus,
    MetadataSource,
    PublicationType,
)
from app.metadata.extractor import extract_metadata
from app.metadata.models import AcademicMetadata
from app.parsing.exceptions import DocumentParseError, EncryptedDocumentError

_REPORTED_SOURCES: tuple[MetadataSource, ...] = (
    MetadataSource.PDF_METADATA,
    MetadataSource.DOCX_PROPERTIES,
    MetadataSource.DOCUMENT_TEXT,
    MetadataSource.FILENAME,
)


class OfflineReport(BaseModel):
    """Aggregate coverage of local metadata extraction (phase 1)."""

    model_config = ConfigDict(extra="forbid")

    unique_documents: int = 0
    by_extension: dict[str, int] = Field(default_factory=dict)
    title_available: int = 0
    title_missing: int = 0
    doi_detected: int = 0
    authors_detected: int = 0
    year_detected: int = 0
    venue_detected: int = 0
    publication_type_known: int = 0
    publication_type_unknown: int = 0
    source_counts: dict[str, int] = Field(default_factory=dict)
    by_corpus_role: dict[str, int] = Field(default_factory=dict)
    extraction_errors: int = 0


class SampleRow(BaseModel):
    """One document's result in the external-enrichment sample (phase 2)."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    source_path: str
    corpus_role: str
    local_title: str | None = None
    local_title_source: str | None = None
    local_doi: str | None = None
    lookup_method: str | None = None
    match_status: str | None = None
    resolved_title: str | None = None
    resolved_year: int | None = None
    resolved_doi: str | None = None
    resolved_type: str | None = None
    resolved_venue: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)


def collect_offline(
    root: str | Path,
) -> tuple[OfflineReport, list[AcademicMetadata]]:
    """Extract local metadata for every unique document under ``root``."""
    result = scan_directory(Path(root))
    report = OfflineReport(unique_documents=result.unique_documents)
    extension_counts: collections.Counter[str] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    metadatas: list[AcademicMetadata] = []

    for document in result.documents:
        path = Path(document.canonical_path)
        extension_counts[document.extension] += 1
        try:
            metadata = extract_metadata(
                path,
                file_hash=document.file_hash,
                physical_paths=[location.path for location in document.physical_locations],
            )
        except (DocumentParseError, EncryptedDocumentError):
            report.extraction_errors += 1
            continue

        metadatas.append(metadata)
        role_counts[metadata.corpus_role.value] += 1

        if metadata.title is not None:
            report.title_available += 1
        else:
            report.title_missing += 1
        if metadata.doi is not None:
            report.doi_detected += 1
        if metadata.authors is not None:
            report.authors_detected += 1
        if metadata.year is not None:
            report.year_detected += 1
        if metadata.venue is not None:
            report.venue_detected += 1
        if (
            metadata.publication_type is not None
            and metadata.publication_type.value is not PublicationType.UNKNOWN
        ):
            report.publication_type_known += 1
        else:
            report.publication_type_unknown += 1

        for values in metadata.candidates.values():
            for candidate in values:
                if candidate.source in _REPORTED_SOURCES:
                    source_counts[candidate.source.value] += 1

    report.by_extension = dict(sorted(extension_counts.items()))
    report.by_corpus_role = dict(sorted(role_counts.items()))
    report.source_counts = {source.value: source_counts.get(source.value, 0) for source in _REPORTED_SOURCES}
    return report, metadatas


def select_sample(metadatas: list[AcademicMetadata], size: int = 20) -> list[AcademicMetadata]:
    """Choose a deterministic, representative sample for external enrichment.

    Buckets are filled in a fixed order so the sample always covers DOI-present,
    DOI-absent-with-title, embedded-title-only, title-missing and arXiv-style
    cases. Only scholarly literature is sampled; the author's own work, rules,
    supervision records and other non-scholarly roles are excluded because they
    must not be enriched.
    """

    def bucket(metadata: AcademicMetadata) -> str:
        if metadata.arxiv_id is not None and metadata.doi is None:
            return "arxiv"
        if metadata.doi is not None:
            return "doi"
        if metadata.title is not None and metadata.title.source is MetadataSource.PDF_METADATA:
            return "embedded_title"
        if metadata.title is not None:
            return "text_title"
        return "no_title"

    order = ("doi", "text_title", "embedded_title", "arxiv", "no_title")
    grouped: dict[str, list[AcademicMetadata]] = {name: [] for name in order}
    for metadata in metadatas:
        if metadata.corpus_role is not CorpusRole.LITERATURE:
            continue
        grouped[bucket(metadata)].append(metadata)

    sample: list[AcademicMetadata] = []
    cursor = 0
    while len(sample) < size:
        progressed = False
        for name in order:
            values = grouped[name]
            if cursor < len(values):
                sample.append(values[cursor])
                progressed = True
                if len(sample) >= size:
                    break
        if not progressed:
            break
        cursor += 1
    return sample


def _local_title_source(metadata: AcademicMetadata) -> str | None:
    return metadata.title.source.value if metadata.title is not None else None


def run_sample(
    metadatas: list[AcademicMetadata],
    *,
    enricher: MetadataEnricher,
    size: int = 20,
) -> list[SampleRow]:
    """Run the enricher over a sample and capture per-document results."""
    rows: list[SampleRow] = []
    for metadata in select_sample(metadatas, size):
        before = metadata
        after = enricher.enrich(metadata)

        conflicts: list[str] = []
        before_title = before.title.value if before.title else None
        after_title = after.title.value if after.title else None
        for conflict in after.conflicts:
            conflicts.append(
                f"{conflict.route} conflict on {conflict.field}: "
                f"local={conflict.local_value!r} vs "
                f"{conflict.external_source.value}={conflict.external_value!r} "
                f"(similarity={conflict.similarity})"
            )
        if (
            before_title
            and after_title
            and before_title != after_title
            and after.external_match_status is not ExternalMatchStatus.CONFLICT
        ):
            route = "DOI" if before.doi is not None else "title"
            conflicts.append(
                f"{route} match replaced title: local={before_title!r} -> external={after_title!r}"
            )

        provenance: dict[str, str] = {}
        for field_name in ("title", "authors", "year", "doi", "venue", "publication_type"):
            value = getattr(after, field_name)
            if value is not None:
                provenance[field_name] = f"{value.source.value}/{value.confidence.value}"

        rows.append(
            SampleRow(
                filename=after.filename,
                source_path=after.source_path,
                corpus_role=after.corpus_role.value,
                local_title=before_title,
                local_title_source=_local_title_source(before),
                local_doi=before.doi.value if before.doi else None,
                lookup_method="DOI" if before.doi else ("title" if before_title else None),
                match_status=(
                    after.external_match_status.value
                    if after.external_match_status is not None
                    else None
                ),
                resolved_title=after_title,
                resolved_year=after.year.value if after.year else None,
                resolved_doi=after.doi.value if after.doi else None,
                resolved_type=(
                    after.publication_type.value if after.publication_type else None
                ),
                resolved_venue=after.venue.value if after.venue else None,
                provenance=provenance,
                conflicts=conflicts,
            )
        )
    return rows


def _print_offline(report: OfflineReport) -> None:
    print("Offline metadata coverage")
    print(f"  unique documents examined: {report.unique_documents}")
    print(f"  extraction errors:         {report.extraction_errors}")
    print("  by extension:")
    for extension, count in report.by_extension.items():
        print(f"    {extension or '(none)'}: {count}")
    print("  by corpus role:")
    for role, count in report.by_corpus_role.items():
        print(f"    {role}: {count}")
    print(f"  title available:           {report.title_available}")
    print(f"  title missing:             {report.title_missing}")
    print(f"  DOI detected:              {report.doi_detected}")
    print(f"  authors detected:          {report.authors_detected}")
    print(f"  year detected:             {report.year_detected}")
    print(f"  venue detected:            {report.venue_detected}")
    print(f"  publication type known:    {report.publication_type_known}")
    print(f"  publication type unknown:  {report.publication_type_unknown}")
    print("  metadata source counts (candidates):")
    for source, count in report.source_counts.items():
        print(f"    {source}: {count}")


def _print_sample(rows: list[SampleRow]) -> None:
    print()
    print(f"External enrichment sample ({len(rows)} documents)")
    for row in rows:
        print()
        print(f"  {row.filename} [{row.corpus_role}]")
        print(f"    local title:   {row.local_title or '—'} ({row.local_title_source or '—'})")
        print(f"    local DOI:     {row.local_doi or '—'}")
        print(f"    lookup:        {row.lookup_method or '—'}")
        print(f"    match status:  {row.match_status or '—'}")
        print(f"    resolved:      title={row.resolved_title or '—'} year={row.resolved_year or '—'} "
              f"doi={row.resolved_doi or '—'}")
        print(f"    type/venue:    {row.resolved_type or '—'} / {row.resolved_venue or '—'}")
        if row.provenance:
            print("    provenance:    " + ", ".join(f"{k}={v}" for k, v in row.provenance.items()))
        if row.conflicts:
            for conflict in row.conflicts:
                print(f"    conflict:      {conflict}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.metadata.report",
        description="Read-only corpus metadata reporting (offline stats and optional sample).",
    )
    parser.add_argument("root", type=Path, help="Corpus root to scan.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Run external enrichment on N documents (0 = offline only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reporter and return an exit code."""
    args = _build_parser().parse_args(argv)
    report, metadatas = collect_offline(args.root)
    _print_offline(report)

    if args.sample > 0:
        enricher = build_default_enricher()
        if not enricher.enabled:
            print("\nExternal enrichment disabled; skipping sample.", file=sys.stderr)
            return 0
        rows = run_sample(metadatas, enricher=enricher, size=args.sample)
        _print_sample(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
