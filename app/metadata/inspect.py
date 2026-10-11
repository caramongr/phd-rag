"""Development CLI for inspecting a document's academic metadata.

Usage::

    python -m app.metadata.inspect "path\\to\\paper.pdf" [--offline]

It prints the preferred metadata and the provenance (source / confidence) of
each field. It never prints document contents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.metadata.enrichment import build_default_enricher
from app.metadata.corpus import physical_paths_for_hash
from app.metadata.extractor import extract_metadata
from app.metadata.models import AcademicMetadata, MetadataValue
from app.parsing.exceptions import DocumentParseError, UnsupportedDocumentFormat

_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", "Title"),
    ("authors", "Authors"),
    ("year", "Year"),
    ("doi", "DOI"),
    ("venue", "Venue"),
    ("publisher", "Publisher"),
    ("publication_type", "Type"),
    ("url", "URL"),
    ("arxiv_id", "arXiv"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.metadata.inspect",
        description="Inspect academic metadata and provenance for one document.",
    )
    parser.add_argument("path", type=Path, help="Document to inspect.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not contact Crossref or OpenAlex.",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Also print the full candidate history for each field.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Corpus root used to find every physical copy of this file, so the "
            "corpus role is resolved across all of them (a duplicate under "
            "my_work/ or summaries/ must not become evidence because another copy "
            "sits under literature/)."
        ),
    )
    return parser


def _format_value(field_name: str, value: MetadataValue) -> str:
    raw = value.value
    if field_name == "authors":
        return ", ".join(author.full_name for author in raw) if raw else "missing"
    return str(raw)


def _print_role(metadata: AcademicMetadata) -> None:
    """Print the effective role and, when copies disagree, why it was chosen."""
    print(f"Corpus role: {metadata.corpus_role.value}")
    resolution = metadata.role_resolution
    if resolution is None:
        return
    print(f"Role resolution: {resolution.describe()}")
    if resolution.has_conflicting_roles:
        for physical in resolution.physical_roles:
            print(f"  {physical.role.value:<16} {physical.path}")
        print(
            "  note: identical bytes in several roles are one logical document; the "
            "most conservative role (non-evidence) is used for all of its chunks"
        )


def _print_metadata(metadata: AcademicMetadata) -> None:
    print(f"File: {metadata.source_path}")
    _print_role(metadata)
    print()
    print("Preferred metadata:")
    for field_name, label in _FIELDS:
        value = getattr(metadata, field_name)
        rendered = _format_value(field_name, value) if value is not None else "missing"
        print(f"{label}: {rendered}")

    print()
    print("Provenance:")
    for field_name, label in _FIELDS:
        value = getattr(metadata, field_name)
        if value is None:
            print(f"{label}: –")
        else:
            print(f"{label}: {value.source.value} / {value.confidence.value}")

    if metadata.external_match_status is not None:
        print()
        print(f"External match status: {metadata.external_match_status.value}")

    if metadata.conflicts:
        print()
        print("Conflicts (external value withheld, both candidates preserved):")
        for conflict in metadata.conflicts:
            print(
                f"  {conflict.field}: local={conflict.local_value!r} vs "
                f"{conflict.external_source.value}={conflict.external_value!r} "
                f"(similarity={conflict.similarity}, route={conflict.route})"
            )


def _print_candidates(metadata: AcademicMetadata) -> None:
    print()
    print("Candidate history:")
    for field_name in metadata.field_names():
        print(f"  {field_name}:")
        for candidate in metadata.candidates[field_name]:
            preview = _candidate_preview(field_name, candidate)
            print(
                f"    - [{candidate.source.value} / {candidate.confidence.value}] "
                f"{preview} ({candidate.evidence})"
            )


def _candidate_preview(field_name: str, candidate: MetadataValue) -> str:
    if field_name == "authors":
        return ", ".join(author.full_name for author in candidate.value)
    if field_name == "abstract":
        text = str(candidate.value)
        return text[:80] + ("…" if len(text) > 80 else "")
    return str(candidate.value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inspector and return an exit code."""
    args = _build_parser().parse_args(argv)
    path: Path = args.path

    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    try:
        physical_paths: list[str] | None = None
        if args.corpus is not None:
            from app.ingestion.file_hash import calculate_sha256

            locations = physical_paths_for_hash(args.corpus, calculate_sha256(path))
            physical_paths = locations or None
        metadata = extract_metadata(path, physical_paths=physical_paths)
    except UnsupportedDocumentFormat as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except DocumentParseError as exc:
        print(f"Error: could not read document: {exc}", file=sys.stderr)
        return 3

    if not args.offline:
        enricher = build_default_enricher()
        metadata = enricher.enrich(metadata)

    _print_metadata(metadata)
    if args.show_candidates:
        _print_candidates(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
