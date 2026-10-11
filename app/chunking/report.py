"""Read-only corpus chunking report (RAG-007).

Usage::

    python -m app.chunking.report data [--out docs/rag007-chunking.txt] [--limit N] [--no-ocr]

Runs the full RAG-007 chunking pass over the supported corpus (PDF, DOCX,
Markdown) and reports the metrics the task requires: document counts, chunk
counts by eligibility, token-size statistics and distribution, content-type /
corpus-role / text-source distributions, page-crossing and OCR involvement, and
every chunk above the hard maximum. It never writes to the corpus.

XLSX is deliberately **not** chunked: flattening spreadsheet cells into prose
would be wrong, so spreadsheets are reported as unsupported for chunking in
RAG-007 and counted rather than silently dropped.
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.chunking.corpus import (
    CHUNKABLE_EXTENSIONS,
    STATUS_CHUNKING_ERROR,
    STATUS_PARSE_ERROR,
    STATUS_UNSUPPORTED,
    STATUS_WITHOUT_CHUNKS,
    UNSUPPORTED_FOR_CHUNKING,
    iter_corpus_documents,
    scan_corpus,
)
from app.chunking.models import ChunkingOptions
from app.chunking.policy import policy_for_role
from app.metadata.corpus import corpus_role_for_path, resolve_corpus_role

_SIZE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("<200 tokens", 0, 199),
    ("200-699 tokens", 200, 699),
    ("700-1000 tokens", 700, 1000),
    ("1001-1200 tokens", 1001, 1200),
    ("1201-1500 tokens", 1201, 1500),
    (">1500 tokens", 1501, None),
)

#: Report chunks above this size individually (the soft ceiling extension).
REPORT_THRESHOLD_TOKENS = 1200
#: A chunk above this is a failure.
VIOLATION_THRESHOLD_TOKENS = 1500

#: Label for chunks whose format has no page-level extraction source (DOCX,
#: Markdown): RAG-003 does not record one, and RAG-007 will not invent it.
TEXT_SOURCE_NOT_APPLICABLE = "n/a (DOCX/Markdown: no page-level text source)"


class ChunkRow(BaseModel):
    """One reported chunk, referenced back to its document."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    chunk_index: int
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    content_type: str
    corpus_role: str
    retrieval_eligible: bool = True
    evidence_eligible: bool = True
    note: str | None = None


class RoleAuditRow(BaseModel):
    """One logical document whose physical copies imply more than one role."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    file_hash: str
    physical_roles: list[str]
    effective_role_before: str
    effective_role_after: str
    evidence_eligible_before: bool
    evidence_eligible_after: bool
    detail: str


class CorpusChunkReport(BaseModel):
    """Aggregate outcome of chunking the corpus."""

    model_config = ConfigDict(extra="forbid")

    # Documents
    unique_documents: int = 0
    documents_chunked: int = 0
    documents_without_chunks: int = 0
    documents_skipped: int = 0
    parse_errors: int = 0
    chunking_errors: int = 0
    skipped_details: list[str] = Field(default_factory=list)
    unsupported_for_chunking: dict[str, int] = Field(default_factory=dict)
    unsupported_paths: list[str] = Field(default_factory=list)
    by_extension: dict[str, int] = Field(default_factory=dict)

    # Chunks
    total_chunks: int = 0
    retrieval_eligible_chunks: int = 0
    evidence_eligible_chunks: int = 0
    total_tokens: int = 0

    # Token size statistics
    token_min: int = 0
    token_median: int = 0
    token_mean: float = 0.0
    token_p95: int = 0
    token_max: int = 0
    size_histogram: dict[str, int] = Field(default_factory=dict)

    # Distributions
    by_content_type: dict[str, int] = Field(default_factory=dict)
    by_corpus_role: dict[str, int] = Field(default_factory=dict)
    by_text_source: dict[str, int] = Field(default_factory=dict)

    # Structural checks
    chunks_crossing_pages: int = 0
    max_pages_spanned: int = 1
    chunks_with_ocr: int = 0
    chunks_with_mixed_sources: int = 0
    reference_chunks: int = 0
    table_chunks: int = 0
    chunks_ineligible_for_retrieval: int = 0

    # Duplicate/cleanup counters (documented, not silent)
    duplicate_table_blocks_removed: int = 0
    running_lines_removed: int = 0
    headings_without_content: int = 0
    skipped_items: int = 0

    # Oversized chunks
    oversized_chunks: list[ChunkRow] = Field(default_factory=list)
    violations: list[ChunkRow] = Field(default_factory=list)

    # Duplicate copies with conflicting corpus roles
    multi_location_documents: int = 0
    multi_role_documents: int = 0
    role_audit_rows: list[RoleAuditRow] = Field(default_factory=list)

    tokenizer_name: str = ""


def _percentile(ordered: list[int], fraction: float) -> int:
    """Nearest-rank percentile of an already sorted list."""
    if not ordered:
        return 0
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _bucket_for(tokens: int) -> str:
    for label, low, high in _SIZE_BUCKETS:
        if tokens >= low and (high is None or tokens <= high):
            return label
    return _SIZE_BUCKETS[-1][0]


def collect(
    root: Path,
    *,
    options: ChunkingOptions | None = None,
    limit: int = 0,
    ocr: bool | None = None,
) -> CorpusChunkReport:
    """Chunk every supported document under ``root`` and aggregate the metrics."""
    report = CorpusChunkReport()
    scan = scan_corpus(Path(root), limit=limit)
    report.unique_documents = scan.unique_documents

    sizes: list[int] = []
    histogram: dict[str, int] = {label: 0 for label, _low, _high in _SIZE_BUCKETS}
    content_types: collections.Counter[str] = collections.Counter()
    roles: collections.Counter[str] = collections.Counter()
    sources: collections.Counter[str] = collections.Counter()

    for record in iter_corpus_documents(
        Path(root),
        options=options or ChunkingOptions(),
        limit=limit,
        ocr=ocr,
        scan=scan,
    ):
        document = record.document
        extension = record.extension
        report.by_extension[extension] = report.by_extension.get(extension, 0) + 1

        locations = [location.path for location in document.physical_locations]
        if document.canonical_path not in locations:
            locations.insert(0, document.canonical_path)
        if len(locations) > 1:
            report.multi_location_documents += 1
        resolution = resolve_corpus_role(locations)
        if resolution.has_conflicting_roles:
            canonical_role = corpus_role_for_path(document.canonical_path)
            report.multi_role_documents += 1
            report.role_audit_rows.append(
                RoleAuditRow(
                    filename=document.filename,
                    file_hash=document.file_hash,
                    physical_roles=[
                        f"{physical.role.value} ({physical.path})"
                        for physical in resolution.physical_roles
                    ],
                    effective_role_before=canonical_role.value,
                    effective_role_after=resolution.effective_role.value,
                    evidence_eligible_before=policy_for_role(
                        canonical_role
                    ).evidence_eligible,
                    evidence_eligible_after=policy_for_role(
                        resolution.effective_role
                    ).evidence_eligible,
                    detail=resolution.reason,
                )
            )

        if record.status in (STATUS_UNSUPPORTED, STATUS_PARSE_ERROR, STATUS_CHUNKING_ERROR):
            if record.status == STATUS_UNSUPPORTED:
                report.documents_skipped += 1
                report.unsupported_for_chunking[extension.lstrip(".")] = (
                    report.unsupported_for_chunking.get(extension.lstrip("."), 0) + 1
                )
                if extension in UNSUPPORTED_FOR_CHUNKING:
                    report.unsupported_paths.append(document.canonical_path)
            elif record.status == STATUS_PARSE_ERROR:
                report.parse_errors += 1
                report.documents_skipped += 1
            else:
                report.chunking_errors += 1
                report.documents_skipped += 1
            report.skipped_details.append(f"{document.filename}: {record.detail}")
            continue

        result = record.chunked
        if record.status == STATUS_WITHOUT_CHUNKS:
            report.documents_without_chunks += 1
        else:
            report.documents_chunked += 1

        report.duplicate_table_blocks_removed += result.duplicate_table_blocks_removed
        report.running_lines_removed += result.running_lines_removed
        report.headings_without_content += result.headings_without_content
        report.skipped_items += len(result.skipped_items)
        report.tokenizer_name = result.tokenizer_name

        for chunk in result.chunks:
            tokens = chunk.token_count or 0
            report.total_chunks += 1
            report.total_tokens += tokens
            sizes.append(tokens)
            histogram[_bucket_for(tokens)] += 1
            content_types[chunk.content_type.value] += 1
            roles[chunk.corpus_role.value if chunk.corpus_role else "unknown"] += 1
            source_key = "+".join(
                sorted(source.value for source in chunk.text_sources)
            ) or TEXT_SOURCE_NOT_APPLICABLE
            sources[source_key] += 1

            if chunk.retrieval_eligible:
                report.retrieval_eligible_chunks += 1
            else:
                report.chunks_ineligible_for_retrieval += 1
            if chunk.evidence_eligible:
                report.evidence_eligible_chunks += 1
            if (
                chunk.page_start is not None
                and chunk.page_end is not None
                and chunk.page_end != chunk.page_start
            ):
                report.chunks_crossing_pages += 1
            if chunk.page_start is not None and chunk.page_end is not None:
                report.max_pages_spanned = max(
                    report.max_pages_spanned, chunk.page_end - chunk.page_start + 1
                )
            if chunk.ocr_involved:
                report.chunks_with_ocr += 1
            if len(chunk.text_sources) > 1:
                report.chunks_with_mixed_sources += 1
            if chunk.content_type.value == "references":
                report.reference_chunks += 1
            if chunk.content_type.value == "table":
                report.table_chunks += 1

            if tokens > REPORT_THRESHOLD_TOKENS:
                row = ChunkRow(
                    filename=document.filename,
                    chunk_index=chunk.chunk_index,
                    token_count=tokens,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    heading_path=list(chunk.heading_path),
                    content_type=chunk.content_type.value,
                    corpus_role=chunk.corpus_role.value if chunk.corpus_role else "unknown",
                    retrieval_eligible=chunk.retrieval_eligible,
                    evidence_eligible=chunk.evidence_eligible,
                    note=(
                        "split continuation"
                        if chunk.split_from_previous
                        else "single structural unit (table or intact paragraph)"
                    ),
                )
                report.oversized_chunks.append(row)
                if tokens > VIOLATION_THRESHOLD_TOKENS:
                    report.violations.append(row)

    report.size_histogram = histogram
    report.by_content_type = dict(sorted(content_types.items()))
    report.by_corpus_role = dict(sorted(roles.items()))
    report.by_text_source = dict(sorted(sources.items()))
    report.unsupported_for_chunking = dict(sorted(report.unsupported_for_chunking.items()))

    if sizes:
        ordered = sorted(sizes)
        report.token_min = ordered[0]
        report.token_median = _percentile(ordered, 0.5)
        report.token_mean = round(sum(ordered) / len(ordered), 1)
        report.token_p95 = _percentile(ordered, 0.95)
        report.token_max = ordered[-1]
    return report


def _counts_block(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"{title}:"]
    if not counts:
        lines.append("  (none)")
        return lines
    width = max(len(name) for name in counts)
    for name, count in counts.items():
        lines.append(f"  {name:<{width}}  {count}")
    return lines


def render(report: CorpusChunkReport) -> str:
    """Render the corpus chunking report as text."""
    lines: list[str] = []
    lines.append("RAG-007 chunking report")
    lines.append("======================")
    lines.append("")
    lines.append("Documents")
    lines.append("---------")
    lines.append(f"unique documents considered:      {report.unique_documents}")
    lines.append(f"documents successfully chunked:   {report.documents_chunked}")
    lines.append(f"documents without any chunk:      {report.documents_without_chunks}")
    lines.append(f"documents skipped:                {report.documents_skipped}")
    lines.append(f"parsing errors:                   {report.parse_errors}")
    lines.append(f"chunking errors:                  {report.chunking_errors}")
    lines.append("")
    lines.extend(_counts_block("by extension", report.by_extension))
    lines.append("")
    if report.unsupported_for_chunking:
        lines.append("unsupported for chunking in RAG-007 (counted, not discarded):")
        for extension, count in report.unsupported_for_chunking.items():
            lines.append(f"  .{extension}: {count}")
        lines.append("")
    lines.append("Chunks")
    lines.append("------")
    lines.append(f"total chunks:                     {report.total_chunks}")
    lines.append(f"retrieval-eligible chunks:        {report.retrieval_eligible_chunks}")
    lines.append(f"evidence-eligible chunks:         {report.evidence_eligible_chunks}")
    lines.append(f"chunks ineligible for retrieval:  {report.chunks_ineligible_for_retrieval}")
    lines.append(f"total tokens:                     {report.total_tokens}")
    lines.append(f"tokenizer:                        {report.tokenizer_name or 'unknown'}")
    lines.append("")
    lines.append("Token counts")
    lines.append("------------")
    lines.append(f"min:    {report.token_min}")
    lines.append(f"median: {report.token_median}")
    lines.append(f"mean:   {report.token_mean}")
    lines.append(f"p95:    {report.token_p95}")
    lines.append(f"max:    {report.token_max}")
    lines.append("")
    lines.append("Size distribution")
    lines.append("-----------------")
    for label, count in report.size_histogram.items():
        lines.append(f"  {label:<18} {count}")
    lines.append("")
    lines.extend(_counts_block("by content type", report.by_content_type))
    lines.append("")
    lines.extend(_counts_block("by corpus role", report.by_corpus_role))
    lines.append("")
    lines.extend(_counts_block("by text source", report.by_text_source))
    lines.append("")
    lines.append("Duplicate copies and corpus roles")
    lines.append("---------------------------------")
    lines.append(
        f"documents with more than one physical copy: {report.multi_location_documents}"
    )
    lines.append(
        f"documents whose copies imply >1 role:       {report.multi_role_documents}"
    )
    for row in report.role_audit_rows:
        lines.append(f"  {row.filename}  ({row.file_hash[:12]}…)")
        for physical in row.physical_roles:
            lines.append(f"    location: {physical}")
        lines.append(
            f"    effective role: {row.effective_role_before} -> "
            f"{row.effective_role_after}"
        )
        lines.append(
            f"    evidence eligible: {row.evidence_eligible_before} -> "
            f"{row.evidence_eligible_after}"
        )
        lines.append(f"    reason: {row.detail}")
    lines.append("")
    lines.append("Structure and provenance")
    lines.append("------------------------")
    lines.append(f"chunks crossing page boundaries:  {report.chunks_crossing_pages}")
    lines.append(f"max pages spanned by one chunk:   {report.max_pages_spanned}")
    lines.append(f"chunks containing OCR text:       {report.chunks_with_ocr}")
    lines.append(f"chunks with mixed text sources:   {report.chunks_with_mixed_sources}")
    lines.append(f"reference chunks:                 {report.reference_chunks}")
    lines.append(f"table chunks:                     {report.table_chunks}")
    lines.append(f"duplicate table blocks removed:   {report.duplicate_table_blocks_removed}")
    lines.append(f"running header/footer lines removed: {report.running_lines_removed}")
    lines.append(f"headings without content:         {report.headings_without_content}")
    lines.append(f"skipped parsed items:             {report.skipped_items}")
    lines.append("")
    lines.append(
        f"chunks above {REPORT_THRESHOLD_TOKENS} tokens: {len(report.oversized_chunks)}"
    )
    for row in report.oversized_chunks:
        section = " > ".join(row.heading_path) or "–"
        lines.append(
            f"  {row.token_count:5d} tokens  {row.filename} [{row.chunk_index}]"
            f"  p{row.page_start}-{row.page_end}  {row.content_type}  {section}"
            f"  ({row.note})"
        )
    lines.append("")
    lines.append(
        f"chunks above {VIOLATION_THRESHOLD_TOKENS} tokens (failures): "
        f"{len(report.violations)}"
    )
    for row in report.violations:
        section = " > ".join(row.heading_path) or "–"
        lines.append(
            f"  {row.token_count:5d} tokens  {row.filename} [{row.chunk_index}]  {section}"
        )
    if report.skipped_details:
        lines.append("")
        lines.append("Skipped documents")
        lines.append("-----------------")
        for detail in report.skipped_details:
            lines.append(f"  {detail}")
    lines.append("")
    lines.append("No source file was modified: chunking reads parsed content only.")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.chunking.report",
        description="Read-only corpus chunking report (RAG-007).",
    )
    parser.add_argument("root", type=Path, help="Corpus root to scan.")
    parser.add_argument("--out", type=Path, default=None, help="Write the report here.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N unique documents (0 = all).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip the OCR fallback for PDFs (native text only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus chunking report and print/write it."""
    args = _build_parser().parse_args(argv)
    report = collect(args.root, limit=args.limit, ocr=False if args.no_ocr else None)
    text = render(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
