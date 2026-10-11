"""Readable rendering of discovery results (RAG-010 §24-§26).

The default output groups results the way a researcher reads them - already have,
new/verified, discovery only, conflicts - and shows the metadata a reader needs
(title, authors, year, venue, DOI, providers, local-match status, verification status,
open-access status and an abstract preview). Raw provider JSON is only shown with
``--json``, because it is not the useful view.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from app.discovery.models import (
    DiscoveryAudit,
    DiscoveryResult,
    DiscoveryStatus,
    ExternalWork,
    LocalMatchKind,
)


def _shorten(text: str | None, width: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


def format_record(
    record: ExternalWork,
    *,
    index: int | None = None,
    show_abstract: bool = False,
    width: int = 100,
) -> str:
    """Render one external record as a readable block."""
    prefix = f"{index}. " if index is not None else ""
    status = record.discovery_status.value
    year = record.year if record.year is not None else "?"
    venue = _shorten(record.venue or record.publisher, 60) or "(no venue supplied)"
    authors = ", ".join(record.authors[:3]) + (" et al." if len(record.authors) > 3 else "")
    verification = (
        "verified" if record.fulltext_verified else record.fulltext_status.value
    )
    if not record.fulltext_verified and record.fulltext_note:
        verification += f" ({record.fulltext_note})"
    lines = [
        f"  {prefix}[{status}] {_shorten(record.display_title, width)}",
        f"     authors      : {authors or '(not supplied)'}",
        f"     year / venue : {year}  |  {venue}",
        f"     doi          : {record.doi or '(not supplied)'}",
        f"     providers    : {record.provider_label}",
        f"     local        : {_local_summary(record)}",
        f"     verification : identity={record.identity_confidence:.2f} fulltext={verification}",
    ]
    if record.fulltext_verified:
        lines.append(
            f"     fulltext sha : {record.fulltext_sha256} "
            f"({record.fulltext_bytes} bytes, {record.fulltext_content_type}, "
            f"{record.fulltext_fetched_at})"
        )
    if record.cited_by_count is not None:
        lines.append(f"     cited by     : {record.cited_by_count} (context only, not a score)")
    if record.open_access is not None:
        lines.append(
            f"     open access  : {'yes' if record.open_access else 'no'}"
            + (f" ({record.open_access_status})" if record.open_access_status else "")
        )
    if record.fulltext_url:
        lines.append(f"     fulltext url : {_shorten(record.fulltext_url, width)}")
    if record.conflicts:
        for conflict in record.conflicts:
            lines.append(
                f"     CONFLICT {conflict.field_name}: {', '.join(conflict.values)} "
                f"({', '.join(conflict.providers)})"
            )
    if record.missing_fields:
        lines.append(f"     missing      : {', '.join(record.missing_fields)}")
    if show_abstract:
        lines.append(f"     abstract     : {_shorten(record.abstract, width) or '(not supplied)'}")
    return "\n".join(lines)


def _local_summary(record: ExternalWork) -> str:
    match = record.local_match
    if not match.already_in_corpus:
        return "not in local corpus (NEW)"
    role = match.local_corpus_role or "unknown"
    evidence = "evidence" if match.local_evidence_eligible else "NOT evidence"
    return (
        f"{match.kind.value} - {match.match_reason}; {match.local_filename} "
        f"[{role}, {evidence}]"
    )


def format_audit(audit: DiscoveryAudit, *, verbose: bool = True) -> str:
    """Render the audit summary of a discovery request."""
    lines = [
        f"query        : {audit.normalized_query}",
        f"operation    : {audit.operation}"
        + (f"  filters: {audit.filters}" if audit.filters else ""),
        f"providers    : requested={', '.join(audit.providers_requested) or 'none'} "
        f"succeeded={', '.join(audit.providers_succeeded) or 'none'} "
        f"failed={', '.join(audit.providers_failed) or 'none'}",
        f"counts       : provider_records={audit.provider_records} "
        f"merged={audit.merged_records} duplicates_merged={audit.duplicates_merged} "
        f"new={audit.new_records} local={audit.local_matches} "
        f"possible={audit.possible_matches} conflicts={audit.conflicts}",
        f"cache        : hits={audit.cache_hits} misses={audit.cache_misses} "
        f"refresh={audit.refresh}",
        f"timestamp    : {audit.timestamp}",
    ]
    if audit.failures:
        for failure in audit.failures:
            retry = f" retry_after={failure.retry_after:.0f}s" if failure.retry_after else ""
            lines.append(
                f"provider failure: {failure.provider} [{failure.kind}]"
                f"{f' HTTP {failure.status_code}' if failure.status_code else ''}{retry}"
            )
            lines.append(f"    {_shorten(failure.message, 200)}")
    if audit.all_providers_failed:
        lines.append(
            "NOTE: every external provider failed - this is a provider problem, not "
            "'no results'. Local retrieval is unaffected."
        )
    if verbose:
        for item in audit.queries:
            lines.append(
                f"  {item.provider:<16} {item.operation:<7} ok={item.ok} records={item.records} "
                f"cache_hit={item.cache_hit} http={item.http_requests} {item.seconds:.3f}s"
            )
    return "\n".join(lines)


def _group(records: Iterable[ExternalWork]) -> tuple[list[ExternalWork], list[ExternalWork], list[ExternalWork], list[ExternalWork]]:
    """Split records into already-have / verified-new / discovery-only / conflicts."""
    already: list[ExternalWork] = []
    verified: list[ExternalWork] = []
    discovery: list[ExternalWork] = []
    conflicts: list[ExternalWork] = []
    for record in records:
        if record.conflicts or record.discovery_status is DiscoveryStatus.CONFLICT:
            conflicts.append(record)
            continue
        if record.local_match.already_in_corpus:
            already.append(record)
            continue
        if record.discovery_status in (
            DiscoveryStatus.EXTERNAL_METADATA_VERIFIED,
            DiscoveryStatus.EXTERNAL_FULLTEXT_VERIFIED,
        ):
            verified.append(record)
        else:
            discovery.append(record)
    return already, verified, discovery, conflicts


def format_result(
    result: DiscoveryResult,
    *,
    show_abstract: bool = False,
    new_only: bool = False,
    width: int = 100,
    show_audit: bool = True,
) -> str:
    """Render a whole discovery result, grouped by what it means."""
    records = list(result.records)
    if new_only:
        records = [record for record in records if not record.local_match.already_in_corpus]
    already, verified, discovery, conflicts = _group(records)
    lines: list[str] = []
    if show_audit:
        lines.append(format_audit(result.audit))
        lines.append("")

    def section(title: str, items: Sequence[ExternalWork], note: str = "") -> None:
        lines.append(f"{title} ({len(items)})")
        if note:
            lines.append(f"  {note}")
        if not items:
            lines.append("  (none)")
        for index, record in enumerate(items, start=1):
            lines.append(
                format_record(record, index=index, show_abstract=show_abstract, width=width)
            )
        lines.append("")

    section(
        "ALREADY IN LOCAL CORPUS",
        already,
        "these are in your corpus; the external metadata is shown for reference only",
    )
    section("NEW / VERIFIED METADATA", verified)
    section("DISCOVERY ONLY (identity not verified)", discovery)
    section("CONFLICTS (providers disagree)", conflicts)
    return "\n".join(lines).rstrip()


def format_combined(
    combined: Any, *, show_abstract: bool = False, width: int = 100
) -> str:
    """Render local results and external discoveries as separate sections."""
    local = combined.local
    lines = [
        f"query: {combined.query}",
        "",
        f"LOCAL CORPUS (RAG-009, mode={local.mode}, {len(local.chunks)} chunks, "
        f"{local.seconds:.3f}s) - trusted evidence",
    ]
    if local.error:
        lines.append(f"  local retrieval error: {local.error}")
    elif not local.chunks:
        lines.append("  (no local results)")
    for chunk in local.chunks:
        title = _shorten(str(chunk.get("title") or chunk.get("filename") or "(untitled)"), width)
        lines.append(
            f"  {chunk.get('rank')}. {title} | {chunk.get('first_author') or '?'} "
            f"({chunk.get('year') or '?'}) p.{chunk.get('pages')} | "
            f"role={chunk.get('corpus_role')} evidence={chunk.get('evidence_eligible')}"
        )
        lines.append(f"     {_shorten(str(chunk.get('text_preview') or ''), width)}")
    lines.append("")
    lines.append(
        "NEW EXTERNAL DISCOVERY (RAG-010) - not trusted evidence, not imported"
    )
    if not combined.external_available:
        lines.append("  external discovery unavailable (all providers failed)")
        for failure in combined.external.audit.failures:
            lines.append(f"    {failure.provider}: {failure.kind} {failure.status_code or ''}")
        lines.append("")
        return "\n".join(lines).rstrip()
    lines.append(format_audit(combined.external.audit, verbose=False))
    lines.append("")
    _, verified, discovery, conflicts = _group(list(combined.external.records))
    for title, items in (
        ("NEW / VERIFIED METADATA", verified),
        ("DISCOVERY ONLY (identity not verified)", discovery),
        ("CONFLICTS (providers disagree)", conflicts),
        (
            "ALREADY IN LOCAL CORPUS",
            [
                record
                for record in combined.external.records
                if record.local_match.already_in_corpus and not record.conflicts
            ],
        ),
    ):
        lines.append(f"{title} ({len(items)})")
        if not items:
            lines.append("  (none)")
        for index, record in enumerate(items, start=1):
            lines.append(
                format_record(record, index=index, show_abstract=show_abstract, width=width)
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def status_counts(records: Sequence[ExternalWork]) -> dict[str, int]:
    """Count records by status and local-match kind (for reports)."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.discovery_status.value] = counts.get(record.discovery_status.value, 0) + 1
        key = f"local:{record.local_match.kind.value}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def local_kind_counts(records: Sequence[ExternalWork]) -> dict[LocalMatchKind, int]:
    """Count records by local-match kind."""
    counts: dict[LocalMatchKind, int] = {}
    for record in records:
        counts[record.local_match.kind] = counts.get(record.local_match.kind, 0) + 1
    return counts
