"""Mode CLI (RAG-013 §11).

    python -m app.modes.inspect evidence "What is AI governance in higher education?"
    python -m app.modes.inspect writing "Write a paragraph about AI governance."
    python -m app.modes.inspect consistency "How does my current chapter describe AI governance?"
    python -m app.modes.inspect supervisor "What guidance do I have about methodology?"
    python -m app.modes.inspect research "Find recent work on AI governance in universities."

One command per mode, one result shape. The default output shows what the mode produced and
what it rested on: a validated, Harvard-cited answer for EVIDENCE and WRITING, a
deterministic comparison report for CONSISTENCY, labelled supervision excerpts for SUPERVISOR,
and the separated local/external sections for RESEARCH. ``--show-sources`` lists every local
block with the role it was allowed to play, ``--show-audit`` prints the audit trail, and
``--json`` emits the whole structured result.

Exit codes: ``0`` produced, ``1`` the model could not answer, ``3`` at least one claim was
withheld as unsupported.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from app.citations.inspect import format_claims
from app.config import get_settings
from app.generation.models import OutputLength
from app.modes.factory import build_mode_service
from app.modes.models import (
    MODES,
    ModeName,
    ModeResult,
    ModeSource,
    ModeStatus,
    OutputStyle,
)
from app.modes.service import ModeService

#: One-line explanation per status.
_STATUS_NOTES = {
    ModeStatus.OK: "the mode produced its output",
    ModeStatus.INSUFFICIENT_EVIDENCE: "the model reported that the evidence is insufficient",
    ModeStatus.NO_CITABLE_CONTEXT: "nothing citable was retrieved, so no answer was generated",
    ModeStatus.MODEL_ERROR: "the model could not produce an answer",
    ModeStatus.NO_RESULTS: "nothing matched this query",
}

#: Section title per output style.
_SECTION_TITLES = {
    OutputStyle.VALIDATED_ANSWER: "VALIDATED ANSWER",
    OutputStyle.STRUCTURED_COMPARISON: "CONSISTENCY REPORT",
    OutputStyle.LABELLED_EXCERPTS: "SUPERVISION EXCERPTS (not evidence)",
    OutputStyle.SEPARATED_RESULTS: "RESULTS",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.modes.inspect",
        description=(
            "Run one academic working mode over the existing pipeline: EVIDENCE, WRITING, "
            "CONSISTENCY, SUPERVISOR or RESEARCH."
        ),
    )
    parser.add_argument("mode", choices=[mode.value for mode in ModeName], help="working mode")
    parser.add_argument("query", help="the question or instruction for this mode")
    parser.add_argument("--model", default=settings.generation_model, help="Ollama model")
    parser.add_argument("--candidate-k", type=int, default=None, help="RAG-009 first-stage depth")
    parser.add_argument("--final-k", type=int, default=None, help="chunks to consider")
    parser.add_argument("--local-k", type=int, default=None, help="chunks a mode shows")
    parser.add_argument("--no-rerank", action="store_true", help="dense-only retrieval")
    parser.add_argument(
        "--length",
        choices=[length.value for length in OutputLength],
        default=OutputLength.NORMAL.value,
        help="answer length (EVIDENCE, WRITING)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--context-tokens", type=int, default=None, help="context budget")
    parser.add_argument("--min-rerank-score", type=float, default=None)
    parser.add_argument(
        "--external",
        action="store_true",
        help="also run RAG-010 discovery (shown separately, never evidence)",
    )
    parser.add_argument("--external-k", type=int, default=20, help="external records to consider")
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="external provider to use (repeatable); default is all configured",
    )
    parser.add_argument("--refresh", action="store_true", help="ignore the discovery cache")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--show-sources", action="store_true", help="list local sources and roles")
    parser.add_argument("--show-audit", action="store_true", help="print the audit trail")
    parser.add_argument("--json", action="store_true", help="emit the structured result")
    parser.add_argument("--width", type=int, default=100)
    return parser.parse_args(argv)


def build_service(args: argparse.Namespace) -> ModeService:
    """Build a mode service over the real retrieval, generation and discovery services."""
    return build_mode_service(
        get_settings(), model=args.model, rerank=not args.no_rerank
    )


def _clean(text: str, width: int = 100) -> str:
    return " ".join((text or "").split())[:width]


def format_header(result: ModeResult) -> str:
    """The mode, the query, the status and the config the mode promises."""
    lines = [
        f"MODE   : {result.mode.value}",
        f"QUERY  : {result.query}",
        f"STATUS : {result.status.value} ({_STATUS_NOTES[result.status]})",
        f"PURPOSE: {result.config.purpose}",
        (
            "CONFIG : retrieval={} generation={} my_work={} guidance={} supervision={} "
            "external={} validation={} style={}"
        ).format(
            result.config.retrieval_mode.value,
            "yes" if result.config.run_generation else "no",
            result.config.include_my_work,
            result.config.include_guidance,
            result.config.include_supervision,
            result.config.use_external_discovery,
            "yes" if result.config.run_citation_validation else "no",
            result.config.output_style.value,
        ),
    ]
    if result.config.context_policy:
        roles = ", ".join(
            f"{role}={value.value}" for role, value in sorted(result.config.context_policy.items())
        )
        lines.append(f"ROLES  : {roles}")
    return "\n".join(lines)


def format_sources(result: ModeResult, *, width: int = 100) -> str:
    """Every local source with the role the mode allowed it to play."""
    lines = ["SOURCES"]
    if not result.local_sources:
        lines.append("  (no local sources)")
    for section in result.sections:
        if not section.local_indices:
            continue
        lines.append(f"  {section.name}")
        for index in section.local_indices:
            source = result.local_sources[index]
            lines.append(f"    - {_source_line(source, width=width)}")
    return "\n".join(lines)


def _source_line(source: ModeSource, *, width: int = 100) -> str:
    parts = [
        source.chunk_id or "(no chunk id)",
        str(source.title or "(untitled)")[:width],
        str(source.first_author or "unknown"),
        str(source.year or "unknown"),
    ]
    if source.pages:
        parts.append(f"p.{source.pages}")
    extras = [
        f"role={source.corpus_role or '-'}",
        f"evidence={source.evidence_eligible}",
        f"used_as={source.role.value}",
        f"via={source.retrieval_mode}",
    ]
    if source.ocr_involved:
        extras.append("OCR=true")
    if source.content_type and source.content_type != "body":
        extras.append(f"type={source.content_type}")
    return " | ".join(parts) + f" ({', '.join(extras)})"


def format_external(result: ModeResult) -> str:
    """External records, grouped and always labelled as non-evidence."""
    external_sections = [
        section for section in result.sections if section.name != "LOCAL EVIDENCE"
    ]
    if not external_sections:
        return ""
    lines = ["EXTERNAL RECORDS (Internet metadata; never evidence, never cited)"]
    for section in external_sections:
        lines.append(f"  {section.name} ({len(section.external_indices)})")
        if section.note:
            lines.append(f"    {section.note}")
        if not section.external_indices:
            lines.append("    (none)")
        for index in section.external_indices:
            source = result.external_sources[index]
            authors = ", ".join(source.authors[:2]) or "unknown"
            lines.append(
                f"    - {source.title or '(untitled)'} | {authors} | {source.year or 'n.d.'}"
                + (f" | doi:{source.doi}" if source.doi else "")
                + f" | providers={','.join(source.providers) or '-'}"
                + f" | status={source.discovery_status}"
                + (" | already in corpus" if source.already_in_corpus else "")
            )
    return "\n".join(lines)


def format_validation(result: ModeResult) -> str:
    """The RAG-012 validation summary, when citation validation ran."""
    if result.validation_summary is None:
        return ""
    summary = result.validation_summary
    lines = [
        "VALIDATION",
        f"  claims           : {summary.claims_total} "
        f"(sentences checked: {summary.sentences_total})",
        f"  supported        : {summary.supported}",
        f"  partial          : {summary.partially_supported}",
        f"  unsupported      : {summary.unsupported}",
        f"  uncertain        : {summary.uncertain}",
        f"  withheld         : {summary.withheld_claims}",
        f"  citations written: {summary.markers_replaced} "
        f"({summary.sources_cited} sources, {summary.references_total} references)",
        f"  quotations       : {summary.quotations_checked} checked, "
        f"{summary.quotations_failed} not found",
        f"  numbers          : {summary.numeric_mentions_checked} checked, "
        f"{summary.numeric_mentions_missing} absent from the evidence",
    ]
    if result.audits.get("citations", {}).get("policy_note"):
        lines.append(f"  policy           : {result.audits['citations']['policy_note']}")
    return "\n".join(lines)


def format_findings(result: ModeResult, *, width: int = 110) -> str:
    """The CONSISTENCY findings, by kind."""
    if not result.findings:
        return ""
    lines = ["FINDINGS"]
    for finding in result.findings:
        lines.append(f"  [{finding.label}] {_clean(finding.summary, width)}")
        if finding.your_text:
            lines.append(f"      your text  : {_clean(finding.your_text, width)}")
        if finding.your_chunk_id:
            lines.append(f"      your chunk : {finding.your_chunk_id}")
        if finding.other_text:
            lines.append(f"      literature : {_clean(finding.other_text, width)}")
        if finding.other_chunk_id:
            lines.append(f"      lit. chunk : {finding.other_chunk_id}")
    return "\n".join(lines)


def format_warnings(result: ModeResult, *, limit: int = 25) -> str:
    """Warnings, if any."""
    if not result.warnings:
        return ""
    lines = ["WARNINGS"]
    for warning in result.warnings[:limit]:
        lines.append(f"  - {warning}")
    if len(result.warnings) > limit:
        lines.append(f"  ... {len(result.warnings) - limit} more")
    return "\n".join(lines)


def format_response(result: ModeResult, args: argparse.Namespace) -> str:
    """Render the whole CLI report for one mode run."""
    sections: list[str] = [format_header(result)]
    body = result.validated_answer or result.answer
    sections.append(
        _SECTION_TITLES[result.config.output_style] + "\n" + (body or "(no output)")
    )
    if result.references:
        sections.append(
            "REFERENCES\n"
            + "\n".join(
                f"  {index}. {reference}"
                for index, reference in enumerate(result.references, start=1)
            )
        )
    validation = format_validation(result)
    if validation:
        sections.append(validation)
    findings = format_findings(result)
    if findings:
        sections.append(findings)
    external = format_external(result)
    if external:
        sections.append(external)
    if args.show_sources or result.config.output_style is OutputStyle.SEPARATED_RESULTS:
        # RESEARCH has no prose body, so its local results are part of the report.
        sections.append(format_sources(result, width=args.width))
    if args.show_audit:
        sections.append("AUDIT\n" + json.dumps(result.audits, indent=2, default=str))
        if result.claim_validations:
            sections.append(format_claims(result))
        if result.generation_status:
            sections.append(f"GENERATION STATUS\n  {result.generation_status}")
    warnings = format_warnings(result)
    if warnings:
        sections.append(warnings)
    sections.append(
        "POLICY\n  only evidence-eligible literature supports scholarly claims; your own work, "
        "supervision and rules are never cited; external records are never evidence"
    )
    return "\n\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    service = build_service(args)
    try:
        result = service.run_mode(
            args.mode,
            args.query,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            local_k=args.local_k if args.local_k is not None else 6,
            rerank=not args.no_rerank,
            output_length=args.length,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            min_generation_rerank_score=args.min_rerank_score,
            context_budget_tokens=args.context_tokens,
            external_k=args.external_k,
            providers=tuple(args.provider) if args.provider else None,
            refresh=args.refresh,
            year_min=args.year_min,
            year_max=args.year_max,
            external=True if args.external else None,
        )
    finally:
        service.close()

    if args.json:
        print(json.dumps(result.to_json(), indent=2, default=str))
        return _exit_code(result)
    print(format_response(result, args))
    return _exit_code(result)


def _exit_code(result: ModeResult) -> int:
    if result.status is ModeStatus.MODEL_ERROR:
        return 1
    if result.validation_summary is not None and result.validation_summary.unsupported:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
