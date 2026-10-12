"""Citation validation CLI (RAG-012 §16).

    python -m app.citations.inspect "What is AI governance in higher education?"

The command runs RAG-011 generation for the question and then validates and cites the
result, printing three sections:

```
VALIDATED ANSWER   the answer with [Sx] replaced by Harvard in-text citations
REFERENCES         one deduplicated Harvard reference per cited work
VALIDATION         supported / partial / unsupported / uncertain, with counts
```

``--show-claims`` prints the claim-by-claim audit (verdict, method, confidence, numbers,
quotations), ``--show-source-map`` prints every ``S<n>`` with its chunk provenance and its
citation, ``--show-warnings`` prints the warnings only, and ``--json`` emits the whole
structured result. ``--show-original`` prints the untouched RAG-011 answer next to the
validated one.

Exit codes: ``0`` validated, ``1`` the model could not answer, ``3`` at least one claim
was withheld as unsupported.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from app.citations.models import ClaimKind, CitationValidationResult, SupportVerdict
from app.citations.service import CitationService
from app.config import get_settings
from app.generation.models import GenerationMode, OutputLength
from app.retrieval.models import RetrievalMode

#: One-line explanation per verdict, so a reader never has to guess.
_VERDICT_NOTES = {
    SupportVerdict.SUPPORTED: "the cited source contains the claim",
    SupportVerdict.PARTIALLY_SUPPORTED: "only part of the claim is supported by the evidence",
    SupportVerdict.UNSUPPORTED: "the claim failed verification and was withheld",
    SupportVerdict.UNCERTAIN: "the deterministic checks could not confirm the claim",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.citations.inspect",
        description=(
            "Answer a question with RAG-011, validate every claim against the sources it "
            "cites, and replace the internal [Sx] markers with Harvard citations."
        ),
    )
    parser.add_argument("question", help="the question to answer from the corpus")
    parser.add_argument("--model", default=settings.generation_model, help="Ollama model")
    parser.add_argument("--candidate-k", type=int, default=None, help="RAG-009 first-stage depth")
    parser.add_argument("--final-k", type=int, default=None, help="evidence chunks to consider")
    parser.add_argument("--no-rerank", action="store_true", help="dense-only retrieval")
    parser.add_argument(
        "--length",
        choices=[length.value for length in OutputLength],
        default=OutputLength.NORMAL.value,
        help="answer length control",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in GenerationMode],
        default=None,
        help="context mode (evidence_only by default)",
    )
    parser.add_argument(
        "--include-my-work",
        action="store_true",
        help="add the author's own writing as non-citable continuity context",
    )
    parser.add_argument(
        "--include-guidance",
        action="store_true",
        help="also add supervision/writing guidance as non-citable context",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.EVIDENCE.value,
        help="RAG-009 retrieval mode (evidence by default)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=None,
        help="drop evidence below this rerank score before generation",
    )
    parser.add_argument("--context-tokens", type=int, default=None, help="context budget")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="escalate claims the deterministic checks cannot decide to the local BGE-M3",
    )
    parser.add_argument(
        "--support-coverage",
        type=float,
        default=None,
        help="single-sentence coverage needed for SUPPORTED (default 0.60)",
    )
    parser.add_argument("--show-claims", action="store_true", help="print the claim audit")
    parser.add_argument(
        "--show-source-map", action="store_true", help="print every source and its citation"
    )
    parser.add_argument("--show-warnings", action="store_true", help="print warnings only")
    parser.add_argument("--show-original", action="store_true", help="also print the RAG-011 answer")
    parser.add_argument("--show-prompt", action="store_true", help="print both generation prompts")
    parser.add_argument("--json", action="store_true", help="emit the structured result")
    parser.add_argument("--width", type=int, default=100)
    return parser.parse_args(argv)


def build_service(args: argparse.Namespace) -> CitationService:
    """Build a citation service over the real retrieval pipeline and model."""
    from app.generation.ollama import OllamaClient
    from app.generation.service import GenerationService
    from app.retrieval.factory import build_retrieval_service
    from app.citations.models import ValidationPolicy

    settings = get_settings()
    retrieval = build_retrieval_service(settings, rerank=not args.no_rerank)
    model = OllamaClient(
        base_url=settings.ollama_base_url,
        model=args.model,
        timeout=settings.generation_timeout,
        keep_alive=settings.generation_keep_alive,
        think=settings.generation_think,
    )
    policy = ValidationPolicy()
    if args.support_coverage is not None:
        policy = policy.model_copy(update={"support_coverage": args.support_coverage})
    return CitationService(
        generation_service=GenerationService(
            retrieval_service=retrieval, model=model, settings=settings
        ),
        policy=policy,
        semantic=args.semantic,
    )


def _clean(text: str, width: int = 100) -> str:
    return " ".join((text or "").split())[:width]


def format_validation(result: CitationValidationResult, *, width: int = 100) -> str:
    """The ``VALIDATION`` section: counts per verdict, then the claims that need attention."""
    summary = result.validation_summary
    lines = [
        "VALIDATION",
        f"  validator        : {result.validator}",
        f"  claims           : {summary.claims_total} "
        f"(sentences checked: {summary.sentences_total})",
        f"  supported        : {summary.supported}",
        f"  partial          : {summary.partially_supported}",
        f"  unsupported      : {summary.unsupported}",
        f"  uncertain        : {summary.uncertain}",
        f"  withheld         : {summary.withheld_claims}",
        f"  annotated        : {summary.annotated_claims}",
        f"  citations written: {summary.markers_replaced} "
        f"({summary.sources_cited} sources, {summary.references_total} references)",
        f"  quotations       : {summary.quotations_checked} checked, "
        f"{summary.quotations_failed} not found",
        f"  numbers          : {summary.numeric_mentions_checked} checked, "
        f"{summary.numeric_mentions_missing} absent from the evidence",
    ]
    if summary.invalid_markers:
        lines.append(f"  invalid markers  : {', '.join(summary.invalid_markers)}")
    if summary.non_evidence_sources_cited:
        lines.append(
            f"  non-evidence     : {', '.join(summary.non_evidence_sources_cited)}"
        )
    if summary.elevated_risk_claims:
        lines.append(f"  elevated risk    : {summary.elevated_risk_claims} claims")
    if summary.unmarked_claims:
        lines.append(f"  unmarked claims  : {summary.unmarked_claims} (RAG-011 reports these)")
    lines.append(f"  policy           : {result.policy_note}")
    problem = [
        claim
        for claim in result.claim_validations
        if claim.kind is ClaimKind.CLAIM and claim.verdict is not SupportVerdict.SUPPORTED
    ]
    for claim in problem:
        lines.append(
            f"    {claim.verdict.value} [{', '.join(claim.source_ids) or '-'}]: "
            f"{_clean(claim.sentence, width)}"
        )
        lines.append(f"      {_VERDICT_NOTES[claim.verdict]} ({claim.validator_method.value})")
    return "\n".join(lines)


def format_claims(result: CitationValidationResult, *, width: int = 120) -> str:
    """The ``CLAIMS`` section: the full claim-by-claim audit."""
    lines = ["CLAIMS"]
    for claim in result.claim_validations:
        lines.append(
            f"  {claim.index + 1:>3}. {claim.kind.value:<18} {claim.verdict.value:<19} "
            f"method={claim.validator_method.value} confidence={claim.confidence} "
            f"sources={','.join(claim.source_ids) or '-'}"
        )
        lines.append(f"       {_clean(claim.sentence, width)}")
        lines.append(f"       reason: {claim.reason}")
        if claim.cited_ids:
            for record in claim.per_source:
                lines.append(
                    f"       - [{record.source_id}] {record.verdict.value} "
                    f"single={record.best_sentence_coverage} union={record.union_coverage}"
                )
        if claim.combination_required:
            lines.append("       combination of sources required: no single source supports it")
        for mention in claim.numeric_mentions:
            state = {True: "found", False: "MISSING", None: "not checked"}[mention.found]
            where = f" in {','.join(mention.matched_source_ids)}" if mention.matched_source_ids else ""
            lines.append(f"       number: {mention.raw} ({mention.kind.value}) {state}{where}")
        for check in claim.quotations:
            state = "found" if check.found_in_source else "NOT FOUND"
            lines.append(f"       quote: {state} ({check.match_level}) {_clean(check.text, 80)}")
        if claim.elevated_validation_risk:
            lines.append(f"       elevated_validation_risk: {'; '.join(claim.risk_reasons)}")
        if claim.marker_replacements:
            for group, replacement in claim.marker_replacements.items():
                lines.append(f"       {group} -> {replacement or '(withheld)'}")
        if claim.decision.value in ("annotated", "withheld"):
            lines.append(f"       decision: {claim.decision.value}")
    return "\n".join(lines)


def format_source_map(result: CitationValidationResult, *, width: int = 90) -> str:
    """The ``SOURCE MAP`` section: every source with its provenance and citation."""
    if not result.source_map:
        return "SOURCE MAP\n  (no sources were supplied)"
    citations = {entry.source_id: entry for entry in result.citation_map}
    lines = ["SOURCE MAP"]
    for source_id in sorted(
        result.source_map, key=lambda item: int(item[1:]) if item[1:].isdigit() else 0
    ):
        block = result.source_map[source_id]
        entry = citations.get(source_id)
        lines.append(
            f"  [{source_id}] {_clean(str(block.get('title') or '(untitled)'), width)}"
        )
        lines.append(
            f"       author={block.get('first_author') or 'unknown'} "
            f"year={block.get('year') or 'unknown'} "
            f"pages={block.get('pages') or 'none'} doi={block.get('doi') or 'none'}"
        )
        lines.append(
            f"       class={block.get('source_class')} citable={block.get('citable')} "
            f"role={block.get('corpus_role')} evidence={block.get('evidence_eligible')} "
            f"ocr={block.get('ocr_involved')} content_type={block.get('content_type')}"
        )
        lines.append(
            f"       chunk={block.get('chunk_id')} doc={block.get('document_id')}"
        )
        if entry is not None:
            lines.append(
                f"       citation={entry.in_text} reference={entry.reference_key}"
                + (f" fallbacks={','.join(entry.fallbacks)}" if entry.fallbacks else "")
            )
        else:
            lines.append("       citation=(none: this source received no citation)")
    return "\n".join(lines)


def format_references(result: CitationValidationResult) -> str:
    """The ``REFERENCES`` section."""
    if not result.reference_list:
        return "REFERENCES\n  (no citable sources were used)"
    lines = ["REFERENCES"]
    for index, reference in enumerate(result.reference_list, start=1):
        lines.append(f"  {index}. {reference}")
    return "\n".join(lines)


def format_warnings(result: CitationValidationResult) -> str:
    """The ``WARNINGS`` section, from RAG-012 and preserved from RAG-011."""
    lines: list[str] = []
    if result.warnings or result.generation_warnings or result.generation_errors:
        lines.append("WARNINGS")
        for error in result.generation_errors:
            lines.append(f"  rag011 error: {error}")
        for warning in result.generation_warnings:
            lines.append(f"  rag011: {warning}")
        for warning in result.warnings:
            lines.append(f"  {warning}")
    return "\n".join(lines)


def format_response(result: CitationValidationResult, args: argparse.Namespace) -> str:
    """Render the whole CLI report."""
    sections: list[str] = []
    if args.show_warnings:
        warnings = format_warnings(result)
        sections.append(warnings or "WARNINGS\n  (none)")
        sections.append(format_validation(result, width=args.width))
        return "\n\n".join(sections)
    sections.append("VALIDATED ANSWER")
    sections.append(result.validated_answer.strip() or "(no answer)")
    if args.show_original and result.original_answer.strip():
        sections.append("ORIGINAL RAG-011 ANSWER\n" + result.original_answer.strip())
    sections.append(format_references(result))
    sections.append(format_validation(result, width=args.width))
    if args.show_claims:
        sections.append(format_claims(result, width=max(args.width, 120)))
    if args.show_source_map:
        sections.append(format_source_map(result))
    if args.show_prompt:
        sections.append("GENERATION STATUS\n  " + result.generation_status)
    warnings = format_warnings(result)
    if warnings:
        sections.append(warnings)
    return "\n\n".join(sections)


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "retrieval_mode": args.retrieval_mode,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "rerank": not args.no_rerank,
        "include_my_work": args.include_my_work,
        "include_guidance": args.include_guidance,
        "output_length": args.length,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "min_generation_rerank_score": args.min_rerank_score,
        "context_budget_tokens": args.context_tokens,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    service = build_service(args)
    try:
        response, result = service.answer_and_validate(
            args.question, **_generation_kwargs(args)
        )
    finally:
        service.close()

    if args.json:
        payload = result.model_dump(mode="json")
        payload["generation"] = {
            "status": response.status.value,
            "answer": response.answer,
            "model": response.generation_audit.model,
            "prompt_tokens": response.generation_audit.prompt_tokens,
            "generated_tokens": response.generation_audit.generated_tokens,
        }
        print(json.dumps(payload, indent=2))
        return _exit_code(response.status.value, result)

    print(format_response(result, args))
    return _exit_code(response.status.value, result)


def _exit_code(status: str, result: CitationValidationResult) -> int:
    if status == "model_error":
        return 1
    if result.validation_summary.unsupported:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
