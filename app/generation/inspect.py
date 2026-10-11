"""Generation CLI (RAG-011 §21).

    python -m app.generation.inspect "How is AI governance being addressed in higher education?"

Default output is the answer, the sources it used, and the generation facts (model,
latency, tokens). ``--show-context`` and ``--show-audit`` expose exactly what was sent
and what was checked, and ``--json`` emits the whole structured response.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from app.config import get_settings
from app.generation.models import (
    GenerationMode,
    GenerationResponse,
    GenerationStatus,
    OutputLength,
    SourceBlock,
)
from app.generation.service import GenerationService
from app.retrieval.models import RetrievalMode

#: One-line explanation per status, so a reader never has to guess what happened.
_STATUS_NOTES = {
    GenerationStatus.OK: "answer generated, no grounding problem detected",
    GenerationStatus.GROUNDING_WARNINGS: "answer generated, but see the grounding warnings",
    GenerationStatus.INSUFFICIENT_EVIDENCE: "the model reported that the evidence is insufficient",
    GenerationStatus.NO_CITABLE_CONTEXT: "no citable evidence was retrieved; no answer was generated",
    GenerationStatus.MODEL_ERROR: "the model could not produce an answer",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.generation.inspect",
        description=(
            "Answer a question from the local corpus using RAG-009 retrieval and a local "
            "Qwen model. Sources are cited with [S1] markers; nothing is imported."
        ),
    )
    parser.add_argument("question", help="the question to answer from the corpus")
    parser.add_argument(
        "--model", default=settings.generation_model, help="Ollama generation model"
    )
    parser.add_argument("--candidate-k", type=int, default=None, help="RAG-009 first-stage depth")
    parser.add_argument("--final-k", type=int, default=None, help="evidence chunks to consider")
    parser.add_argument("--no-rerank", action="store_true", help="dense-only retrieval (no reranker)")
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
        "--mode",
        choices=[mode.value for mode in GenerationMode],
        default=None,
        help="context mode (evidence_only by default)",
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
        "--length",
        choices=[length.value for length in OutputLength],
        default=OutputLength.NORMAL.value,
        help="answer length control",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=None,
        help="drop evidence below this rerank score before generation",
    )
    parser.add_argument("--context-tokens", type=int, default=None, help="context budget")
    parser.add_argument("--show-context", action="store_true", help="print the exact context")
    parser.add_argument("--show-prompt", action="store_true", help="print both prompts")
    parser.add_argument("--show-audit", action="store_true", help="print the audit trail")
    parser.add_argument("--json", action="store_true", help="emit the structured response")
    parser.add_argument("--width", type=int, default=100)
    return parser.parse_args(argv)


def build_service(args: argparse.Namespace) -> GenerationService:
    """Build a generation service using the real retrieval pipeline and model."""
    from app.generation.ollama import OllamaClient
    from app.retrieval.factory import build_retrieval_service

    settings = get_settings()
    retrieval = build_retrieval_service(settings, rerank=not args.no_rerank)
    model = OllamaClient(
        base_url=settings.ollama_base_url,
        model=args.model,
        timeout=settings.generation_timeout,
        keep_alive=settings.generation_keep_alive,
        think=settings.generation_think,
    )
    return GenerationService(retrieval_service=retrieval, model=model, settings=settings)


def _clean(text: str, width: int = 100) -> str:
    return " ".join((text or "").split())[:width]


def format_sources(response: GenerationResponse, *, width: int = 100) -> str:
    """The ``SOURCES USED`` section: one entry per marker the answer cited."""
    used = response.used_source_ids
    if not used:
        return "SOURCES USED\n  (the answer cited no sources)"
    lines = ["SOURCES USED"]
    for source_id in used:
        block = response.source_map.get(source_id, {})
        lines.append(f"  {source_id} {_source_line(block, width=width)}")
    unused = response.unused_source_ids
    if unused:
        lines.append(
            f"  (supplied but not cited: {', '.join(unused)})"
        )
    return "\n".join(lines)


def _source_line(block: dict[str, Any], *, width: int = 100) -> str:
    parts = [
        str(block.get("title") or "(untitled)")[:width],
        str(block.get("first_author") or "unknown"),
        str(block.get("year") or "unknown"),
    ]
    if block.get("pages"):
        parts.append(f"p.{block['pages']}")
    if block.get("section"):
        parts.append(f"section: {block['section']}")
    line = " | ".join(parts)
    extras = [
        f"chunk={block.get('chunk_id')}",
        f"doc={block.get('document_id')}",
        f"role={block.get('corpus_role')}",
        f"evidence={block.get('evidence_eligible')}",
    ]
    if block.get("ocr_involved"):
        extras.append("OCR=true")
    if block.get("content_type") == "table":
        extras.append("type=table")
    if block.get("source_class") and block.get("source_class") != "evidence":
        extras.append(f"class={block.get('source_class')}")
    return f"{line} ({', '.join(extras)})"


def format_generation(response: GenerationResponse) -> str:
    """The ``GENERATION`` section: model, timing, tokens and grounding state."""
    audit = response.generation_audit
    lines = [
        "GENERATION",
        f"  provider      : {audit.provider} {audit.ollama_version or ''}".rstrip(),
        f"  model         : {audit.model}",
        f"  endpoint      : {audit.base_url}",
        f"  prompt        : {audit.prompt_version} (sha256 {audit.prompt_sha256[:12]})",
        f"  prompt tokens : {audit.prompt_tokens} (estimated {audit.prompt_tokens_estimated})",
        f"  output tokens : {audit.generated_tokens}",
        f"  tokens/sec    : {audit.tokens_per_second}",
        f"  latency       : retrieval {audit.retrieval_seconds:.2f}s + context "
        f"{audit.context_seconds:.3f}s + generation {audit.wall_seconds:.2f}s "
        f"= {audit.retrieval_seconds + audit.context_seconds + audit.wall_seconds:.2f}s",
        f"  settings      : temperature={audit.temperature} top_p={audit.top_p} "
        f"num_ctx={audit.num_ctx} max_output={audit.max_output_tokens} think={audit.think} "
        f"done={audit.done_reason}",
        f"  context       : {response.context.included_chunks}/"
        f"{response.context.retrieved_chunks} blocks, ~{response.context.context_tokens_estimated}"
        f"/{response.context.budget_tokens} tokens, {response.context.excluded_chunks} excluded",
    ]
    if response.status is not GenerationStatus.OK:
        lines.append(f"  status        : {response.status.value} ({_STATUS_NOTES[response.status]})")
    if response.context.ocr_source_ids:
        lines.append(
            f"  OCR sources    : {', '.join(response.context.ocr_source_ids)}"
        )
    if response.context.table_source_ids:
        lines.append(f"  table sources  : {', '.join(response.context.table_source_ids)}")
    return "\n".join(lines)


def format_warnings(response: GenerationResponse) -> str:
    """Grounding warnings, if any, as their own section."""
    if not response.warnings and not response.errors:
        return ""
    lines = ["WARNINGS"]
    for error in response.errors:
        lines.append(f"  error: {error}")
    for warning in response.warnings:
        lines.append(f"  {warning}")
    return "\n".join(lines)


def format_context(context_blocks: Sequence[SourceBlock]) -> str:
    """Print the labelled source blocks exactly as the model saw them."""
    lines = ["CONTEXT"]
    for block in context_blocks:
        lines.append(f"  [{block.source_id}] {block.source_class.value} "
                     f"citable={block.citable} tokens~{block.token_estimate}")
        lines.append(f"      {_clean(block.title, 90)}")
    return "\n".join(lines)


def format_response(
    response: GenerationResponse, args: argparse.Namespace
) -> str:
    """Render the whole CLI report."""
    sections: list[str] = []
    sections.append("ANSWER")
    sections.append(response.answer.strip() or "(no answer)")
    if response.external_discovery_notes:
        sections.append("EXTERNAL RECORDS")
        sections.extend(f"  {note}" for note in response.external_discovery_notes)
    sections.append(format_sources(response, width=args.width))
    sections.append(format_generation(response))
    if args.show_context:
        sections.append(format_context(response.context.blocks))
        sections.append("CONTEXT TEXT\n" + response.context.rendered)
    if args.show_prompt:
        sections.append("SYSTEM PROMPT\n" + response.system_prompt)
        sections.append("USER PROMPT\n" + response.user_prompt)
    if args.show_audit:
        sections.append("RETRIEVAL AUDIT\n" + json.dumps(response.retrieval_audit, indent=2))
        sections.append("MARKER AUDIT\n" + json.dumps(response.marker_audit.model_dump(), indent=2))
    warnings = format_warnings(response)
    if warnings:
        sections.append(warnings)
    return "\n\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    service = build_service(args)
    try:
        response = service.answer(
            args.question,
            mode=args.mode,
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            rerank=not args.no_rerank,
            include_my_work=args.include_my_work,
            include_guidance=args.include_guidance,
            output_length=args.length,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            min_generation_rerank_score=args.min_rerank_score,
            context_budget_tokens=args.context_tokens,
        )
    finally:
        service.close()

    if args.json:
        print(json.dumps(response.model_dump(mode="json"), indent=2))
        return 0
    print(format_response(response, args))
    return 0 if response.status is not GenerationStatus.MODEL_ERROR else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
