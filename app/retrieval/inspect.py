"""Retrieval inspection CLI (RAG-009).

    python -m app.retrieval.inspect "artificial intelligence governance in higher education"

Readable by default (one block per result, no JSON dump) because the point of this
tool is to *look* at what retrieval returns before RAG-010 is allowed to use it.
``--json`` exists for scripting.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from app.config import get_settings
from app.retrieval.models import Candidate, RetrievalFilters, RetrievalMode, RetrievalResponse
from app.retrieval.service import RetrievalService


def _preview(text: str, width: int = 220) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


def format_result(candidate: Candidate, *, width: int = 220) -> str:
    """Render one result as a readable block."""
    lines = [
        f"  {candidate.final_rank}. [{candidate.chunk_id[:8]}] "
        f"dense={_score(candidate.dense_score)} rerank={_score(candidate.rerank_score)}",
        f"     title   : {candidate.title or candidate.filename or '(untitled)'}",
        f"     author  : {candidate.first_author or '?'} ({candidate.year or '?'})"
        + (f" doi={candidate.doi}" if candidate.doi else ""),
        f"     pages   : p.{candidate.pages}  section: {candidate.section_title or '-'}",
        f"     role    : {candidate.corpus_role or '-'}  evidence={'yes' if candidate.evidence_eligible else 'no'}"
        f"  ocr={'yes' if candidate.ocr_involved else 'no'}  type={candidate.content_type}"
        + (
            f"  biblio={candidate.bibliography_reason}"
            if candidate.bibliography_like
            else ""
        ),
        f"     file    : {candidate.filename}",
        f"     text    : {_preview(candidate.text, width)}",
    ]
    if candidate.source_locators:
        lines.append(f"     locators: {', '.join(candidate.source_locators[:4])}")
    return "\n".join(lines)


def _score(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.4f}"


def format_response(response: RetrievalResponse, *, width: int = 220) -> str:
    """Render a whole response: audit header, results, drop summary."""
    audit = response.audit
    stats = response.stats
    header = [
        f"query      : {audit.normalized_query}",
        f"mode       : {audit.retrieval_mode.value}"
        + (f"  filters: {audit.filters}" if audit.filters else ""),
        f"stage      : candidate_k={audit.candidate_k} final_k={audit.final_k} "
        f"rerank={'on' if audit.rerank_enabled else 'off'}",
        f"models     : {audit.embedding_model}@{_short(audit.embedding_revision)} ({audit.embedding_device})"
        + (
            f" | {audit.rerank_model}@{_short(audit.rerank_revision)} ({audit.rerank_device})"
            if audit.rerank_enabled
            else ""
        ),
        f"collection : {audit.collection}",
        f"counts     : retrieved={stats.candidates_retrieved} "
        f"filtered={stats.candidates_after_filter} duplicates={stats.dropped_as_duplicate} "
        f"bibliography={stats.dropped_as_bibliography} "
        f"diversity={stats.dropped_by_diversity} rounds={stats.candidate_rounds} "
        f"returned={stats.returned}",
        f"timing     : embed={stats.embedding_seconds:.3f}s search={stats.search_seconds:.3f}s "
        f"rerank={stats.rerank_seconds:.3f}s total={stats.total_seconds:.3f}s",
        "",
        f"top {len(response.results)} results:",
    ]
    blocks = [format_result(candidate, width=width) for candidate in response.results]
    if not blocks:
        blocks = ["  (no results)"]
    return "\n".join(header + blocks)


def _short(revision: str | None) -> str:
    return (revision or "unknown")[:12]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.inspect",
        description="Retrieve ranked chunks from the RAG-008 Qdrant index.",
    )
    parser.add_argument("query", help="natural-language query")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=None,
        help="retrieval mode (default: settings, i.e. evidence)",
    )
    parser.add_argument("--candidate-k", type=int, default=None, help="dense candidates (default 25)")
    parser.add_argument("--final-k", type=int, default=None, help="results returned (default 8)")
    parser.add_argument("--no-rerank", action="store_true", help="dense-only ranking")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument(
        "--content-type",
        action="append",
        default=None,
        help="restrict to a content type (repeatable), e.g. body, abstract, table",
    )
    parser.add_argument("--corpus-role", default=None, help="restrict to one corpus role")
    parser.add_argument("--document-id", default=None, help="restrict to one document UUID")
    parser.add_argument("--file-hash", default=None, help="restrict to one document SHA-256")
    parser.add_argument("--doi", default=None, help="restrict to one DOI")
    parser.add_argument("--publication-type", default=None)
    parser.add_argument("--ocr-only", action="store_true", help="only chunks produced with OCR")
    parser.add_argument("--no-dedup", action="store_true", help="keep near-duplicate chunks")
    parser.add_argument(
        "--include-bibliography",
        action="store_true",
        help="do not withhold bibliography-like chunks (citation analysis only)",
    )
    parser.add_argument(
        "--max-per-document",
        type=int,
        default=None,
        help="chunks one document may contribute (default 3)",
    )
    parser.add_argument("--min-dense-score", type=float, default=None)
    parser.add_argument("--min-rerank-score", type=float, default=None)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="widening search rounds when duplicates starve the list (default 3)",
    )
    parser.add_argument("--show-text", action="store_true", help="full source text per result")
    parser.add_argument("--show-retrieval-text", action="store_true", help="full embedded text")
    parser.add_argument("--show-locators", action="store_true", help="all source locators")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--qdrant-path", default=None, help="override the local Qdrant directory")
    parser.add_argument("--collection", default=None, help="override the collection name")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--width", type=int, default=220, help="text preview width")
    return parser.parse_args(argv)


def build_filters(args: argparse.Namespace) -> RetrievalFilters:
    """Turn CLI flags into explicit, testable filters."""
    return RetrievalFilters(
        year_min=args.year_min,
        year_max=args.year_max,
        document_id=args.document_id,
        file_hash=args.file_hash,
        doi=args.doi,
        corpus_role=args.corpus_role,
        ocr_involved=True if args.ocr_only else None,
        publication_type=args.publication_type,
        content_types=tuple(args.content_type or ()),
    )


def build_service(args: argparse.Namespace) -> RetrievalService:
    """Build the real service: RAG-008 embedder plus the RAG-009 reranker."""
    from app.embedding.bge_m3 import BgeM3Embedder
    from app.reranking.bge_reranker import BgeReranker

    settings = get_settings()
    embedder = BgeM3Embedder(
        model_name=settings.embedding_model,
        device=args.device or settings.embedding_device,
        local_files_only=settings.embedding_local_files_only,
    )
    reranker = None
    if not args.no_rerank:
        reranker = BgeReranker(
            model_name=settings.reranker_model,
            device=args.device or settings.reranker_device,
            batch_size=settings.reranker_batch_size,
            local_files_only=settings.embedding_local_files_only,
        )
    return RetrievalService(
        embedder=embedder,
        reranker=reranker,
        collection=args.collection or settings.qdrant_collection,
        qdrant_path=args.qdrant_path or (settings.qdrant_path or None),
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        candidate_k=args.candidate_k or settings.retrieval_candidate_k,
        final_k=args.final_k or settings.retrieval_final_k,
        max_chunks_per_document=(
            args.max_per_document
            if args.max_per_document is not None
            else settings.retrieval_max_chunks_per_document
        ),
        suppress_duplicates=not args.no_dedup,
        duplicate_overlap_threshold=settings.retrieval_duplicate_overlap,
        max_candidate_rounds=args.max_rounds or settings.retrieval_max_rounds,
        exclude_bibliography=(
            False if args.include_bibliography else settings.retrieval_exclude_bibliography
        ),
    )


def render(candidate: Candidate, args: argparse.Namespace) -> str:
    """Render a result honouring the ``--show-*`` flags."""
    block = format_result(candidate, width=args.width)
    if args.show_text:
        block += f"\n     ---- text ----\n{candidate.text}"
    if args.show_retrieval_text:
        block += f"\n     ---- retrieval_text ----\n{candidate.retrieval_text}"
    if args.show_locators:
        block += "\n     ---- locators ----\n" + "\n".join(
            f"       {locator}" for locator in candidate.source_locators
        )
    return block


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    mode = RetrievalMode(args.mode or get_settings().retrieval_mode)
    service = build_service(args)
    try:
        response = service.retrieve(
            args.query,
            mode=mode,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            filters=build_filters(args),
            rerank=not args.no_rerank,
            min_dense_score=args.min_dense_score,
            min_rerank_score=args.min_rerank_score,
        )
    finally:
        service.close()

    if args.json:
        print(json.dumps(response.model_dump(mode="json"), indent=2))
        return 0

    print(format_response(response, width=args.width))
    if args.show_text or args.show_retrieval_text or args.show_locators:
        print()
        for candidate in response.results:
            print(render(candidate, args))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
