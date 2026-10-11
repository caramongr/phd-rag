"""Combined local + external research discovery (RAG-010 §21).

``research_discover`` answers one query in two clearly separated sections:

```
LOCAL RESULTS      (RAG-009 evidence retrieval over the trusted corpus)
EXTERNAL RESULTS   (RAG-010 discovery, never evidence)
```

The sections are never merged into one ranked list and no prose answer is generated.
If every external provider fails, the local section is still returned with a clear
provider-failure status, because Internet access is an enhancement and not a
requirement for the existing RAG (§28).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Sequence

from app.discovery.models import (
    CombinedDiscovery,
    DiscoveryResult,
    LocalDiscoveryResult,
)
from app.discovery.render import format_combined
from app.discovery.service import DiscoveryService
from app.retrieval.models import RetrievalMode


def _pages(candidate: Any) -> str:
    """Page span as a display string (the candidate stores start/end, not ``pages``)."""
    start = getattr(candidate, "page_start", None)
    end = getattr(candidate, "page_end", None)
    if start is None:
        return ""
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def _chunk_summary(candidate: Any) -> dict[str, object]:
    """The fields of a local result that a combined report shows."""
    return {
        "rank": candidate.final_rank,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "file_hash": candidate.file_hash,
        "filename": candidate.filename,
        "title": candidate.title,
        "first_author": candidate.first_author,
        "year": candidate.year,
        "doi": candidate.doi,
        "pages": _pages(candidate),
        "section_title": candidate.section_title,
        "corpus_role": candidate.corpus_role,
        "evidence_eligible": candidate.evidence_eligible,
        "ocr_involved": candidate.ocr_involved,
        "content_type": candidate.content_type,
        "bibliography_like": candidate.bibliography_like,
        "dense_score": candidate.dense_score,
        "rerank_score": candidate.rerank_score,
        "text_preview": " ".join((candidate.text or "").split())[:280],
    }


def _share_qdrant_client(retrieval_service: Any, discovery_service: DiscoveryService) -> None:
    """Let discovery reuse the Qdrant client the retrieval service already holds.

    A local Qdrant store permits a single open client per process, so opening a second
    one for the local-corpus comparison would fail. The client is only read from.
    """
    if retrieval_service is None or discovery_service.has_local_index:
        return
    client = getattr(getattr(retrieval_service, "retriever", None), "client", None)
    if client is None:
        return
    collection = getattr(retrieval_service, "collection", None)
    discovery_service.share_client(client, collection)


def research_discover(
    query: str,
    *,
    local_k: int = 8,
    external_k: int = 20,
    retrieval_service: Any,
    discovery_service: DiscoveryService,
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    candidate_k: int | None = None,
    providers: Sequence[str] | None = None,
    refresh: bool = False,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    sort: str = "relevance",
    new_only: bool = False,
) -> CombinedDiscovery:
    """Return local evidence and external discoveries side by side."""
    local_started = time.perf_counter()
    error: str | None = None
    chunks: tuple[dict[str, object], ...] = ()
    collection = getattr(retrieval_service, "collection", "") if retrieval_service else ""
    try:
        response = retrieval_service.retrieve(
            query,
            mode=mode,
            candidate_k=candidate_k,
            final_k=local_k,
        )
        chunks = tuple(_chunk_summary(candidate) for candidate in response.results)
    except Exception as exc:  # local retrieval must not be lost to an external failure
        error = f"{type(exc).__name__}: {exc}"
    local_seconds = time.perf_counter() - local_started

    local = LocalDiscoveryResult(
        query=query,
        collection=str(collection),
        mode=mode.value,
        candidate_k=candidate_k or getattr(retrieval_service, "candidate_k", 0),
        final_k=local_k,
        chunks=chunks,
        seconds=round(local_seconds, 4),
        error=error,
    )

    _share_qdrant_client(retrieval_service, discovery_service)

    external: DiscoveryResult = discovery_service.discover(
        query,
        year_min=year_min,
        year_max=year_max,
        max_results=external_k,
        providers=providers,
        refresh=refresh,
        author=author,
        sort=sort,
    )
    if new_only:
        external = DiscoveryResult(
            audit=external.audit,
            records=tuple(
                record
                for record in external.records
                if not record.local_match.already_in_corpus
            ),
        )
    return CombinedDiscovery(query=query, local=local, external=external)


# --------------------------------------------------------------------------- CLI


def build_retrieval_service() -> Any:
    """Build the RAG-009 retrieval service from settings (real models, real index)."""
    from app.retrieval.factory import build_retrieval_service as build

    return build()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.discovery.research",
        description=(
            "Local evidence retrieval (RAG-009) plus external academic discovery "
            "(RAG-010), reported as two separate sections. No answer is generated."
        ),
    )
    parser.add_argument("query", help="natural-language research question")
    parser.add_argument("--local-k", type=int, default=8, help="local chunks to show")
    parser.add_argument(
        "--external-k", type=int, default=20, help="external results to consider"
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="RAG-009 candidate depth (local mode only)",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.EVIDENCE.value,
        help="local retrieval mode",
    )
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument(
        "--sort",
        choices=("relevance", "recent"),
        default="relevance",
        help="provider sort order (recent = newest first)",
    )
    parser.add_argument(
        "--provider", action="append", default=None, help="provider to use (repeatable)"
    )
    parser.add_argument("--refresh", action="store_true", help="bypass a valid cache entry")
    parser.add_argument("--no-cache", action="store_true", help="disable the cache")
    parser.add_argument("--new-only", action="store_true", help="hide already-held results")
    parser.add_argument("--show-abstract", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="skip external providers entirely (local retrieval only)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--width", type=int, default=100, help="preview width")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m app.discovery.research``."""
    from app.discovery.search import build_service as build_discovery_service

    args = parse_args(argv)
    discovery_service = build_discovery_service(args)
    retrieval_service = None if args.local_only else build_retrieval_service()
    try:
        combined = research_discover(
            args.query,
            local_k=args.local_k,
            external_k=args.external_k,
            retrieval_service=retrieval_service,
            discovery_service=discovery_service,
            mode=RetrievalMode(args.mode),
            candidate_k=args.candidate_k,
            providers=discovery_service.provider_names or None,
            refresh=args.refresh,
            year_min=args.year_min,
            year_max=args.year_max,
            author=args.author,
            sort=args.sort,
            new_only=args.new_only,
        )
    finally:
        if retrieval_service is not None:
            retrieval_service.close()

    if args.json:
        print(json.dumps(combined.model_dump(mode="json"), indent=2))
        return 0
    print(format_combined(combined, show_abstract=args.show_abstract, width=args.width))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
