"""Development CLI for building the RAG-008 vector index.

Usage::

    python -m app.indexing.build data [--collection phd_rag_chunks_v1]
        [--qdrant-url http://localhost:6333 | --qdrant-path .qdrant_data]
        [--embedding-batch-size 8] [--upload-batch-size 128]
        [--device cpu] [--limit N] [--pilot N] [--dry-run] [--delete-stale]
        [--measure-tokens] [--report-out docs/rag008-indexing.txt]

``--dry-run`` parses and chunks the corpus, reports the intended indexing
population and validates the configuration, but computes no embeddings and never
touches Qdrant. ``--pilot N`` indexes about N chunks (one eligible chunk per
document, plus whatever is needed to cover literature, OCR, ``my_work`` and
``future_chapters``) instead of the whole population.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.chunking.models import ChunkingOptions
from app.config import get_settings
from app.embedding.bge_m3 import BgeM3Embedder
from app.embedding.cache import unit_norm
from app.embedding.models import DEFAULT_MODEL_NAME, EmbeddingModelLoadError
from app.indexing.collection import (
    CollectionDimensionMismatch,
    IndexingError,
    QdrantConfigurationError,
    build_client,
    qdrant_target,
)
from app.indexing.indexer import dry_run, run_index
from app.indexing.manifest import manifest_path
from app.indexing.models import DryRunReport, IndexReport, IndexingOptions

#: Exit codes used by this CLI (documented so scripts can react).
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_TOKEN_LIMIT = 4
EXIT_QDRANT = 5


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.indexing.build",
        description="Embed RAG-007 chunks with BGE-M3 and index them into Qdrant.",
    )
    parser.add_argument("root", nargs="?", default="data", type=Path, help="Corpus root.")
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--qdrant-url", default=settings.qdrant_url)
    parser.add_argument("--qdrant-api-key", default=settings.qdrant_api_key)
    parser.add_argument("--qdrant-path", default=settings.qdrant_path)
    parser.add_argument(
        "--embedding-batch-size", type=int, default=settings.embedding_batch_size
    )
    parser.add_argument(
        "--upload-batch-size", type=int, default=settings.qdrant_upload_batch_size
    )
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=settings.embedding_local_files_only,
        help="Never contact the Hugging Face hub (use the cached snapshot only).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only visit the first N documents.")
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Index about N chunks (validation sample) instead of the whole population.",
    )
    parser.add_argument("--pilot-max-documents", type=int, default=150)
    parser.add_argument("--cache-path", default=settings.embedding_cache_path)
    parser.add_argument("--no-cache", action="store_true", help="Disable the embedding cache.")
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Where to write the index manifest (default: <manifest dir>/<collection>.json).",
    )
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the intended population; no embeddings, no Qdrant writes.",
    )
    parser.add_argument(
        "--delete-stale",
        action="store_true",
        help="Delete points that are no longer in the chunk set (default: report only).",
    )
    parser.add_argument(
        "--measure-tokens",
        action="store_true",
        help="Measure retrieval texts with the real BGE-M3 tokenizer while streaming.",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable the OCR fallback.")
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--show-vector",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N dimensions of one sample vector (0 = none).",
    )
    return parser


def _options(args: argparse.Namespace, *, manifest: str | None) -> IndexingOptions:
    return IndexingOptions(
        collection=args.collection,
        embedding_batch_size=args.embedding_batch_size,
        upload_batch_size=args.upload_batch_size,
        limit=args.limit,
        pilot=args.pilot,
        pilot_max_documents=args.pilot_max_documents,
        ocr=False if args.no_ocr else None,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
        delete_stale=args.delete_stale,
        manifest_path=manifest,
        measure_tokens=args.measure_tokens,
    )


def _validate(args: argparse.Namespace) -> None:
    if args.embedding_batch_size < 1:
        raise QdrantConfigurationError("--embedding-batch-size must be at least 1")
    if args.upload_batch_size < 1:
        raise QdrantConfigurationError("--upload-batch-size must be at least 1")
    if args.pilot < 0:
        raise QdrantConfigurationError("--pilot cannot be negative")
    if not args.qdrant_path and not args.qdrant_url:
        raise QdrantConfigurationError(
            "configure --qdrant-url (server) or --qdrant-path (local persistent index)"
        )


def render_dry_run(report: DryRunReport) -> str:
    """Render a dry-run report as text."""
    population = report.population
    lines = [
        "RAG-008 indexing dry run",
        "========================",
        "",
        f"corpus root:                {report.root}",
        f"collection:                 {report.collection}",
        f"qdrant target:              {report.qdrant_target}",
        f"embedding model:            {report.embedding_model}",
        f"embedding batch size:       {report.embedding_batch_size}",
        f"upload batch size:          {report.upload_batch_size}",
        f"embedding cache:            {report.cache_path or 'disabled'}",
        f"index manifest:             {report.manifest_path or 'disabled'}",
        "",
        "Intended indexing population",
        "----------------------------",
        f"unique documents:           {population.unique_documents}",
        f"documents with chunks:      {population.documents_chunked}",
        f"total RAG-007 chunks:       {population.total_chunks}",
        f"retrieval-eligible chunks:  {population.retrieval_eligible_chunks}",
        f"excluded (ineligible):      {population.excluded_ineligible_chunks}",
        f"  of which references:      {population.excluded_reference_chunks}",
    ]
    if population.by_corpus_role:
        lines.append("")
        lines.append("indexed chunks by corpus role:")
        width = max(len(name) for name in population.by_corpus_role)
        for name, count in population.by_corpus_role.items():
            lines.append(f"  {name:<{width}}  {count}")
    if report.pilot_chunks:
        lines.extend(
            [
                "",
                "Pilot sample (not yet indexed)",
                "------------------------------",
                f"pilot chunks:               {report.pilot_chunks}",
                f"documents visited:          {report.pilot_documents_visited}",
                f"kinds covered:              {', '.join(report.pilot_kinds) or 'none'}",
                f"kinds missing:              {', '.join(report.pilot_missing_kinds) or 'none'}",
            ]
        )
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"note: {note}")
    return "\n".join(lines) + "\n"


def render_report(report: IndexReport) -> str:
    """Render a completed indexing run as text."""
    population = report.population
    lines = [
        "RAG-008 indexing report",
        "=======================",
        "",
        f"corpus root:                {report.root}",
        f"collection:                 {report.collection}",
        f"qdrant target:              {report.qdrant_target}",
        f"collection created:         {report.collection_created}",
        "",
        "Embedding model",
        "---------------",
        f"model:                      {report.embedding_model}",
        f"revision:                   {report.embedding_revision or 'unknown'}",
        f"identifier:                 {report.embedding_identifier}",
        f"library:                    {report.embedding_library} {report.embedding_library_version or ''}".rstrip(),
        f"device:                     {report.device}",
        f"vector dimension:           {report.vector_dimension}",
        f"distance:                   {report.distance}",
        f"normalization:              {report.normalization}",
        f"max input tokens:           {report.max_tokens or 'unknown'}",
        "",
        "Corpus and population",
        "---------------------",
        f"documents embedded:         {report.unique_documents}",
        f"documents visited:          {report.documents_visited}",
    ]
    if population is not None:
        lines.extend(
            [
                f"total RAG-007 chunks seen:  {population.total_chunks}",
                f"retrieval-eligible chunks:  {population.retrieval_eligible_chunks}",
                f"excluded (ineligible):      {population.excluded_ineligible_chunks}",
                f"  of which references:      {population.excluded_reference_chunks}",
            ]
        )
    lines.extend(
        [
            f"chunks embedded:            {report.chunks_embedded}",
            f"points indexed:             {report.points_indexed}",
            f"skipped chunks:             {report.skipped_chunks}",
            "",
            "Timing",
            "------",
            f"embedding seconds:          {report.embedding_seconds}",
            f"embeddings/second:          {report.embeddings_per_second}",
            f"chunks/second:              {report.chunks_per_second}",
            f"real tokens/second:         {report.tokens_per_second or 'n/a'}",
            f"upload seconds:             {report.upload_seconds}",
            f"total seconds:              {report.total_seconds}",
            "",
            "GPU / device",
            "------------",
            f"device:                     {report.device}",
            f"peak allocated VRAM:        {report.peak_allocated_gib or 'n/a'} GiB",
            f"peak reserved VRAM:         {report.peak_reserved_gib or 'n/a'} GiB",
            f"embedding batch size:       {report.embedding_batch_size or 'n/a'}",
            "",
            "Embedding cache",
            "---------------",
            f"cache hits:                 {report.cache_hits}",
            f"cache misses:               {report.cache_misses}",
            f"cache entries:              {report.cache_entries}",
            "",
            "Qdrant",
            "------",
            f"payload indexes:            {', '.join(report.payload_indexes) or 'none'}",
            f"points before:              {report.points_before}",
            f"points after:               {report.points_after}",
            f"inserted:                   {report.inserted}",
            f"updated/upserted:           {report.updated}",
            f"duplicate point ids:        {report.duplicate_point_ids}",
            f"stale points:               {report.stale_points}",
            f"stale points deleted:       {report.deleted_stale_points}",
        ]
    )
    if report.stale_point_ids:
        lines.append("  stale point ids (first 50):")
        for point_id in report.stale_point_ids:
            lines.append(f"    {point_id}")
    lines.extend(
        [
            "",
            "Corpus identity",
            "---------------",
            f"chunking version:           {report.chunking_version}",
            f"corpus fingerprint:         {report.corpus_fingerprint}",
            f"document hashes digest:     {report.document_hashes_digest}",
            f"chunk ids digest:           {report.chunk_ids_digest}",
        ]
    )
    if report.token_stats is not None:
        stats = report.token_stats
        lines.extend(
            [
                "",
                "BGE-M3 tokenizer measurement (measurement only)",
                "-----------------------------------------------",
                f"chunks measured:            {stats.measured_chunks}",
                f"max BGE-M3 tokens:          {stats.token_max}",
                f"median:                     {stats.token_median}",
                f"p95:                        {stats.token_p95}",
                f"model max input tokens:     {stats.model_limit or 'unknown'}",
                f"chunks above 1200:          {stats.chunks_above_1200}",
                f"chunks above model limit:   {stats.chunks_above_model_limit}",
                f"largest estimator count:    {stats.estimator_token_max}",
                f"measured real tokens:       {stats.token_total}",
            ]
        )
    if report.manifest_path:
        lines.extend(["", f"index manifest:             {report.manifest_path}"])
    if report.embedding_failures or report.upload_failures:
        lines.append("")
        lines.append("Failures (never silently dropped)")
        lines.append("---------------------------------")
        for failure in report.embedding_failures:
            lines.append(f"  embedding {failure.chunk_id}: {failure.error}")
        for failure in report.upload_failures:
            lines.append(f"  upload {failure.chunk_id}: {failure.error}")
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"note: {note}")
    return "\n".join(lines) + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    embedder_factory=None,
    client_factory=None,
) -> int:
    """Run the indexing CLI (factories exist so tests can inject doubles)."""
    args = _build_parser().parse_args(argv)
    root: Path = args.root
    if not root.exists():
        print(f"Error: corpus root not found: {root}", file=sys.stderr)
        return EXIT_USAGE

    settings = get_settings()
    manifest = None
    if not args.no_manifest:
        manifest = args.manifest_path or str(
            manifest_path(settings.index_manifest_dir, args.collection)
        )
    options = _options(args, manifest=manifest)

    try:
        _validate(args)
    except QdrantConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    chunking = ChunkingOptions()

    if args.dry_run:
        if not args.measure_tokens:
            report = dry_run(
                root,
                options=options,
                embedding_model=args.model,
                qdrant_target=qdrant_target(args.qdrant_url, args.qdrant_path),
                chunking=chunking,
            )
            text = render_dry_run(report)
            sys.stdout.write(text)
            if args.report_out:
                args.report_out.parent.mkdir(parents=True, exist_ok=True)
                args.report_out.write_text(text, encoding="utf-8")
            return EXIT_OK

    try:
        embedder = (embedder_factory or _default_embedder)(args)
    except EmbeddingModelLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_MODEL

    if args.dry_run and args.measure_tokens:
        # Measurement dry run: the tokenizer is needed, so the model is loaded, but
        # no embedding is computed and Qdrant is never contacted.
        report = run_index(
            root,
            options=options.model_copy(
                update={"pilot": 0, "use_cache": False, "compute_embeddings": False}
            ),
            embedder=embedder,
            client=None,
            chunking=chunking,
            qdrant_target="none (dry run)",
        )
        text = render_report(report)
        sys.stdout.write(text)
        if args.report_out:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(text, encoding="utf-8")
        stats = report.token_stats
        if stats is not None and stats.chunks_above_model_limit:
            print(
                f"Error: {stats.chunks_above_model_limit} chunks exceed the model input "
                f"limit of {stats.model_limit} tokens; stopping before indexing",
                file=sys.stderr,
            )
            return EXIT_TOKEN_LIMIT
        return EXIT_OK

    try:
        client = (client_factory or _default_client)(args)
    except QdrantConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_QDRANT

    try:
        report = run_index(
            root,
            options=options,
            embedder=embedder,
            client=client,
            chunking=chunking,
            qdrant_target=qdrant_target(args.qdrant_url, args.qdrant_path),
            progress=lambda message: print(f"  {message}", file=sys.stderr),
        )
    except CollectionDimensionMismatch as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_QDRANT
    except IndexingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_QDRANT
    finally:
        # A local Qdrant store flushes on close; without this a later process
        # (or the inspection CLI) would not see the points we just wrote.
        close = getattr(client, "close", None)
        if callable(close):
            close()

    text = render_report(report)
    sys.stdout.write(text)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")

    if args.show_vector:
        sample = embedder.embed_query("RAG-008 sample vector")
        first = ", ".join(f"{value:.6f}" for value in sample[: args.show_vector])
        print(f"\nsample vector (first {args.show_vector} dims): {first}")
        print(f"sample vector L2 norm: {unit_norm(sample):.6f}")
    return EXIT_OK


def _default_embedder(args: argparse.Namespace) -> BgeM3Embedder:
    return BgeM3Embedder(
        model_name=args.model or DEFAULT_MODEL_NAME,
        revision=args.revision,
        device=args.device,
        batch_size=args.embedding_batch_size,
        local_files_only=args.local_files_only,
    )


def _default_client(args: argparse.Namespace):
    return build_client(
        url=args.qdrant_url, api_key=args.qdrant_api_key, path=args.qdrant_path
    )


if __name__ == "__main__":
    raise SystemExit(main())
