"""Streaming index build: chunks -> embeddings -> Qdrant points (RAG-008).

The indexer streams: it never holds the whole corpus of retrieval texts or vectors
in memory, it embeds in ``embedding_batch_size`` batches, uploads in
``upload_batch_size`` batches, and discards each batch as soon as it is written.

Indexing is limited to retrieval-eligible chunks (the RAG-007 §8 population), so
reference lists stay in the chunk corpus but never enter the normal retrieval
collection. Role and evidence policy are *not* used as a filter: they travel in the
payload so RAG-009 can decide.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from app.chunking.corpus import (
    CorpusDocument,
    CorpusScan,
    iter_corpus_documents,
    scan_corpus,
)
from app.chunking.models import ChunkingOptions
from app.chunking.normalize import NORMALIZATION_VERSION
from app.embedding.batch import EmbedRequest, embed_requests, iter_batches
from app.embedding.cache import EmbeddingCache
from app.embedding.models import EmbeddingModel, count_tokens_many
from app.indexing.collection import (
    CollectionSetup,
    count_points,
    delete_points,
    ensure_collection,
    locate_client,
    make_points,
    scroll_point_ids,
    upsert_points,
)
from app.indexing.fingerprint import (
    chunk_ids_digest,
    corpus_fingerprint,
    document_hashes_digest,
)
from app.indexing.manifest import IndexManifest, write_manifest
from app.indexing.models import (
    REQUIRED_PILOT_KINDS,
    CorpusPopulation,
    DryRunReport,
    FailureRow,
    IndexableChunk,
    IndexingOptions,
    IndexReport,
    TokenStats,
    chunk_kinds,
)
from app.indexing.payload import build_payload
from app.metadata.corpus import resolve_corpus_role
from app.metadata.enums import CorpusRole
from app.models.chunk import Chunk

#: Distance metric recorded in the manifest (matches the created collection).
DISTANCE = "Cosine"
#: Normalization described in the manifest and the report.
NORMALIZATION = "l2 (unit-norm, cosine-ready)"
#: Token threshold reported separately in the tokenizer measurement (§27).
TOKEN_REPORT_THRESHOLD = 1200


def _retrieval_text(chunk: Chunk) -> str:
    """Text to embed: the RAG-007 retrieval text, never the source text."""
    return chunk.retrieval_text if chunk.retrieval_text else chunk.text


def _indexable(record: CorpusDocument, chunk: Chunk) -> IndexableChunk:
    return IndexableChunk(
        chunk=chunk,
        file_hash=record.file_hash,
        filename=record.filename,
        retrieval_text=_retrieval_text(chunk),
    )


def _pilot_scan(scan: CorpusScan) -> CorpusScan:
    """Visit non-literature roles first, then filename order.

    The pilot must exercise ``my_work``, ``future_chapters``, rules, guidance,
    summaries and supervision (small folders that would otherwise sit far behind
    hundreds of literature PDFs in filename order). Ordering the *visit* this way
    keeps the sample the same size while guaranteeing role coverage; the effective
    role is still resolved from all physical locations (RAG-007 §25).
    """
    def key(document) -> tuple[int, str]:
        resolution = resolve_corpus_role(
            [location.path for location in document.physical_locations]
            or [document.canonical_path]
        )
        priority = 1 if resolution.effective_role is CorpusRole.LITERATURE else 0
        return (priority, document.filename)

    return scan.model_copy(update={"ordered": sorted(scan.ordered, key=key)})


def iter_population(
    root: str | Path,
    *,
    options: IndexingOptions,
    chunking: ChunkingOptions | None = None,
) -> Iterator[tuple[IndexableChunk, bool]]:
    """Stream ``(chunk, retrieval_eligible)`` in deterministic corpus order.

    Default mode yields **every** chunk with its eligibility, so the run can report
    the real corpus denominators while only embedding the eligible ones.

    Pilot mode (``options.pilot`` > 0) yields only the sampled chunks: the first
    retrieval-eligible chunk of each document - visiting the non-literature roles
    first - until the pilot size is reached, plus (while a required kind is still
    missing) the first eligible chunk of that kind, so the pilot always covers
    literature, OCR, ``my_work`` and ``future_chapters``.
    """
    scan = scan_corpus(root, limit=options.limit)
    if options.pilot > 0:
        scan = _pilot_scan(scan)
    records = iter_corpus_documents(
        root,
        options=chunking or ChunkingOptions(),
        limit=options.limit,
        ocr=options.ocr,
        scan=scan,
    )
    if options.pilot <= 0:
        for record in records:
            for chunk in record.chunks:
                yield _indexable(record, chunk), bool(chunk.retrieval_eligible)
        return

    pending = set(REQUIRED_PILOT_KINDS)
    taken = 0
    documents_seen = 0
    for record in records:
        documents_seen += 1
        first = next((chunk for chunk in record.chunks if chunk.retrieval_eligible), None)
        if first is not None:
            kinds = chunk_kinds(first)
            if taken < options.pilot or kinds & pending:
                pending -= kinds
                taken += 1
                yield _indexable(record, first), True
        if taken >= options.pilot and not pending:
            return
        if documents_seen >= options.pilot_max_documents:
            return


def describe_population(
    root: str | Path,
    *,
    options: IndexingOptions,
    chunking: ChunkingOptions | None = None,
) -> CorpusPopulation:
    """Count the whole corpus and the indexing population (no embeddings)."""
    population = CorpusPopulation(chunking_version=NORMALIZATION_VERSION)
    scan = scan_corpus(root, limit=options.limit)
    population.unique_documents = scan.unique_documents
    roles: Counter[str] = Counter()
    content_types: Counter[str] = Counter()

    for record in iter_corpus_documents(
        root, options=chunking or ChunkingOptions(), limit=options.limit, ocr=options.ocr
    ):
        if record.chunks:
            population.documents_chunked += 1
        if record.chunked is not None and record.chunked.tokenizer_name:
            population.tokenizer_name = record.chunked.tokenizer_name
        for chunk in record.chunks:
            population.total_chunks += 1
            if chunk.retrieval_eligible:
                population.retrieval_eligible_chunks += 1
                roles[chunk.corpus_role.value if chunk.corpus_role else "unknown"] += 1
                content_types[chunk.content_type.value] += 1
            else:
                population.excluded_ineligible_chunks += 1
                if chunk.content_type.value == "references":
                    population.excluded_reference_chunks += 1

    population.by_corpus_role = dict(sorted(roles.items()))
    population.by_content_type = dict(sorted(content_types.items()))
    return population


def describe_pilot(
    root: str | Path,
    *,
    options: IndexingOptions,
    chunking: ChunkingOptions | None = None,
) -> tuple[int, list[str], int]:
    """Describe the pilot selection without embedding: (chunks, kinds, documents)."""
    indexables = list(iter_population(root, options=options, chunking=chunking))
    kinds: set[str] = set()
    documents: set[str] = set()
    for item, _eligible in indexables:
        kinds |= chunk_kinds(item.chunk)
        documents.add(item.filename)
    return len(indexables), sorted(kinds), len(documents)


def dry_run(
    root: str | Path,
    *,
    options: IndexingOptions,
    embedding_model: str,
    qdrant_target: str,
    chunking: ChunkingOptions | None = None,
) -> DryRunReport:
    """Validate configuration and report the intended population (§21)."""
    population = describe_population(root, options=options, chunking=chunking)
    report = DryRunReport(
        root=str(root),
        collection=options.collection,
        qdrant_target=qdrant_target,
        embedding_model=embedding_model,
        embedding_batch_size=options.embedding_batch_size,
        upload_batch_size=options.upload_batch_size,
        population=population,
        cache_path=options.cache_path if options.use_cache else None,
        manifest_path=options.manifest_path,
    )
    if options.pilot > 0:
        count, kinds, documents = describe_pilot(root, options=options, chunking=chunking)
        report.pilot_chunks = count
        report.pilot_kinds = kinds
        report.pilot_documents_visited = documents
        report.pilot_missing_kinds = sorted(set(REQUIRED_PILOT_KINDS) - set(kinds))
    report.notes.append("dry run: no embeddings computed, no Qdrant mutation")
    if population.excluded_ineligible_chunks:
        report.notes.append(
            f"{population.excluded_ineligible_chunks} retrieval-ineligible chunks stay in "
            "the RAG-007 corpus but are excluded from normal indexing"
        )
    return report


def _token_stats(
    token_counts: list[int],
    estimator_counts: list[int],
    *,
    model_limit: int | None,
) -> TokenStats:
    ordered = sorted(token_counts)
    if ordered:
        middle = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    else:
        middle = p95 = 0
    above_limit = (
        sum(1 for count in token_counts if count > model_limit) if model_limit else 0
    )
    return TokenStats(
        measured_chunks=len(token_counts),
        token_total=sum(token_counts),
        token_max=ordered[-1] if ordered else 0,
        token_median=middle,
        token_p95=p95,
        token_min=ordered[0] if ordered else 0,
        chunks_above_1200=sum(1 for count in token_counts if count > TOKEN_REPORT_THRESHOLD),
        chunks_above_model_limit=above_limit,
        model_limit=model_limit,
        estimator_token_max=max(estimator_counts) if estimator_counts else 0,
    )


def _cuda_peaks() -> tuple[float, float]:
    """Peak allocated/reserved GPU memory in GiB (0, 0 when there is no CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        )
    except Exception:  # pragma: no cover - defensive
        return 0.0, 0.0


def _reset_cuda_peaks() -> None:
    """Start a fresh peak-memory window for this run."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - defensive
        return


def run_index(
    root: str | Path,
    *,
    options: IndexingOptions,
    embedder: EmbeddingModel,
    client=None,
    chunking: ChunkingOptions | None = None,
    qdrant_target: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> IndexReport:
    """Index the eligible corpus into Qdrant and return the run report.

    The embedder's spec is read before anything is written, and a collection
    dimension mismatch aborts before a single point is uploaded.
    """
    reporting = progress or (lambda _message: None)
    spec = embedder.spec
    _reset_cuda_peaks()

    setup: CollectionSetup | None = None
    if client is not None:
        setup = ensure_collection(
            client, collection=options.collection, dimension=spec.dimension
        )
        reporting(
            f"collection {options.collection}: "
            f"{'created' if setup.created else 'reused'} "
            f"(dimension {spec.dimension}, distance {DISTANCE}, "
            f"{setup.points_before} points before)"
        )
    target = qdrant_target or (locate_client(client) if client is not None else "n/a")

    existing_ids = scroll_point_ids(client, options.collection) if client is not None else []
    existing = set(existing_ids)

    cache: EmbeddingCache | None = None
    if options.use_cache and options.cache_path:
        cache = EmbeddingCache(options.cache_path, model_identifier=spec.identifier)

    report = IndexReport(
        root=str(root),
        collection=options.collection,
        qdrant_target=target,
        embedding_model=spec.model_name,
        embedding_revision=spec.model_revision,
        embedding_identifier=spec.identifier,
        embedding_library=spec.library,
        embedding_library_version=spec.library_version,
        vector_dimension=spec.dimension,
        distance=DISTANCE,
        normalization=NORMALIZATION if spec.normalized else "none",
        device=spec.device,
        max_tokens=spec.max_tokens,
        embedding_batch_size=options.embedding_batch_size,
        upload_batch_size=options.upload_batch_size,
        collection_created=bool(setup and setup.created),
        payload_indexes=list(setup.payload_indexes) if setup else [],
        points_before=setup.points_before if setup else 0,
        chunking_version=NORMALIZATION_VERSION,
        manifest_path=options.manifest_path,
    )

    population = CorpusPopulation(chunking_version=NORMALIZATION_VERSION)
    roles: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    document_hashes: set[str] = set()
    embedded_documents: set[str] = set()
    chunk_ids: list[str] = []
    token_counts: list[int] = []
    estimator_counts: list[int] = []
    seen_documents: set[str] = set()
    indexed_ids: set[str] = set()

    started = time.perf_counter()
    population_source = iter_population(root, options=options, chunking=chunking)
    for batch in iter_batches(population_source, options.upload_batch_size):
        population.total_chunks += len(batch)
        selected: list[IndexableChunk] = []
        for item, eligible in batch:
            seen_documents.add(item.filename)
            if eligible:
                population.retrieval_eligible_chunks += 1
                selected.append(item)
                # The fingerprint describes the corpus this run targets, so it is
                # independent of whether an individual chunk later failed to embed.
                document_hashes.add(item.file_hash)
                chunk_ids.append(item.chunk_id)
                if item.chunk.corpus_role is not None:
                    roles[item.chunk.corpus_role.value] += 1
                content_types[item.chunk.content_type.value] += 1
            else:
                population.excluded_ineligible_chunks += 1
                if item.chunk.content_type.value == "references":
                    population.excluded_reference_chunks += 1

        if options.measure_tokens:
            for item in selected:
                estimator_counts.append(item.chunk.token_count or 0)
            measured = count_tokens_many(embedder, [item.retrieval_text for item in selected])
            token_counts.extend([count for count in measured if count is not None])
        if not selected or not options.compute_embeddings:
            continue

        requests = [EmbedRequest(key=item.chunk_id, text=item.retrieval_text) for item in selected]
        embed_started = time.perf_counter()
        outcomes = embed_requests(
            embedder, requests, batch_size=options.embedding_batch_size, cache=cache
        )
        report.embedding_seconds += time.perf_counter() - embed_started

        points: list[tuple[str, Sequence[float], dict]] = []
        for item, outcome in zip(selected, outcomes):
            if not outcome.ok:
                report.embedding_failures.append(
                    FailureRow(
                        chunk_id=item.chunk_id,
                        filename=item.filename,
                        stage="embedding",
                        error=outcome.error or "unknown embedding failure",
                    )
                )
                report.skipped_chunks += 1
                continue
            vector = outcome.vector or []
            if len(vector) != spec.dimension:
                report.embedding_failures.append(
                    FailureRow(
                        chunk_id=item.chunk_id,
                        filename=item.filename,
                        stage="embedding",
                        error=(
                            f"vector dimension {len(vector)} does not match the model "
                            f"dimension {spec.dimension}"
                        ),
                    )
                )
                report.skipped_chunks += 1
                continue
            points.append(
                (
                    item.chunk_id,
                    vector,
                    build_payload(item.chunk, file_hash=item.file_hash, filename=item.filename),
                )
            )
            report.chunks_embedded += 1
            indexed_ids.add(item.chunk_id)
            embedded_documents.add(item.filename)

        if points and client is not None:
            upload_started = time.perf_counter()
            try:
                report.points_indexed += upsert_points(
                    client,
                    options.collection,
                    make_points(points),
                    batch_size=options.upload_batch_size,
                )
            except Exception as exc:
                report.upload_failures.append(
                    FailureRow(
                        chunk_id=points[0][0],
                        filename=selected[0].filename if selected else None,
                        stage="upload",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                report.skipped_chunks += len(points)
            finally:
                report.upload_seconds += time.perf_counter() - upload_started

    population.by_corpus_role = dict(sorted(roles.items()))
    population.by_content_type = dict(sorted(content_types.items()))
    report.population = population
    report.total_chunks = population.total_chunks
    report.retrieval_eligible_chunks = population.retrieval_eligible_chunks
    report.excluded_ineligible_chunks = population.excluded_ineligible_chunks
    report.excluded_reference_chunks = population.excluded_reference_chunks
    report.unique_documents = len(embedded_documents) or len(seen_documents)
    report.documents_visited = len(seen_documents)

    duplicates = len(chunk_ids) - len(set(chunk_ids))
    report.duplicate_point_ids = duplicates
    if duplicates:
        report.notes.append(
            f"{duplicates} duplicate chunk ids appeared in the population; the later "
            "point wins on upsert (this should always be zero)"
        )

    report.corpus_fingerprint = corpus_fingerprint(
        file_hashes=document_hashes,
        chunk_ids=chunk_ids,
        chunking_version=NORMALIZATION_VERSION,
    )
    report.document_hashes_digest = document_hashes_digest(document_hashes)
    report.chunk_ids_digest = chunk_ids_digest(chunk_ids)
    report.inserted = len(indexed_ids - existing)
    report.updated = len(indexed_ids & existing)

    if options.measure_tokens:
        report.token_stats = _token_stats(
            token_counts, estimator_counts, model_limit=spec.max_tokens
        )

    if client is not None:
        stale = sorted(existing - indexed_ids)
        report.stale_points = len(stale)
        report.stale_point_ids = stale[:50]
        if options.delete_stale and stale:
            report.deleted_stale_points = delete_points(client, options.collection, stale)
        report.points_after = count_points(client, options.collection)

    elapsed = time.perf_counter() - started
    report.total_seconds = round(elapsed, 3)
    report.embedding_seconds = round(report.embedding_seconds, 3)
    report.upload_seconds = round(report.upload_seconds, 3)
    report.embeddings_per_second = (
        round(report.chunks_embedded / report.embedding_seconds, 2)
        if report.embedding_seconds
        else 0.0
    )
    report.chunks_per_second = round(report.chunks_embedded / elapsed, 2) if elapsed else 0.0
    if (
        report.token_stats is not None
        and report.token_stats.token_total
        and report.embedding_seconds
    ):
        report.tokens_per_second = round(
            report.token_stats.token_total / report.embedding_seconds, 1
        )
    report.peak_allocated_gib, report.peak_reserved_gib = _cuda_peaks()

    if cache is not None:
        cache.save()
        stats = cache.stats
        report.cache_hits = stats["hits"]
        report.cache_misses = stats["misses"]
        report.cache_entries = stats["entries"]

    if options.manifest_path:
        manifest = IndexManifest(
            collection=options.collection,
            qdrant_target=target,
            distance=DISTANCE,
            vector_dimension=spec.dimension,
            embedding_model=spec.model_name,
            embedding_revision=spec.model_revision,
            embedding_identifier=spec.identifier,
            embedding_library=spec.library,
            embedding_library_version=spec.library_version,
            normalization=NORMALIZATION if spec.normalized else "none",
            device=spec.device,
            chunking_version=NORMALIZATION_VERSION,
            chunk_count=report.total_chunks,
            indexed_chunk_count=report.chunks_embedded,
            document_count=report.unique_documents,
            pilot_chunk_limit=options.pilot,
            corpus_fingerprint=report.corpus_fingerprint,
            document_hashes_digest=report.document_hashes_digest,
            chunk_ids_digest=report.chunk_ids_digest,
            points_total=report.points_after,
            stale_points=report.stale_points,
            deleted_stale_points=report.deleted_stale_points,
            duplicate_point_ids=report.duplicate_point_ids,
        )
        written = write_manifest(options.manifest_path, manifest)
        report.manifest_path = str(written)

    return report
