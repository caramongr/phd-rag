"""Retrieval service: dense candidates, evidence filter, rerank, diversity (RAG-009).

The pipeline, in order:

1. normalize the query (conservatively);
2. embed it with the RAG-008 BGE-M3 wrapper;
3. ask Qdrant for ``candidate_k`` neighbours **under the mode/filter condition**;
4. re-check every returned payload locally (defense in depth, catches REFERENCES
   and any mode violation Qdrant did not honour);
5. optionally rerank with the local BGE cross-encoder;
6. apply optional minimum scores (off by default - no invented thresholds);
7. suppress near-duplicates, then cap chunks per document;
8. return ``final_k`` results with both scores, full provenance and an audit record.

Scores are never fused: ordering uses the rerank score when reranking is enabled,
otherwise the dense score, and both stay visible in the output.
"""

from __future__ import annotations

import datetime as _datetime
import time
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.embedding.models import EmbeddingModel
from app.reranking.models import Reranker, RerankInput
from app.retrieval.bibliography import BibliographyVerdict, classify, is_bibliography_like, measure
from app.retrieval.dense import DenseRetriever, candidate_from_point
from app.retrieval.diversity import (
    REASON_DOCUMENT_LIMIT,
    REASON_DUPLICATE_NEIGHBOUR,
    REASON_DUPLICATE_TEXT,
    apply_document_limit,
    suppress_near_duplicates,
)
from app.retrieval.diversity import DEFAULT_MAX_INDEX_DISTANCE, DEFAULT_OVERLAP_THRESHOLD
from app.retrieval.filters import (
    REASON_BIBLIOGRAPHY,
    REASON_INELIGIBLE,
    REASON_MODE,
    REASON_REFERENCES,
    mode_excludes_bibliography,
    rejection_reason,
)
from app.retrieval.models import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_CANDIDATE_ROUNDS,
    Candidate,
    DroppedCandidate,
    QueryAudit,
    RetrievalFilters,
    RetrievalMode,
    RetrievalResponse,
    RetrievalStats,
)
from app.retrieval.query import normalize_query

#: Defaults from the task brief; both are configurable per call and per environment.
DEFAULT_CANDIDATE_K = 25
DEFAULT_FINAL_K = 8
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 3


def _filter_payload(candidate: Candidate) -> dict[str, Any]:
    """The payload subset the defensive filter reads."""
    return {
        "content_type": candidate.content_type,
        "retrieval_eligible": candidate.retrieval_eligible,
        "evidence_eligible": candidate.evidence_eligible,
        "corpus_role": candidate.corpus_role,
        "year": candidate.year,
        "document_id": candidate.document_id,
        "file_hash": candidate.file_hash,
        "doi": candidate.doi,
        "publication_type": candidate.publication_type,
        "ocr_involved": candidate.ocr_involved,
    }


def _order(candidates: Iterable[Candidate], *, rerank: bool) -> list[Candidate]:
    """Deterministic order: rerank score when ranking, else dense score."""
    key = (
        (lambda candidate: (
            -(candidate.rerank_score if candidate.rerank_score is not None else float("-inf")),
            candidate.dense_rank if candidate.dense_rank is not None else 10**6,
            candidate.chunk_id,
        ))
        if rerank
        else (lambda candidate: (
            -(candidate.dense_score if candidate.dense_score is not None else float("-inf")),
            candidate.dense_rank if candidate.dense_rank is not None else 10**6,
            candidate.chunk_id,
        ))
    )
    return sorted(candidates, key=key)


class RetrievalService:
    """Two-stage retrieval over the RAG-008 Qdrant collection."""

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        reranker: Reranker | None = None,
        client: QdrantClient | None = None,
        settings: Settings | None = None,
        collection: str | None = None,
        qdrant_path: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        final_k: int = DEFAULT_FINAL_K,
        max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
        suppress_duplicates: bool = True,
        duplicate_overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
        max_neighbour_distance: int = DEFAULT_MAX_INDEX_DISTANCE,
        max_candidate_rounds: int = DEFAULT_MAX_CANDIDATE_ROUNDS,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        exclude_bibliography: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.max_chunks_per_document = max_chunks_per_document
        self.suppress_duplicates = suppress_duplicates
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.max_neighbour_distance = max_neighbour_distance
        self.max_candidate_rounds = max(1, max_candidate_rounds)
        self.max_candidates = max(candidate_k, max_candidates)
        self.exclude_bibliography = exclude_bibliography
        self.collection = collection or self.settings.qdrant_collection
        self.retriever = DenseRetriever(
            collection=self.collection,
            client=client,
            qdrant_path=qdrant_path if client is None else None,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedder=embedder,
        )

    # --- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        """Close the Qdrant client if this service created it."""
        self.retriever.close()

    def __enter__(self) -> "RetrievalService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def warm_up(self) -> dict[str, float]:
        """Load both models and run trivial work, returning warm-up seconds."""
        timings: dict[str, float] = {}
        timings["embedder_seconds"] = self.embedder.warm_up()
        if self.reranker is not None:
            warm_up = getattr(self.reranker, "warm_up", None)
            timings["reranker_seconds"] = float(warm_up()) if callable(warm_up) else 0.0
        return timings

    # --- audit ----------------------------------------------------------------
    def _audit(
        self,
        *,
        raw_query: str,
        normalized: str,
        mode: RetrievalMode,
        filters: RetrievalFilters,
        candidate_k: int,
        final_k: int,
        rerank: bool,
        min_dense_score: float | None,
        min_rerank_score: float | None,
        max_rounds: int,
        exclude_bibliography: bool,
    ) -> QueryAudit:
        embedding_spec = self.embedder.spec
        reranker_spec = self.reranker.spec if rerank and self.reranker is not None else None
        return QueryAudit(
            raw_query=raw_query,
            normalized_query=normalized,
            retrieval_mode=mode,
            filters=filters.describe(),
            candidate_k=candidate_k,
            final_k=final_k,
            rerank_enabled=bool(rerank and self.reranker is not None),
            rerank_model=reranker_spec.model_name if reranker_spec else None,
            rerank_revision=reranker_spec.model_revision if reranker_spec else None,
            rerank_device=reranker_spec.device if reranker_spec else None,
            embedding_model=embedding_spec.model_name,
            embedding_revision=embedding_spec.model_revision,
            embedding_device=embedding_spec.device,
            collection=self.collection,
            max_chunks_per_document=self.max_chunks_per_document,
            suppress_duplicates=self.suppress_duplicates,
            duplicate_overlap_threshold=self.duplicate_overlap_threshold,
            min_dense_score=min_dense_score,
            min_rerank_score=min_rerank_score,
            max_candidate_rounds=max_rounds,
            exclude_bibliography=exclude_bibliography,
            timestamp=_datetime.datetime.now(_datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
        )

    # --- main entry point -----------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.EVIDENCE,
        candidate_k: int | None = None,
        final_k: int | None = None,
        filters: RetrievalFilters | None = None,
        rerank: bool = True,
        min_dense_score: float | None = None,
        min_rerank_score: float | None = None,
        max_chunks_per_document: int | None = None,
        suppress_duplicates: bool | None = None,
        max_rounds: int | None = None,
        exclude_bibliography: bool | None = None,
    ) -> RetrievalResponse:
        """Run the full pipeline for one query.

        ``candidate_k`` is the *first* search depth. Because this corpus holds
        several physical copies of some papers, a first stage can be dominated by
        one or two works and yield fewer than ``final_k`` survivors; in that case
        the search is widened (up to ``max_rounds`` rounds, each adding
        ``candidate_k``) instead of silently returning a short list. Every extra
        round is reported in the stats and the same candidate is never scored twice.

        ``exclude_bibliography`` controls the retrieval-time bibliography guard
        (default: on; see :mod:`app.retrieval.bibliography`). Turning it on does
        nothing for GUIDANCE and SUPERVISION, whose results are scaffolding material
        rather than answer evidence.
        """
        candidate_k = self.candidate_k if candidate_k is None else candidate_k
        final_k = self.final_k if final_k is None else final_k
        rounds_cap = self.max_candidate_rounds if max_rounds is None else max(1, max_rounds)
        filters = filters or RetrievalFilters()
        suppress = self.suppress_duplicates if suppress_duplicates is None else suppress_duplicates
        keep_bibliography = (
            not self.exclude_bibliography
            if exclude_bibliography is None
            else not exclude_bibliography
        )
        bibliography_guard = (not keep_bibliography) and mode_excludes_bibliography(mode)
        limit = self.max_chunks_per_document if max_chunks_per_document is None else max_chunks_per_document
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be positive")

        normalized = normalize_query(query)
        if not normalized:
            raise ValueError("query is empty after normalization")

        wanted_rerank = bool(rerank and self.reranker is not None)
        started = time.perf_counter()

        embedding_started = time.perf_counter()
        query_vector = self.retriever.embed_query(normalized)
        embedding_seconds = time.perf_counter() - embedding_started

        dropped: list[DroppedCandidate] = []
        recorded: set[str] = set()
        pool: list[Candidate] = []
        seen: set[str] = set()
        dense_candidates: list[Candidate] = []
        search_seconds = 0.0
        rerank_seconds = 0.0
        rounds = 0
        search_limit = candidate_k
        while True:
            points, seconds = self.retriever.search(
                query_vector, limit=search_limit, mode=mode, filters=filters
            )
            search_seconds += seconds
            rounds += 1
            fresh = self._filter_new(
                points,
                seen,
                mode,
                filters,
                dropped,
                recorded,
                bibliography_guard=bibliography_guard,
            )
            if rounds == 1:
                dense_candidates = list(fresh)
            if fresh and wanted_rerank:
                fresh, spent = self._rerank(normalized, fresh)
                rerank_seconds += spent
            pool = self._merge(pool, fresh, rerank=wanted_rerank)

            survivors, by_min = self._apply_min_scores(
                pool, min_dense_score, min_rerank_score, dropped, recorded
            )
            if suppress:
                survivors, _ = suppress_near_duplicates(
                    survivors,
                    overlap_threshold=self.duplicate_overlap_threshold,
                    max_index_distance=self.max_neighbour_distance,
                )
            survivors, _ = apply_document_limit(survivors, max_chunks_per_document=limit)

            exhausted = len(points) < search_limit
            if len(survivors) >= final_k or rounds >= rounds_cap or exhausted:
                break
            search_limit = min(search_limit + candidate_k, self.max_candidates)

        # One final pass records the drops of the configuration that is returned.
        kept, _ = self._apply_min_scores(
            pool, min_dense_score, min_rerank_score, dropped, recorded
        )
        if suppress:
            kept, duplicate_drops = suppress_near_duplicates(
                kept,
                overlap_threshold=self.duplicate_overlap_threshold,
                max_index_distance=self.max_neighbour_distance,
            )
            self._record(dropped, recorded, duplicate_drops)
        kept, limit_drops = apply_document_limit(kept, max_chunks_per_document=limit)
        self._record(dropped, recorded, limit_drops)

        results = [
            candidate.model_copy(update={"final_rank": rank})
            for rank, candidate in enumerate(kept[:final_k], start=1)
        ]
        for candidate in kept[final_k:]:
            dropped.append(
                DroppedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    reason="beyond_final_k",
                    detail=f"final_k = {final_k}",
                    dense_rank=candidate.dense_rank,
                    rerank_score=candidate.rerank_score,
                )
            )

        total_seconds = time.perf_counter() - started
        audit = self._audit(
            raw_query=query,
            normalized=normalized,
            mode=mode,
            filters=filters,
            candidate_k=candidate_k,
            final_k=final_k,
            rerank=wanted_rerank,
            min_dense_score=min_dense_score,
            min_rerank_score=min_rerank_score,
            max_rounds=rounds_cap,
            exclude_bibliography=bibliography_guard,
        )
        stats = RetrievalStats(
            candidates_retrieved=len(seen),
            candidates_after_filter=len(dense_candidates),
            dropped_by_post_filter=sum(
                1
                for item in dropped
                if item.reason
                in (REASON_MODE, REASON_REFERENCES, REASON_INELIGIBLE, REASON_BIBLIOGRAPHY)
            ),
            dropped_as_bibliography=sum(
                1 for item in dropped if item.reason == REASON_BIBLIOGRAPHY
            ),
            dropped_by_min_score=sum(
                1 for item in dropped if item.reason.startswith("min_")
            ),
            dropped_as_duplicate=sum(
                1 for item in dropped if item.reason in (REASON_DUPLICATE_TEXT, REASON_DUPLICATE_NEIGHBOUR)
            ),
            dropped_by_diversity=sum(
                1 for item in dropped if item.reason == REASON_DOCUMENT_LIMIT
            ),
            returned=len(results),
            candidate_rounds=rounds,
            candidate_pool=len(pool),
            embedding_seconds=embedding_seconds,
            search_seconds=search_seconds,
            rerank_seconds=rerank_seconds,
            total_seconds=total_seconds,
        )
        return RetrievalResponse(
            audit=audit,
            results=results,
            dense_candidates=dense_candidates,
            dropped=dropped,
            stats=stats,
        )

    # --- pipeline steps -------------------------------------------------------
    def _filter_new(
        self,
        points: Sequence[Any],
        seen: set[str],
        mode: RetrievalMode,
        filters: RetrievalFilters,
        dropped: list[DroppedCandidate],
        recorded: set[str],
        *,
        bibliography_guard: bool = False,
    ) -> list[Candidate]:
        """Convert unseen points, verify them locally and drop the rest."""
        fresh: list[Candidate] = []
        for point in points:
            candidate = candidate_from_point(point)
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            candidate = candidate.model_copy(update={"dense_rank": len(seen)})
            reason = rejection_reason(_filter_payload(candidate), mode, filters)
            if reason is None and bibliography_guard and candidate.bibliography_like:
                reason = REASON_BIBLIOGRAPHY
            if reason is not None:
                self._record(
                    dropped,
                    recorded,
                    [
                        DroppedCandidate(
                            chunk_id=candidate.chunk_id,
                            document_id=candidate.document_id,
                            reason=reason,
                            detail=(
                                candidate.bibliography_reason
                                if reason == REASON_BIBLIOGRAPHY
                                else None
                            ),
                            dense_rank=candidate.dense_rank,
                        )
                    ],
                )
                continue
            fresh.append(candidate)
        return fresh

    def _rerank(self, query: str, candidates: Sequence[Candidate]) -> tuple[list[Candidate], float]:
        """Score ``candidates`` with the reranker, returning them sorted."""
        started = time.perf_counter()
        scores = self.reranker.rerank(  # type: ignore[union-attr]
            query,
            [
                RerankInput(
                    chunk_id=candidate.chunk_id,
                    text=candidate.retrieval_text or candidate.text,
                )
                for candidate in candidates
            ],
        )
        spent = time.perf_counter() - started
        score_by_id = {item.chunk_id: item.score for item in scores}
        scored = [
            candidate.model_copy(update={"rerank_score": score_by_id.get(candidate.chunk_id)})
            for candidate in candidates
        ]
        return _order(scored, rerank=True), spent

    @staticmethod
    def _merge(
        existing: Sequence[Candidate], fresh: Sequence[Candidate], *, rerank: bool
    ) -> list[Candidate]:
        """Merge two candidate batches (already scored) into one ordered pool."""
        by_id: dict[str, Candidate] = {candidate.chunk_id: candidate for candidate in existing}
        for candidate in fresh:
            by_id.setdefault(candidate.chunk_id, candidate)
        return _order(list(by_id.values()), rerank=rerank)

    @staticmethod
    def _apply_min_scores(
        pool: Sequence[Candidate],
        min_dense_score: float | None,
        min_rerank_score: float | None,
        dropped: list[DroppedCandidate],
        recorded: set[str],
    ) -> tuple[list[Candidate], list[DroppedCandidate]]:
        """Drop candidates below the optional score floors."""
        if min_dense_score is None and min_rerank_score is None:
            return list(pool), []
        kept: list[Candidate] = []
        rejected: list[DroppedCandidate] = []
        for candidate in pool:
            if (
                min_dense_score is not None
                and candidate.dense_score is not None
                and candidate.dense_score < min_dense_score
            ):
                rejected.append(
                    DroppedCandidate(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        reason="min_dense_score",
                        detail=f"dense {candidate.dense_score:.4f} < {min_dense_score}",
                        dense_rank=candidate.dense_rank,
                    )
                )
                continue
            if (
                min_rerank_score is not None
                and candidate.rerank_score is not None
                and candidate.rerank_score < min_rerank_score
            ):
                rejected.append(
                    DroppedCandidate(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        reason="min_rerank_score",
                        detail=f"rerank {candidate.rerank_score:.4f} < {min_rerank_score}",
                        dense_rank=candidate.dense_rank,
                        rerank_score=candidate.rerank_score,
                    )
                )
                continue
            kept.append(candidate)
        RetrievalService._record(dropped, recorded, rejected)
        return kept, rejected

    @staticmethod
    def _record(
        dropped: list[DroppedCandidate],
        recorded: set[str],
        items: Iterable[DroppedCandidate],
    ) -> None:
        """Append drops once per chunk id (rounds recompute the same lists)."""
        for item in items:
            key = f"{item.reason}:{item.chunk_id}"
            if key in recorded:
                continue
            recorded.add(key)
            dropped.append(item)
