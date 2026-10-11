"""Dense candidate retrieval against Qdrant (RAG-009).

This stage is deliberately thin: embed the query with the *same* RAG-008 wrapper
that embedded the corpus, ask Qdrant for the nearest points under a filter, and
convert the payloads into :class:`~app.retrieval.models.Candidate` objects.

Nothing is re-scored and nothing is re-ranked here. The dense score is kept as
returned, because the reranker's score is a different quantity entirely.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from app.embedding.cache import unit_norm
from app.embedding.models import EmbeddingModel
from app.indexing.collection import build_client
from app.retrieval.bibliography import classify
from app.retrieval.models import Candidate
from app.retrieval.filters import build_qdrant_filter
from app.retrieval.models import RetrievalFilters, RetrievalMode


class RetrievalError(RuntimeError):
    """Base class for retrieval-layer failures."""


class CollectionNotReady(RetrievalError):
    """The Qdrant collection is missing or has the wrong vector geometry."""


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def candidate_from_point(
    point: Any,
    *,
    dense_rank: int | None = None,
) -> Candidate:
    """Convert one Qdrant point (with payload) into a :class:`Candidate`.

    Every indexed field is copied through even when it is ``None``; a result that
    silently drops ``doi`` or ``source_locators`` would make citation rendering
    guess later.

    The bibliography verdict is computed here, once, from the chunk text, and travels
    with the candidate so the filter can consult it and the result can explain it.
    """
    payload: dict[str, Any] = dict(getattr(point, "payload", None) or {})
    text = payload.get("text") or ""
    verdict = classify(text)
    return Candidate(
        chunk_id=str(payload.get("chunk_id") or getattr(point, "id", "")),
        document_id=str(payload.get("document_id") or ""),
        file_hash=str(payload.get("file_hash") or ""),
        filename=str(payload.get("filename") or ""),
        chunk_index=payload.get("chunk_index"),
        dense_score=float(point.score) if getattr(point, "score", None) is not None else None,
        dense_rank=dense_rank,
        text=payload.get("text") or "",
        retrieval_text=payload.get("retrieval_text") or "",
        title=payload.get("title"),
        first_author=payload.get("first_author"),
        year=payload.get("year"),
        doi=payload.get("doi"),
        publication_type=payload.get("publication_type"),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        section_title=payload.get("section_title"),
        heading_path=_as_tuple(payload.get("heading_path")),
        content_type=str(payload.get("content_type") or ""),
        corpus_role=payload.get("corpus_role"),
        authority=payload.get("authority"),
        evidence_eligible=bool(payload.get("evidence_eligible")),
        retrieval_eligible=bool(payload.get("retrieval_eligible")),
        bibliography_like=verdict.is_bibliography_like,
        bibliography_reason=verdict.reason,
        text_sources=_as_tuple(payload.get("text_sources")),
        ocr_involved=bool(payload.get("ocr_involved")),
        source_locators=_as_tuple(payload.get("source_locators")),
        token_count=payload.get("token_count"),
    )


class DenseRetriever:
    """Query embedding plus Qdrant nearest-neighbour search."""

    def __init__(
        self,
        *,
        collection: str,
        client: QdrantClient | None = None,
        qdrant_path: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        embedder: EmbeddingModel,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self._own_client = client is None
        self._client = client or build_client(
            path=qdrant_path, url=qdrant_url, api_key=qdrant_api_key
        )
        self._dimension: int | None = None

    # --- lifecycle ------------------------------------------------------------
    @property
    def client(self) -> QdrantClient:
        """The underlying Qdrant client."""
        return self._client

    def close(self) -> None:
        """Close the client when this retriever created it."""
        if self._own_client:
            self._client.close()

    def __enter__(self) -> "DenseRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- geometry -------------------------------------------------------------
    def _collection_dimension(self) -> int:
        try:
            info = self._client.get_collection(self.collection)
        except Exception as exc:
            raise CollectionNotReady(
                f"collection {self.collection!r} is not available: {exc}"
            ) from exc
        size = getattr(info.config.params.vectors, "size", None)
        if size is None:
            raise CollectionNotReady(
                f"collection {self.collection!r} does not use a single dense vector"
            )
        return int(size)

    def dimension(self) -> int:
        """Vector dimension of the collection (validated once)."""
        if self._dimension is None:
            model_dimension = self.embedder.dimension
            collection_dimension = self._collection_dimension()
            if model_dimension != collection_dimension:
                raise CollectionNotReady(
                    f"embedding model produces {model_dimension}-d vectors but "
                    f"collection {self.collection!r} stores {collection_dimension}-d"
                )
            self._dimension = collection_dimension
        return self._dimension

    def point_count(self) -> int:
        """Number of points in the collection."""
        return int(self._client.count(self.collection, exact=True).count)

    # --- search ---------------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        """Embed one (already normalized) query and verify its geometry."""
        vector = self.embedder.embed_query(query)
        expected = self.dimension()
        if len(vector) != expected:
            raise RetrievalError(
                f"query embedding has {len(vector)} dimensions, expected {expected}"
            )
        norm = unit_norm(vector)
        if abs(norm - 1.0) > 1e-3:
            raise RetrievalError(
                f"query embedding is not L2-normalized (norm={norm:.6f}); "
                "cosine scores would be misleading"
            )
        return vector

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        mode: RetrievalMode = RetrievalMode.EVIDENCE,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[Any], float]:
        """Return Qdrant points (with payload, no vectors) plus search seconds."""
        if limit <= 0:
            return [], 0.0
        query_filter = build_qdrant_filter(mode, filters)
        started = time.perf_counter()
        response = self._client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        seconds = time.perf_counter() - started
        return list(response.points), seconds

    def retrieve(
        self,
        query: str,
        *,
        limit: int,
        mode: RetrievalMode = RetrievalMode.EVIDENCE,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[Candidate], dict[str, float]]:
        """Embed ``query``, search, and return ranked candidates plus timings."""
        embedding_started = time.perf_counter()
        vector = self.embed_query(query)
        embedding_seconds = time.perf_counter() - embedding_started
        points, search_seconds = self.search(
            vector, limit=limit, mode=mode, filters=filters
        )
        candidates = [
            candidate_from_point(point, dense_rank=index)
            for index, point in enumerate(points, start=1)
        ]
        timings = {
            "embedding_seconds": embedding_seconds,
            "search_seconds": search_seconds,
        }
        return candidates, timings


def full_filter(mode: RetrievalMode, filters: RetrievalFilters | None) -> models.Filter | None:
    """Expose the built filter for inspection/tests."""
    return build_qdrant_filter(mode, filters)
