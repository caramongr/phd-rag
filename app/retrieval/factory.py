"""One place that builds the RAG-009 retrieval service from settings.

RAG-009 is the only search system in the project, so the CLI, the combined
local/external discovery service and RAG-011 generation all build the same service
here instead of assembling embedder, reranker and Qdrant client three times. The
model classes are imported *inside* the function so existing tests that patch
``app.embedding.bge_m3.BgeM3Embedder`` / ``app.reranking.bge_reranker.BgeReranker``
keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.retrieval.service import RetrievalService


def build_retrieval_service(
    settings: Settings | None = None,
    *,
    embedder: object | None = None,
    reranker: object | None = None,
    rerank: bool = True,
    device: str | None = None,
    collection: str | None = None,
    qdrant_path: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    candidate_k: int | None = None,
    final_k: int | None = None,
    max_chunks_per_document: int | None = None,
    suppress_duplicates: bool = True,
    duplicate_overlap_threshold: float | None = None,
    max_candidate_rounds: int | None = None,
    exclude_bibliography: bool | None = None,
    client: object | None = None,
) -> "RetrievalService":
    """Build a :class:`RetrievalService` from settings, with optional overrides.

    ``embedder``/``reranker``/``client`` can be injected so callers (and tests) never
    have to load a model or open a second Qdrant store.
    """
    from app.embedding.bge_m3 import BgeM3Embedder
    from app.reranking.bge_reranker import BgeReranker
    from app.retrieval.service import RetrievalService

    settings = settings or get_settings()
    if embedder is None:
        embedder = BgeM3Embedder(
            model_name=settings.embedding_model,
            device=device or settings.embedding_device,
            local_files_only=settings.embedding_local_files_only,
        )
    if reranker is None and rerank:
        reranker = BgeReranker(
            model_name=settings.reranker_model,
            device=device or settings.reranker_device,
            batch_size=settings.reranker_batch_size,
            local_files_only=settings.embedding_local_files_only,
        )
    return RetrievalService(
        embedder=embedder,  # type: ignore[arg-type]
        reranker=reranker if rerank else None,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        collection=collection or settings.qdrant_collection,
        qdrant_path=qdrant_path or (settings.qdrant_path or None),
        qdrant_url=qdrant_url or settings.qdrant_url,
        qdrant_api_key=qdrant_api_key or settings.qdrant_api_key,
        candidate_k=candidate_k or settings.retrieval_candidate_k,
        final_k=final_k or settings.retrieval_final_k,
        max_chunks_per_document=(
            max_chunks_per_document
            if max_chunks_per_document is not None
            else settings.retrieval_max_chunks_per_document
        ),
        suppress_duplicates=suppress_duplicates,
        duplicate_overlap_threshold=(
            duplicate_overlap_threshold
            if duplicate_overlap_threshold is not None
            else settings.retrieval_duplicate_overlap
        ),
        max_candidate_rounds=(
            max_candidate_rounds if max_candidate_rounds is not None else settings.retrieval_max_rounds
        ),
        exclude_bibliography=(
            exclude_bibliography
            if exclude_bibliography is not None
            else settings.retrieval_exclude_bibliography
        ),
    )
