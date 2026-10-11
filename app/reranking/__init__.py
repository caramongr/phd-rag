"""Reranking layer (RAG-009).

A local BGE cross-encoder that re-scores the candidates dense retrieval returned.
Nothing here knows about Qdrant or about the final result shape: the retrieval
service composes this with the dense stage.
"""

from app.reranking.bge_reranker import BgeReranker
from app.reranking.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIBRARY,
    DEFAULT_RERANKER_MODEL,
    Reranker,
    RerankerFailure,
    RerankerLoadError,
    RerankerSpec,
    RerankingError,
    RerankInput,
    RerankScore,
)

__all__ = [
    "BgeReranker",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LIBRARY",
    "DEFAULT_RERANKER_MODEL",
    "RerankInput",
    "RerankScore",
    "Reranker",
    "RerankerFailure",
    "RerankerLoadError",
    "RerankerSpec",
    "RerankingError",
]
