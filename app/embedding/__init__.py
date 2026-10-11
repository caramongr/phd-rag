"""Embedding layer (RAG-008).

Dense BGE-M3 embeddings with a cache and a streaming batch API. Nothing here knows
about Qdrant, chunking or retrieval: the indexer composes these pieces.
"""

from app.embedding.batch import EmbedOutcome, EmbedRequest, embed_requests, embed_stream, iter_batches
from app.embedding.bge_m3 import BgeM3Embedder, resolve_device, resolve_snapshot
from app.embedding.cache import CACHE_FORMAT_VERSION, EmbeddingCache, cache_key, text_hash, unit_norm
from app.embedding.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIBRARY,
    DEFAULT_MODEL_NAME,
    EmbeddingError,
    EmbeddingFailure,
    EmbeddingModel,
    EmbeddingModelLoadError,
    EmbeddingSpec,
    as_vector_list,
    count_tokens_many,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LIBRARY",
    "DEFAULT_MODEL_NAME",
    "BgeM3Embedder",
    "EmbedOutcome",
    "EmbedRequest",
    "EmbeddingCache",
    "EmbeddingError",
    "EmbeddingFailure",
    "EmbeddingModel",
    "EmbeddingModelLoadError",
    "EmbeddingSpec",
    "as_vector_list",
    "cache_key",
    "count_tokens_many",
    "embed_requests",
    "embed_stream",
    "iter_batches",
    "resolve_device",
    "resolve_snapshot",
    "text_hash",
    "unit_norm",
]
