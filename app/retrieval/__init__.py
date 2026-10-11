"""Retrieval layer (RAG-009).

Two-stage retrieval over the RAG-008 Qdrant collection: BGE-M3 dense candidates,
an evidence/mode filter enforced at the vector-search level *and* re-checked
locally, an optional local BGE cross-encoder rerank, near-duplicate suppression
and per-document diversity. No generation happens here.
"""

from app.retrieval.bibliography import (
    BibliographyFeatures,
    BibliographyVerdict,
    classify,
    is_bibliography_like,
    measure,
)
from app.retrieval.dense import (
    CollectionNotReady,
    DenseRetriever,
    RetrievalError,
    candidate_from_point,
)
from app.retrieval.diversity import (
    apply_document_limit,
    normalized_text_hash,
    suppress_near_duplicates,
    token_overlap,
    work_key,
)
from app.retrieval.filters import (
    REASON_BIBLIOGRAPHY,
    REASON_FILTER,
    REASON_INELIGIBLE,
    REASON_MODE,
    REASON_REFERENCES,
    build_qdrant_filter,
    is_allowed,
    mode_excludes_bibliography,
    rejection_reason,
)
from app.retrieval.models import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_CANDIDATE_ROUNDS,
    GUIDANCE_ROLES,
    KNOWN_CONTENT_TYPES,
    Candidate,
    DroppedCandidate,
    QueryAudit,
    RetrievalFilters,
    RetrievalMode,
    RetrievalResponse,
    RetrievalStats,
)
from app.retrieval.query import normalize_query
from app.retrieval.service import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_K,
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    RetrievalService,
)

__all__ = [
    "BibliographyFeatures",
    "BibliographyVerdict",
    "Candidate",
    "CollectionNotReady",
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_FINAL_K",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_CANDIDATE_ROUNDS",
    "DEFAULT_MAX_CHUNKS_PER_DOCUMENT",
    "DenseRetriever",
    "DroppedCandidate",
    "GUIDANCE_ROLES",
    "KNOWN_CONTENT_TYPES",
    "QueryAudit",
    "REASON_BIBLIOGRAPHY",
    "REASON_FILTER",
    "REASON_INELIGIBLE",
    "REASON_MODE",
    "REASON_REFERENCES",
    "RetrievalError",
    "RetrievalFilters",
    "RetrievalMode",
    "RetrievalResponse",
    "RetrievalService",
    "RetrievalStats",
    "apply_document_limit",
    "build_qdrant_filter",
    "candidate_from_point",
    "classify",
    "is_allowed",
    "is_bibliography_like",
    "measure",
    "mode_excludes_bibliography",
    "normalize_query",
    "normalized_text_hash",
    "rejection_reason",
    "suppress_near_duplicates",
    "token_overlap",
    "work_key",
]
