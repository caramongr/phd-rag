"""Academic-aware normalization and structure-preserving chunking (RAG-007).

This layer sits between parsing (RAG-003/004/005) and retrieval (RAG-008). It
consumes parsed documents plus their RAG-006 metadata and produces
:class:`~app.models.chunk.Chunk` objects that keep exact document, page, section
and extraction provenance. It deliberately contains no embeddings, no vector
store, no retrieval and no LLM involvement.
"""

from app.chunking.chunker import (
    DocumentChunker,
    chunk_document,
    chunk_id_for,
    document_id_for,
    split_oversized_text,
)
from app.chunking.classify import SectionKind, heading_kind, looks_like_caption, section_kind
from app.chunking.models import (
    DEFAULT_EXTENDED_MAX_TOKENS,
    DEFAULT_HARD_MAX_TOKENS,
    DEFAULT_TARGET_MAX_TOKENS,
    DEFAULT_TARGET_MIN_TOKENS,
    ChunkedDocument,
    ChunkingOptions,
    SkippedItem,
)
from app.chunking.normalize import (
    NORMALIZATION_VERSION,
    RunningLineReport,
    detect_running_lines,
    normalize_block_text,
    normalize_inline,
)
from app.chunking.policy import RolePolicy, eligibility_for, policy_for_role
from app.chunking.tokenizer import DEFAULT_TOKENIZER, SubwordEstimator, count_tokens

__all__ = [
    "DEFAULT_EXTENDED_MAX_TOKENS",
    "DEFAULT_HARD_MAX_TOKENS",
    "DEFAULT_TARGET_MAX_TOKENS",
    "DEFAULT_TARGET_MIN_TOKENS",
    "DEFAULT_TOKENIZER",
    "NORMALIZATION_VERSION",
    "ChunkedDocument",
    "ChunkingOptions",
    "DocumentChunker",
    "RolePolicy",
    "RunningLineReport",
    "SectionKind",
    "SkippedItem",
    "SubwordEstimator",
    "chunk_document",
    "chunk_id_for",
    "count_tokens",
    "detect_running_lines",
    "document_id_for",
    "eligibility_for",
    "heading_kind",
    "looks_like_caption",
    "normalize_block_text",
    "normalize_inline",
    "policy_for_role",
    "section_kind",
    "split_oversized_text",
]
