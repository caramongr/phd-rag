"""Indexing-layer models: options, per-run counters and failure rows (RAG-008)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.config import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_QDRANT_COLLECTION, DEFAULT_QDRANT_UPLOAD_BATCH_SIZE
from app.models.chunk import Chunk

#: Kinds the pilot must contain so role and OCR handling are exercised (§26).
REQUIRED_PILOT_KINDS: tuple[str, ...] = ("literature", "ocr", "my_work", "future_chapters")

#: Guard against walking the whole corpus while hunting for a missing kind.
DEFAULT_PILOT_MAX_DOCUMENTS = 150


class IndexingOptions(BaseModel):
    """Configuration of one indexing run (Qdrant/embedder objects are injected)."""

    model_config = ConfigDict(extra="forbid")

    collection: str = DEFAULT_QDRANT_COLLECTION
    embedding_batch_size: int = Field(default=DEFAULT_EMBEDDING_BATCH_SIZE, ge=1)
    upload_batch_size: int = Field(default=DEFAULT_QDRANT_UPLOAD_BATCH_SIZE, ge=1)

    #: Restrict the corpus visit order to the first N documents.
    limit: int = Field(default=0, ge=0)
    #: Pilot mode: index ~this many chunks (one per document, required kinds kept).
    pilot: int = Field(default=0, ge=0)
    pilot_max_documents: int = Field(default=DEFAULT_PILOT_MAX_DOCUMENTS, ge=1)
    #: OCR fallback during parsing (``None`` = RAG-005 default).
    ocr: bool | None = None

    #: Embedding cache.
    cache_path: str | None = None
    use_cache: bool = True

    #: Point set synchronisation.
    delete_stale: bool = False

    #: Where to write the index manifest (``None`` = do not write one).
    manifest_path: str | None = None
    #: Measure BGE-M3 token counts while streaming (tokenizer only, no embedding).
    measure_tokens: bool = False
    #: ``False`` turns the run into a measurement pass: no embeddings are computed
    #: (used by ``--dry-run --measure-tokens``, which must never embed).
    compute_embeddings: bool = True


class IndexableChunk(BaseModel):
    """One chunk ready for embedding, with the document identity it belongs to."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    chunk: Chunk
    file_hash: str
    filename: str
    retrieval_text: str

    @property
    def chunk_id(self) -> str:
        """Deterministic RAG-007 chunk identity (used as the point id)."""
        return str(self.chunk.id)

    @property
    def document_id(self) -> str:
        """Deterministic RAG-007 logical-document identity."""
        return str(self.chunk.document_id)


class FailureRow(BaseModel):
    """One chunk (or batch) that failed, kept so failures are never silent."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str | None = None
    filename: str | None = None
    stage: str
    error: str


class TokenStats(BaseModel):
    """BGE-M3 tokenizer measurements over the chunk texts of a pass."""

    model_config = ConfigDict(extra="forbid")

    measured_chunks: int = 0
    token_total: int = 0
    token_max: int = 0
    token_median: int = 0
    token_p95: int = 0
    token_min: int = 0
    chunks_above_1200: int = 0
    chunks_above_model_limit: int = 0
    model_limit: int | None = None
    estimator_token_max: int = 0


class CorpusPopulation(BaseModel):
    """What a pass would index (used by the dry run and the index report)."""

    model_config = ConfigDict(extra="forbid")

    unique_documents: int = 0
    documents_chunked: int = 0
    total_chunks: int = 0
    retrieval_eligible_chunks: int = 0
    excluded_ineligible_chunks: int = 0
    excluded_reference_chunks: int = 0
    by_corpus_role: dict[str, int] = Field(default_factory=dict)
    by_content_type: dict[str, int] = Field(default_factory=dict)
    chunking_version: str = ""
    tokenizer_name: str = ""


class DryRunReport(BaseModel):
    """Result of ``--dry-run``: intent, validated configuration, no mutations."""

    model_config = ConfigDict(extra="forbid")

    root: str
    collection: str
    qdrant_target: str
    embedding_model: str
    embedding_batch_size: int
    upload_batch_size: int
    population: CorpusPopulation
    pilot_chunks: int = 0
    pilot_kinds: list[str] = Field(default_factory=list)
    pilot_documents_visited: int = 0
    pilot_missing_kinds: list[str] = Field(default_factory=list)
    cache_path: str | None = None
    manifest_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class IndexReport(BaseModel):
    """Result of one real indexing run (§23 of the task)."""

    model_config = ConfigDict(extra="forbid")

    root: str
    collection: str
    qdrant_target: str
    dry_run: bool = False

    unique_documents: int = 0
    total_chunks: int = 0
    retrieval_eligible_chunks: int = 0
    chunks_embedded: int = 0
    points_indexed: int = 0
    excluded_reference_chunks: int = 0
    excluded_ineligible_chunks: int = 0
    documents_visited: int = 0
    pilot_kinds: list[str] = Field(default_factory=list)

    embedding_model: str = ""
    embedding_revision: str | None = None
    embedding_identifier: str = ""
    embedding_library: str = ""
    embedding_library_version: str | None = None
    vector_dimension: int = 0
    distance: str = ""
    normalization: str = ""
    device: str = ""
    max_tokens: int | None = None

    embedding_seconds: float = 0.0
    embeddings_per_second: float = 0.0
    chunks_per_second: float = 0.0
    tokens_per_second: float = 0.0
    embedding_batch_size: int = 0
    upload_batch_size: int = 0
    upload_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_allocated_gib: float = 0.0
    peak_reserved_gib: float = 0.0

    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0

    collection_created: bool = False
    payload_indexes: list[str] = Field(default_factory=list)
    points_before: int = 0
    points_after: int = 0
    inserted: int = 0
    updated: int = 0
    duplicate_point_ids: int = 0
    stale_points: int = 0
    deleted_stale_points: int = 0
    stale_point_ids: list[str] = Field(default_factory=list)

    chunking_version: str = ""
    corpus_fingerprint: str = ""
    document_hashes_digest: str = ""
    chunk_ids_digest: str = ""

    embedding_failures: list[FailureRow] = Field(default_factory=list)
    upload_failures: list[FailureRow] = Field(default_factory=list)
    skipped_chunks: int = 0

    token_stats: TokenStats | None = None
    manifest_path: str | None = None
    total_seconds: float = 0.0
    population: CorpusPopulation | None = None
    notes: list[str] = Field(default_factory=list)


def chunk_kinds(chunk: Chunk) -> set[str]:
    """Kinds a chunk represents, for pilot coverage (role plus OCR)."""
    kinds: set[str] = set()
    if chunk.corpus_role is not None:
        kinds.add(chunk.corpus_role.value)
    if chunk.ocr_involved:
        kinds.add("ocr")
    return kinds
