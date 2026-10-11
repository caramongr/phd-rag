"""Qdrant indexing of the RAG-007 chunk corpus (RAG-008).

Dense BGE-M3 vectors for retrieval-eligible chunks, stored as deterministic points
(RAG-007 chunk UUIDs) in a cosine collection whose payload carries the provenance
RAG-009 will filter on. Retrieval ranking, reranking and generation are explicitly
*not* part of this layer.
"""

from app.indexing.collection import (
    DEFAULT_DISTANCE,
    DEFAULT_PAYLOAD_INDEXES,
    CollectionDimensionMismatch,
    CollectionSchemaError,
    CollectionSetup,
    IndexingError,
    QdrantConfigurationError,
    build_client,
    count_points,
    delete_points,
    ensure_collection,
    existing_dimension,
    fetch_payloads,
    make_points,
    sample_payloads,
    scroll_point_ids,
    upsert_points,
)
from app.indexing.fingerprint import (
    FINGERPRINT_VERSION,
    chunk_ids_digest,
    corpus_fingerprint,
    document_hashes_digest,
)
from app.indexing.indexer import (
    DISTANCE,
    NORMALIZATION,
    describe_pilot,
    describe_population,
    dry_run,
    iter_population,
    run_index,
)
from app.indexing.manifest import IndexManifest, manifest_path, read_manifest, write_manifest
from app.indexing.models import (
    REQUIRED_PILOT_KINDS,
    CorpusPopulation,
    DryRunReport,
    FailureRow,
    IndexableChunk,
    IndexingOptions,
    IndexReport,
    TokenStats,
)
from app.indexing.payload import (
    AUDIT_PAYLOAD_FIELDS,
    INDEXED_PAYLOAD_FIELDS,
    build_payload,
    payload_is_complete,
)

__all__ = [
    "AUDIT_PAYLOAD_FIELDS",
    "DEFAULT_DISTANCE",
    "DEFAULT_PAYLOAD_INDEXES",
    "DISTANCE",
    "FINGERPRINT_VERSION",
    "INDEXED_PAYLOAD_FIELDS",
    "NORMALIZATION",
    "REQUIRED_PILOT_KINDS",
    "CollectionDimensionMismatch",
    "CollectionSchemaError",
    "CollectionSetup",
    "CorpusPopulation",
    "DryRunReport",
    "FailureRow",
    "IndexManifest",
    "IndexReport",
    "IndexableChunk",
    "IndexingError",
    "IndexingOptions",
    "QdrantConfigurationError",
    "TokenStats",
    "build_client",
    "build_payload",
    "chunk_ids_digest",
    "corpus_fingerprint",
    "count_points",
    "delete_points",
    "describe_pilot",
    "describe_population",
    "document_hashes_digest",
    "dry_run",
    "ensure_collection",
    "existing_dimension",
    "fetch_payloads",
    "iter_population",
    "make_points",
    "manifest_path",
    "payload_is_complete",
    "read_manifest",
    "run_index",
    "sample_payloads",
    "scroll_point_ids",
    "upsert_points",
    "write_manifest",
]
