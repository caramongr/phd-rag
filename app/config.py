"""Application configuration.

Settings are read from environment variables at process start and cached. The
layer stays deliberately small and dependency-free (no settings library) so the
values and their defaults are easy to audit.

OCR settings (RAG-005) are centralised here so thresholds and the language/DPI
configuration are never scattered through parser code.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict

DEFAULT_APP_NAME = "PhD Academic RAG"
DEFAULT_APP_VERSION = "0.1.0"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_API_PREFIX = "/api"

# --- OCR defaults (RAG-005) ---
DEFAULT_OCR_ENABLED = True
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_MIN_NATIVE_CHARS = 50

# --- Metadata enrichment defaults (RAG-006) ---
DEFAULT_METADATA_ENRICH_EXTERNAL = True
DEFAULT_METADATA_CACHE_PATH = ".metadata_cache/cache.json"
DEFAULT_METADATA_REQUEST_TIMEOUT = 10.0
DEFAULT_METADATA_MAX_RETRIES = 2
DEFAULT_METADATA_MIN_REQUEST_INTERVAL = 1.0

# --- Embedding defaults (RAG-008) ---
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
#: 16 is the measured throughput optimum for BGE-M3 on this corpus: identical
#: speed to 32/64/128 (132.5 embeddings/s steady state) at ~4 GiB reserved VRAM,
#: so it leaves ~28 GiB of headroom on a 32 GiB RTX 5090 (RAG-008 GPU benchmark).
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_CACHE_PATH = ".embedding_cache/embeddings.npz"
DEFAULT_EMBEDDING_LOCAL_FILES_ONLY = False

# --- Qdrant / indexing defaults (RAG-008) ---
DEFAULT_QDRANT_COLLECTION = "phd_rag_chunks_v1"
DEFAULT_QDRANT_PATH = ".qdrant_data"
DEFAULT_QDRANT_UPLOAD_BATCH_SIZE = 128
DEFAULT_INDEX_MANIFEST_DIR = ".index_cache/manifests"

# --- Reranking defaults (RAG-009) ---
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
#: 16 is the measured optimum for a 25-candidate query (86 pairs/s vs 78 at batch 8
#: and 66 at batch 32) at ~4.9 GiB reserved VRAM, so it leaves ~27 GiB of headroom
#: on a 32 GiB RTX 5090 (RAG-009 reranker benchmark).
DEFAULT_RERANKER_BATCH_SIZE = 16

# --- Retrieval defaults (RAG-009) ---
DEFAULT_RETRIEVAL_CANDIDATE_K = 25
DEFAULT_RETRIEVAL_FINAL_K = 8
DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT = 3
DEFAULT_RETRIEVAL_DUPLICATE_OVERLAP = 0.85
DEFAULT_RETRIEVAL_MAX_ROUNDS = 3
DEFAULT_RETRIEVAL_MODE = "evidence"
#: Withhold chunks that are reference lists but were not classified as REFERENCES
#: upstream (RAG-009 bibliography guard).
DEFAULT_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY = True


class Settings(BaseModel):
    """Runtime configuration for the backend.

    Fields are sourced from ``PHD_RAG_*`` environment variables, falling back to
    safe development defaults.
    """

    model_config = ConfigDict(frozen=True)

    app_name: str = DEFAULT_APP_NAME
    app_version: str = DEFAULT_APP_VERSION
    environment: str = DEFAULT_ENVIRONMENT
    debug: bool = False
    api_prefix: str = DEFAULT_API_PREFIX

    # OCR (RAG-005)
    ocr_enabled: bool = DEFAULT_OCR_ENABLED
    ocr_language: str = DEFAULT_OCR_LANGUAGE
    ocr_dpi: int = DEFAULT_OCR_DPI
    ocr_min_native_chars: int = DEFAULT_OCR_MIN_NATIVE_CHARS

    # Metadata enrichment (RAG-006)
    metadata_enrich_external: bool = DEFAULT_METADATA_ENRICH_EXTERNAL
    metadata_cache_path: str = DEFAULT_METADATA_CACHE_PATH
    metadata_request_timeout: float = DEFAULT_METADATA_REQUEST_TIMEOUT
    metadata_max_retries: int = DEFAULT_METADATA_MAX_RETRIES
    metadata_min_request_interval: float = DEFAULT_METADATA_MIN_REQUEST_INTERVAL
    metadata_crossref_mailto: str | None = None

    # Embeddings (RAG-008)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str | None = None
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_cache_path: str = DEFAULT_EMBEDDING_CACHE_PATH
    embedding_local_files_only: bool = DEFAULT_EMBEDDING_LOCAL_FILES_ONLY

    # Qdrant / indexing (RAG-008)
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_upload_batch_size: int = DEFAULT_QDRANT_UPLOAD_BATCH_SIZE
    index_manifest_dir: str = DEFAULT_INDEX_MANIFEST_DIR

    # Reranking (RAG-009)
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_device: str | None = None
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE

    # Retrieval (RAG-009)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE
    retrieval_candidate_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K
    retrieval_final_k: int = DEFAULT_RETRIEVAL_FINAL_K
    retrieval_max_chunks_per_document: int = DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT
    retrieval_duplicate_overlap: float = DEFAULT_RETRIEVAL_DUPLICATE_OVERLAP
    retrieval_max_rounds: int = DEFAULT_RETRIEVAL_MAX_ROUNDS
    retrieval_exclude_bibliography: bool = DEFAULT_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY


def _parse_bool(value: str) -> bool:
    """Interpret common truthy strings from the environment."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings(
        app_name=os.getenv("PHD_RAG_APP_NAME", DEFAULT_APP_NAME),
        app_version=os.getenv("PHD_RAG_APP_VERSION", DEFAULT_APP_VERSION),
        environment=os.getenv("PHD_RAG_ENVIRONMENT", DEFAULT_ENVIRONMENT),
        debug=_parse_bool(os.getenv("PHD_RAG_DEBUG", "false")),
        api_prefix=os.getenv("PHD_RAG_API_PREFIX", DEFAULT_API_PREFIX),
        ocr_enabled=_parse_bool(os.getenv("PHD_RAG_OCR_ENABLED", "true")),
        ocr_language=os.getenv("PHD_RAG_OCR_LANGUAGE", DEFAULT_OCR_LANGUAGE),
        ocr_dpi=int(os.getenv("PHD_RAG_OCR_DPI", str(DEFAULT_OCR_DPI))),
        ocr_min_native_chars=int(
            os.getenv("PHD_RAG_OCR_MIN_NATIVE_CHARS", str(DEFAULT_OCR_MIN_NATIVE_CHARS))
        ),
        metadata_enrich_external=_parse_bool(
            os.getenv("PHD_RAG_METADATA_ENRICH_EXTERNAL", "true")
        ),
        metadata_cache_path=os.getenv(
            "PHD_RAG_METADATA_CACHE_PATH", DEFAULT_METADATA_CACHE_PATH
        ),
        metadata_request_timeout=float(
            os.getenv("PHD_RAG_METADATA_REQUEST_TIMEOUT", str(DEFAULT_METADATA_REQUEST_TIMEOUT))
        ),
        metadata_max_retries=int(
            os.getenv("PHD_RAG_METADATA_MAX_RETRIES", str(DEFAULT_METADATA_MAX_RETRIES))
        ),
        metadata_min_request_interval=float(
            os.getenv(
                "PHD_RAG_METADATA_MIN_REQUEST_INTERVAL",
                str(DEFAULT_METADATA_MIN_REQUEST_INTERVAL),
            )
        ),
        metadata_crossref_mailto=os.getenv("PHD_RAG_METADATA_CROSSREF_MAILTO") or None,
        embedding_model=os.getenv("PHD_RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_device=os.getenv("PHD_RAG_EMBEDDING_DEVICE") or None,
        embedding_batch_size=int(
            os.getenv("PHD_RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))
        ),
        embedding_cache_path=os.getenv(
            "PHD_RAG_EMBEDDING_CACHE_PATH", DEFAULT_EMBEDDING_CACHE_PATH
        ),
        embedding_local_files_only=_parse_bool(
            os.getenv("PHD_RAG_EMBEDDING_LOCAL_FILES_ONLY", "false")
        ),
        qdrant_collection=os.getenv("PHD_RAG_QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        qdrant_url=os.getenv("PHD_RAG_QDRANT_URL") or None,
        qdrant_api_key=os.getenv("PHD_RAG_QDRANT_API_KEY") or None,
        qdrant_path=os.getenv("PHD_RAG_QDRANT_PATH", DEFAULT_QDRANT_PATH),
        qdrant_upload_batch_size=int(
            os.getenv(
                "PHD_RAG_QDRANT_UPLOAD_BATCH_SIZE", str(DEFAULT_QDRANT_UPLOAD_BATCH_SIZE)
            )
        ),
        index_manifest_dir=os.getenv("PHD_RAG_INDEX_MANIFEST_DIR", DEFAULT_INDEX_MANIFEST_DIR),
        reranker_model=os.getenv("PHD_RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        reranker_device=os.getenv("PHD_RAG_RERANKER_DEVICE") or None,
        reranker_batch_size=int(
            os.getenv("PHD_RAG_RERANKER_BATCH_SIZE", str(DEFAULT_RERANKER_BATCH_SIZE))
        ),
        retrieval_mode=os.getenv("PHD_RAG_RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE),
        retrieval_candidate_k=int(
            os.getenv("PHD_RAG_RETRIEVAL_CANDIDATE_K", str(DEFAULT_RETRIEVAL_CANDIDATE_K))
        ),
        retrieval_final_k=int(
            os.getenv("PHD_RAG_RETRIEVAL_FINAL_K", str(DEFAULT_RETRIEVAL_FINAL_K))
        ),
        retrieval_max_chunks_per_document=int(
            os.getenv(
                "PHD_RAG_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT",
                str(DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT),
            )
        ),
        retrieval_duplicate_overlap=float(
            os.getenv(
                "PHD_RAG_RETRIEVAL_DUPLICATE_OVERLAP",
                str(DEFAULT_RETRIEVAL_DUPLICATE_OVERLAP),
            )
        ),
        retrieval_max_rounds=int(
            os.getenv("PHD_RAG_RETRIEVAL_MAX_ROUNDS", str(DEFAULT_RETRIEVAL_MAX_ROUNDS))
        ),
        retrieval_exclude_bibliography=_parse_bool(
            os.getenv(
                "PHD_RAG_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY",
                "true" if DEFAULT_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY else "false",
            )
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return load_settings()
