"""The index manifest: what produced the vectors in a collection (RAG-008).

A collection name alone cannot tell a future reader which model, revision,
normalization, chunking version or corpus produced the points, so every indexing
run writes (and re-writes) a manifest next to its configuration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class IndexManifest(BaseModel):
    """Machine-readable description of one index build."""

    model_config = ConfigDict(extra="forbid")

    collection: str
    qdrant_target: str
    distance: str
    vector_dimension: int

    embedding_model: str
    embedding_revision: str | None = None
    embedding_identifier: str
    embedding_library: str
    embedding_library_version: str | None = None
    normalization: str
    device: str

    chunking_version: str
    chunk_count: int = 0
    indexed_chunk_count: int = 0
    document_count: int = 0
    #: Non-zero when the collection holds a pilot sample rather than the whole
    #: indexing population (the collection name alone must never imply "complete").
    pilot_chunk_limit: int = 0

    corpus_fingerprint: str
    document_hashes_digest: str
    chunk_ids_digest: str

    points_total: int = 0
    stale_points: int = 0
    deleted_stale_points: int = 0
    duplicate_point_ids: int = 0

    build_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def manifest_path(directory: str | Path, collection: str) -> Path:
    """Canonical manifest file for ``collection`` inside ``directory``."""
    return Path(directory) / f"{collection}.index-manifest.json"


def write_manifest(path: str | Path, manifest: IndexManifest) -> Path:
    """Write the manifest as UTF-8 JSON (parent directories created)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def read_manifest(path: str | Path) -> IndexManifest:
    """Read a manifest written by :func:`write_manifest`."""
    return IndexManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
