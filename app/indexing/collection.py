"""Qdrant collection management and point I/O (RAG-008).

Everything that touches Qdrant lives here, so the embedding layer stays free of
vector-database concerns and the indexer only composes the two.

Safety rules encoded in this module:

* the collection is created with the dimension **read from the model** and
  ``COSINE`` distance;
* an existing collection with a different dimension aborts the run - it is never
  recreated or emptied automatically;
* upserts are deterministic (RAG-007 chunk UUIDs), so a re-run updates the same
  points instead of duplicating them;
* a Qdrant server (URL) and a local persistent path are both first-class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient, models

from app.indexing.payload import INDEXED_PAYLOAD_FIELDS

#: Distance metric: vectors are L2-normalized, so cosine is the natural choice.
DEFAULT_DISTANCE = "Cosine"

#: Fields RAG-009 will filter on; kept deliberately short (indexes cost memory).
DEFAULT_PAYLOAD_INDEXES: tuple[tuple[str, str], ...] = (
    ("corpus_role", "keyword"),
    ("evidence_eligible", "bool"),
    ("content_type", "keyword"),
    ("year", "integer"),
    ("document_id", "keyword"),
    ("file_hash", "keyword"),
)

_SCHEMA_TYPES: dict[str, Any] = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "bool": models.PayloadSchemaType.BOOL,
    "integer": models.PayloadSchemaType.INTEGER,
}

#: Collection names must stay recognisable in a shared server.
COLLECTION_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")


class IndexingError(RuntimeError):
    """Base class for indexing failures."""


class QdrantConfigurationError(IndexingError):
    """The Qdrant connection could not be built or reached."""


class CollectionDimensionMismatch(IndexingError):
    """The existing collection was built with a different vector dimension."""


class CollectionSchemaError(IndexingError):
    """The existing collection is not a single unnamed-vector collection."""


@dataclass
class CollectionSetup:
    """What ``ensure_collection`` found or created."""

    name: str
    dimension: int
    distance: str
    created: bool
    payload_indexes: list[str] = field(default_factory=list)
    points_before: int = 0


def validate_collection_name(name: str) -> str:
    """Reject names Qdrant would reject, with a clearer message."""
    if not name or any(char.lower() not in COLLECTION_NAME_CHARS for char in name):
        raise QdrantConfigurationError(
            f"invalid collection name {name!r}: use letters, digits, '-', '_' or '.'"
        )
    return name


def build_client(
    *,
    url: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
    timeout: float = 60.0,
) -> QdrantClient:
    """Build a Qdrant client for a server URL or a local persistent path.

    ``path`` takes precedence, because a local directory is the development
    default and mixing the two would be ambiguous. A URL without a scheme is
    normalised to ``http://``.
    """
    try:
        if path:
            return QdrantClient(path=str(path), timeout=timeout)
        if url:
            normalised = url if "://" in url else f"http://{url}"
            return QdrantClient(url=normalised, api_key=api_key, timeout=timeout)
    except Exception as exc:
        raise QdrantConfigurationError(f"could not create a Qdrant client: {exc}") from exc
    raise QdrantConfigurationError(
        "no Qdrant target configured: pass --qdrant-url or --qdrant-path"
    )


def locate_client(client: QdrantClient) -> str:
    """Human-readable description of where a client points (for the manifest)."""
    if getattr(client, "_client", None) is not None and hasattr(client, "_client"):
        local = getattr(client._client, "location", None)
        if isinstance(local, str) and local:
            return local
    connection = getattr(client, "_client", None)
    if connection is not None:
        base = getattr(connection, "rest_uri", None)
        if base:
            return str(base)
    return "unknown"


def existing_dimension(client: QdrantClient, collection: str) -> int | None:
    """Vector dimension of an existing collection, or ``None`` when absent."""
    try:
        info = client.get_collection(collection)
    except Exception:
        return None
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if size is None:
        raise CollectionSchemaError(
            f"collection {collection!r} does not use a single unnamed dense vector"
        )
    return int(size)


def ensure_collection(
    client: QdrantClient,
    *,
    collection: str,
    dimension: int,
    indexes: Sequence[tuple[str, str]] = DEFAULT_PAYLOAD_INDEXES,
) -> CollectionSetup:
    """Create the collection when missing, validate it when present.

    Never destructive: a dimension mismatch raises instead of dropping data.
    """
    validate_collection_name(collection)
    if dimension < 1:
        raise QdrantConfigurationError(f"invalid vector dimension {dimension}")

    current = existing_dimension(client, collection)
    created = False
    if current is None:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )
        created = True
    elif current != dimension:
        raise CollectionDimensionMismatch(
            f"collection {collection!r} has dimension {current}, but the embedding "
            f"model produces {dimension}; refusing to replace a populated collection "
            "automatically (re-index into a new collection name instead)"
        )

    for field_name, schema in indexes:
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=_SCHEMA_TYPES[schema],
        )

    return CollectionSetup(
        name=collection,
        dimension=dimension,
        distance=DEFAULT_DISTANCE,
        created=created,
        payload_indexes=[field_name for field_name, _schema in indexes],
        points_before=count_points(client, collection),
    )


def count_points(client: QdrantClient, collection: str) -> int:
    """Number of points currently in ``collection``."""
    return int(client.count(collection_name=collection, exact=True).count)


def scroll_point_ids(
    client: QdrantClient, collection: str, *, batch_size: int = 256
) -> list[str]:
    """Every point id in ``collection`` (paged, so large collections are safe)."""
    ids: list[str] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(str(point.id) for point in points)
        if offset is None or not points:
            break
    return ids


def make_points(vectors: Sequence[tuple[str, Sequence[float], dict[str, Any]]]) -> list[models.PointStruct]:
    """Wrap ``(id, vector, payload)`` triples as Qdrant points."""
    return [
        models.PointStruct(id=point_id, vector=list(vector), payload=payload)
        for point_id, vector, payload in vectors
    ]


def upsert_points(
    client: QdrantClient,
    collection: str,
    points: Sequence[models.PointStruct],
    *,
    batch_size: int = 128,
) -> int:
    """Upload ``points`` in batches, returning how many were written."""
    written = 0
    for start in range(0, len(points), batch_size):
        batch = list(points[start : start + batch_size])
        if not batch:
            continue
        client.upsert(collection_name=collection, points=batch, wait=True)
        written += len(batch)
    return written


def delete_points(
    client: QdrantClient,
    collection: str,
    point_ids: Sequence[str],
    *,
    batch_size: int = 256,
) -> int:
    """Delete points by id, returning how many were requested for deletion."""
    deleted = 0
    for start in range(0, len(point_ids), batch_size):
        batch = list(point_ids[start : start + batch_size])
        if not batch:
            continue
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=batch),
            wait=True,
        )
        deleted += len(batch)
    return deleted


def fetch_payloads(
    client: QdrantClient, collection: str, point_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Fetch payloads for ``point_ids`` (used by the inspection CLI)."""
    records = client.retrieve(
        collection_name=collection,
        ids=list(point_ids),
        with_payload=True,
        with_vectors=False,
    )
    return [dict(record.payload or {}) for record in records]


def sample_payloads(
    client: QdrantClient,
    collection: str,
    *,
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[dict[str, Any]]:
    """Fetch up to ``limit`` payloads, optionally filtered."""
    points, _offset = client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=query_filter,
    )
    return [dict(point.payload or {}) for point in points]


def payload_field_names(collection_payload: dict[str, Any]) -> list[str]:
    """Payload fields present in a point, restricted to the documented set."""
    return [field for field in INDEXED_PAYLOAD_FIELDS if field in collection_payload]


def qdrant_target(url: str | None, path: str | Path | None) -> str:
    """Describe the Qdrant target for reports without opening a connection."""
    if path:
        return f"local path {Path(path)}"
    if url:
        return url if "://" in url else f"http://{url}"
    return "unconfigured"
