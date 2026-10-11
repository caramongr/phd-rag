"""RAG-008 tests 15-18: collection schema, dimension safety and payload indexes."""

from __future__ import annotations

from pathlib import Path

import pytest
from embedding_support import RecordingClient, local_client
from qdrant_client import models

from app.indexing.collection import (
    DEFAULT_PAYLOAD_INDEXES,
    CollectionDimensionMismatch,
    QdrantConfigurationError,
    build_client,
    count_points,
    ensure_collection,
    existing_dimension,
    make_points,
    upsert_points,
    validate_collection_name,
)


def test_cosine_collection_created_with_embedding_dimension(tmp_path: Path) -> None:
    client = local_client(tmp_path)

    setup = ensure_collection(client, collection="cosine", dimension=1024)

    assert setup.created is True
    assert setup.dimension == 1024
    assert setup.distance == "Cosine"
    info = client.get_collection("cosine")
    assert info.config.params.vectors.size == 1024
    assert info.config.params.vectors.distance == models.Distance.COSINE


def test_dimension_mismatch_on_existing_collection_raises(tmp_path: Path) -> None:
    client = local_client(tmp_path)
    ensure_collection(client, collection="mismatch", dimension=8)

    with pytest.raises(CollectionDimensionMismatch) as error:
        ensure_collection(client, collection="mismatch", dimension=1024)

    assert "refusing to replace a populated collection" in str(error.value)


def test_populated_incompatible_collection_is_not_destructively_recreated(
    tmp_path: Path,
) -> None:
    client = local_client(tmp_path)
    ensure_collection(client, collection="populated", dimension=4)
    upsert_points(
        client,
        "populated",
        make_points([("11111111-1111-5111-8111-111111111111", [0.1, 0.2, 0.3, 0.4], {"a": 1})]),
    )
    assert count_points(client, "populated") == 1

    with pytest.raises(CollectionDimensionMismatch):
        ensure_collection(client, collection="populated", dimension=8)

    assert count_points(client, "populated") == 1
    assert existing_dimension(client, "populated") == 4


def test_required_payload_indexes_created(tmp_path: Path) -> None:
    client = RecordingClient(local_client(tmp_path))

    setup = ensure_collection(client, collection="indexed", dimension=8)

    created = {field for _collection, field in client.payload_index_calls}
    assert created == {field for field, _schema in DEFAULT_PAYLOAD_INDEXES}
    assert set(setup.payload_indexes) == created
    assert {"corpus_role", "evidence_eligible", "content_type", "year", "document_id", "file_hash"} <= created


def test_reusing_a_matching_collection_does_not_recreate_it(tmp_path: Path) -> None:
    client = local_client(tmp_path)
    first = ensure_collection(client, collection="reuse", dimension=8)
    second = ensure_collection(client, collection="reuse", dimension=8)

    assert first.created is True
    assert second.created is False


def test_collection_name_validation() -> None:
    assert validate_collection_name("phd_rag_chunks_v1") == "phd_rag_chunks_v1"
    with pytest.raises(QdrantConfigurationError):
        validate_collection_name("bad name!")


def test_client_factory_requires_a_target() -> None:
    with pytest.raises(QdrantConfigurationError):
        build_client()

    client = build_client(url="localhost:6333")
    assert "localhost:6333" in str(getattr(client._client, "rest_uri", ""))
