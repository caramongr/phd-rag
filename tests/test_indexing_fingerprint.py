"""RAG-008 tests 25-27: deterministic corpus fingerprint and the index manifest."""

from __future__ import annotations

from pathlib import Path

from app.chunking.normalize import NORMALIZATION_VERSION
from app.indexing.fingerprint import (
    chunk_ids_digest,
    corpus_fingerprint,
    document_hashes_digest,
)
from app.indexing.manifest import IndexManifest, manifest_path, read_manifest, write_manifest

_HASHES = ["a" * 64, "b" * 64, "c" * 64]
_CHUNKS = ["11111111-1111-5111-8111-111111111111", "22222222-2222-5222-8222-222222222222"]


def _fingerprint(hashes=_HASHES, ids=_CHUNKS, version=NORMALIZATION_VERSION) -> str:
    return corpus_fingerprint(file_hashes=hashes, chunk_ids=ids, chunking_version=version)


def test_same_chunk_set_gives_same_fingerprint() -> None:
    assert _fingerprint() == _fingerprint()


def test_ordering_does_not_change_fingerprint() -> None:
    assert _fingerprint() == _fingerprint(hashes=list(reversed(_HASHES)), ids=list(reversed(_CHUNKS)))
    assert document_hashes_digest(["b", "a"]) == document_hashes_digest(["a", "b"])
    assert chunk_ids_digest(["2", "1"]) == chunk_ids_digest(["1", "2"])


def test_chunking_version_changes_fingerprint() -> None:
    assert _fingerprint() != _fingerprint(version="rag007-normalize-v0")
    assert _fingerprint() != _fingerprint(hashes=["d" * 64])
    assert _fingerprint() != _fingerprint(ids=["33333333-3333-5333-8333-333333333333"])


def test_timestamp_does_not_affect_fingerprint(tmp_path: Path) -> None:
    first = IndexManifest(
        collection="c",
        qdrant_target="local",
        distance="Cosine",
        vector_dimension=1024,
        embedding_model="BAAI/bge-m3",
        embedding_identifier="id",
        embedding_library="sentence-transformers",
        normalization="l2",
        device="cpu",
        chunking_version=NORMALIZATION_VERSION,
        corpus_fingerprint=_fingerprint(),
        document_hashes_digest=document_hashes_digest(_HASHES),
        chunk_ids_digest=chunk_ids_digest(_CHUNKS),
        build_timestamp="2026-01-01T00:00:00+00:00",
    )
    second = first.model_copy(update={"build_timestamp": "2026-06-30T12:00:00+00:00"})

    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.model_dump()["corpus_fingerprint"] == second.model_dump()["corpus_fingerprint"]


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = manifest_path(tmp_path / "manifests", "phd_rag_chunks_v1")
    manifest = IndexManifest(
        collection="phd_rag_chunks_v1",
        qdrant_target="local path .qdrant_data",
        distance="Cosine",
        vector_dimension=1024,
        embedding_model="BAAI/bge-m3",
        embedding_revision="5617a9f61b028005a4858fdac845db406aefb181",
        embedding_identifier="BAAI/bge-m3|rev=abc|dim=1024|norm=1|lib=sentence-transformers",
        embedding_library="sentence-transformers",
        embedding_library_version="6.1.0",
        normalization="l2 (unit-norm, cosine-ready)",
        device="cpu",
        chunking_version=NORMALIZATION_VERSION,
        chunk_count=14786,
        indexed_chunk_count=100,
        document_count=96,
        corpus_fingerprint=_fingerprint(),
        document_hashes_digest=document_hashes_digest(_HASHES),
        chunk_ids_digest=chunk_ids_digest(_CHUNKS),
        points_total=100,
    )

    written = write_manifest(path, manifest)

    assert written.name == "phd_rag_chunks_v1.index-manifest.json"
    restored = read_manifest(written)
    assert restored.vector_dimension == 1024
    assert restored.embedding_revision == manifest.embedding_revision
    assert restored.normalization == manifest.normalization
    assert restored.points_total == 100
