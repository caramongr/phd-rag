"""RAG-008 tests 1-5: the embedding wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from embedding_support import FakeSentenceTransformer

from app.embedding.bge_m3 import BgeM3Embedder, resolve_device
from app.embedding.cache import unit_norm
from app.embedding.models import EmbeddingFailure, EmbeddingModelLoadError


def _embedder(tmp_path: Path, **kwargs) -> BgeM3Embedder:
    """A BGE-M3 wrapper whose backend is a deterministic fake sentence-transformer."""
    backend = kwargs.pop("backend", None) or FakeSentenceTransformer(dimension=6)
    return BgeM3Embedder(
        model_name=str(tmp_path),
        device="cpu",
        model=backend,
        **kwargs,
    )


def test_deterministic_vector_for_mocked_model(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path)

    first = embedder.embed_documents(["a deterministic sentence"])
    second = embedder.embed_documents(["a deterministic sentence"])

    assert first == second
    assert len(first[0]) == 6


def test_dimension_exposed_correctly(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path, backend=FakeSentenceTransformer(dimension=11))

    assert embedder.dimension == 11
    assert embedder.spec.dimension == 11
    assert len(embedder.embed_query("dimension check")) == 11


def test_document_batch_embedding_preserves_order(tmp_path: Path) -> None:
    texts = [f"sentence number {index}" for index in range(5)]
    embedder = _embedder(tmp_path)

    batch = embedder.embed_documents(texts, batch_size=2)
    singles = [embedder.embed_query(text) for text in texts]

    assert batch == singles


def test_normalized_vector_is_approximately_unit_norm(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path)

    vectors = embedder.embed_documents(["normalization check", "second sample"])

    for vector in vectors:
        assert unit_norm(vector) == pytest.approx(1.0, abs=1e-6)
    assert embedder.verify_normalization() == pytest.approx(1.0, abs=1e-6)
    assert embedder.spec.normalized is True


def test_model_load_failure_is_explicit(tmp_path: Path) -> None:
    def failing_loader(_path: str, _device: str, **_kwargs):
        raise RuntimeError("no weights here")

    embedder = BgeM3Embedder(model_name=str(tmp_path), device="cpu", loader=failing_loader)

    with pytest.raises(EmbeddingModelLoadError) as error:
        embedder.spec

    assert "could not load embedding model" in str(error.value)
    assert "no weights here" in str(error.value)


def test_embedding_backend_failure_is_wrapped(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path, backend=FakeSentenceTransformer(fail=True))

    with pytest.raises(EmbeddingFailure):
        embedder.embed_documents(["boom"])


def test_explicit_unavailable_device_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device(None) == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(EmbeddingModelLoadError):
        resolve_device("cuda")


def test_tokenizer_counts_without_truncation(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path)

    assert embedder.count_tokens("four word sample text") == 5
    assert embedder.max_tokens == 8192
    assert embedder.spec.max_tokens == 8192
