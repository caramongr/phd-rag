"""RAG-009 checks 1-3 and 15: query handling and query embedding."""

from __future__ import annotations

import pytest

from app.retrieval.query import normalize_query
from app.retrieval.dense import DenseRetriever, RetrievalError
from app.retrieval.service import RetrievalService
from tests.retrieval_support import QueryEmbedder, TEST_COLLECTION, corpus_points, memory_collection


def test_query_is_trimmed_and_whitespace_collapsed() -> None:
    """Check 2: normalized query text."""
    assert normalize_query("  adaptive   learning \n in higher\teducation ") == (
        "adaptive learning in higher education"
    )
    assert normalize_query("\u00a0virtual\u00a0reality\u00a0") == "virtual reality"


def test_query_normalization_is_unicode_aware_but_case_preserving() -> None:
    """Check 2: Unicode is normalized, case is left alone."""
    assert normalize_query("ＡＩ literacy") == "AI literacy"
    assert normalize_query("Artiﬁcial Intelligence") == "Artificial Intelligence"
    assert normalize_query("ChatGPT in HE") == "ChatGPT in HE"


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_query(None)  # type: ignore[arg-type]


def test_service_rejects_a_query_that_normalizes_to_nothing() -> None:
    client = memory_collection(corpus_points())
    service = RetrievalService(embedder=QueryEmbedder(), client=client, collection=TEST_COLLECTION)
    with pytest.raises(ValueError):
        service.retrieve("   ")
    service.close()


def test_query_embedding_uses_the_rag008_wrapper() -> None:
    """Check 1: the existing BGE-M3 wrapper is reused, not re-implemented."""
    client = memory_collection(corpus_points())
    embedder = QueryEmbedder()
    service = RetrievalService(embedder=embedder, client=client, collection=TEST_COLLECTION)
    response = service.retrieve("virtual reality in higher education")
    assert embedder.queries == ["virtual reality in higher education"]
    assert response.audit.embedding_model == "fake/bge-m3"
    assert response.audit.embedding_revision == "test-rev"
    service.close()


def test_query_embedding_dimension_and_normalization_are_verified() -> None:
    """Check 3: vector dimension, plus the normalization the cosine score assumes."""
    points = corpus_points()
    client = memory_collection(points)
    good = QueryEmbedder()
    service = RetrievalService(embedder=good, client=client, collection=TEST_COLLECTION)
    assert service.retriever.dimension() == good.dimension
    service.close()

    wrong_dimension = QueryEmbedder(query_vector=[1.0, 0.0, 0.0])
    service = RetrievalService(embedder=wrong_dimension, client=client, collection=TEST_COLLECTION)
    with pytest.raises(RetrievalError):
        service.retrieve("dimension mismatch")
    service.close()

    unnormalized = QueryEmbedder(query_vector=[3.0] + [0.0] * (good.dimension - 1))
    service = RetrievalService(embedder=unnormalized, client=client, collection=TEST_COLLECTION)
    with pytest.raises(RetrievalError):
        service.retrieve("unnormalized query")
    service.close()


def test_collection_missing_is_reported_clearly() -> None:
    client = memory_collection(corpus_points())
    retriever = DenseRetriever(
        collection="does-not-exist", client=client, embedder=QueryEmbedder()
    )
    with pytest.raises(Exception) as error:
        retriever.dimension()
    assert "does-not-exist" in str(error.value)
