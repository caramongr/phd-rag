"""RAG-009 checks 14-18 and 24: reranking behaviour and score handling."""

from __future__ import annotations

import pytest

from app.retrieval.models import RetrievalMode
from app.retrieval.service import RetrievalService
from tests.retrieval_support import TEST_COLLECTION, FakeReranker, QueryEmbedder, corpus_points, memory_collection


@pytest.fixture()
def client():
    return memory_collection(corpus_points())


def build(client, reranker=None, **kwargs):
    return RetrievalService(
        embedder=QueryEmbedder(),
        reranker=reranker,
        client=client,
        collection=TEST_COLLECTION,
        final_k=5,
        **kwargs,
    )


def test_reranker_preserves_candidate_identity_and_sees_retrieval_text(client) -> None:
    """Check 14 (+ §8: the reranker consumes ``retrieval_text``)."""
    reranker = FakeReranker(score=0.5)
    service = build(client, reranker)
    response = service.retrieve("artificial intelligence higher education", mode=RetrievalMode.GENERAL)

    assert len(reranker.calls) == 1
    query, chunk_ids = reranker.calls[0]
    assert query == "artificial intelligence higher education"
    assert chunk_ids == [candidate.chunk_id for candidate in response.dense_candidates]
    assert set(reranker.seen_texts) == set(chunk_ids)
    for candidate in response.dense_candidates:
        assert reranker.seen_texts[candidate.chunk_id] == candidate.retrieval_text
    assert {candidate.chunk_id for candidate in response.results} <= set(chunk_ids)
    service.close()


def test_dense_score_is_preserved_and_rerank_score_stored_separately(client) -> None:
    """Checks 15/16: two scores, two scales, no fusion."""
    reranker = FakeReranker(score=0.25)
    service = build(client, reranker)
    response = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL)
    for candidate in response.results:
        assert candidate.dense_score is not None
        assert candidate.rerank_score == pytest.approx(0.25)
        assert candidate.dense_score != candidate.rerank_score
    dense_by_id = {c.chunk_id: c.dense_score for c in response.dense_candidates}
    for candidate in response.results:
        assert candidate.dense_score == pytest.approx(dense_by_id[candidate.chunk_id])
    service.close()


def test_output_is_sorted_by_rerank_score(client) -> None:
    """Check 17: reranking decides the order, including against the dense order."""
    scores = {"lit-a1": 0.1, "lit-b1": 0.9, "own-1": 0.5}

    def score_fn(_query: str, chunk_id: str, _text: str) -> float:
        return scores.get(chunk_id, 0.01)

    reranker = FakeReranker(score_fn=score_fn)
    service = build(client, reranker)
    response = service.retrieve("anything", mode=RetrievalMode.GENERAL)
    reranked = [candidate.rerank_score for candidate in response.results]
    assert reranked == sorted(reranked, reverse=True)
    assert response.results[0].chunk_id == "lit-b1"
    assert response.dense_candidates[0].chunk_id == "lit-a1"
    assert [candidate.final_rank for candidate in response.results] == list(
        range(1, len(response.results) + 1)
    )
    service.close()


def test_dense_order_is_used_when_reranking_is_disabled(client) -> None:
    """Check 18: the ``--no-rerank`` path keeps dense order and unchanged scores."""
    reranker = FakeReranker(score=0.99)
    service = build(client, reranker)
    response = service.retrieve(
        "artificial intelligence", mode=RetrievalMode.GENERAL, rerank=False
    )
    assert reranker.calls == []
    assert all(candidate.rerank_score is None for candidate in response.results)
    dense_scores = [candidate.dense_score for candidate in response.results]
    assert dense_scores == sorted(dense_scores, reverse=True)
    assert response.audit.rerank_enabled is False
    service.close()


def test_missing_reranker_falls_back_to_dense_order(client) -> None:
    service = build(client, None)
    response = service.retrieve("anything", mode=RetrievalMode.GENERAL, rerank=True)
    assert response.audit.rerank_enabled is False
    assert all(candidate.rerank_score is None for candidate in response.results)
    service.close()


def test_minimum_score_filters_are_off_by_default_and_optional(client) -> None:
    """§14: no invented thresholds, but a caller may set one."""
    service = build(client, FakeReranker(score=0.4))
    default_response = service.retrieve("anything", mode=RetrievalMode.GENERAL)
    assert default_response.audit.min_dense_score is None
    assert default_response.audit.min_rerank_score is None
    assert default_response.results

    filtered = service.retrieve(
        "anything", mode=RetrievalMode.GENERAL, min_rerank_score=0.9
    )
    assert filtered.results == []
    assert any(item.reason == "min_rerank_score" for item in filtered.dropped)

    dense_filtered = service.retrieve(
        "anything", mode=RetrievalMode.GENERAL, min_dense_score=0.99
    )
    assert all(
        candidate.dense_score >= 0.99 for candidate in dense_filtered.results
    )
    service.close()


def test_audit_records_the_reranker_and_the_configuration(client) -> None:
    """Check 24: the audit record reproduces the configuration."""
    reranker = FakeReranker(device="cpu", batch_size=4)
    service = build(client, reranker)
    response = service.retrieve(
        "artificial intelligence", mode=RetrievalMode.EVIDENCE, candidate_k=7, final_k=3
    )
    audit = response.audit
    assert audit.raw_query == "artificial intelligence"
    assert audit.normalized_query == "artificial intelligence"
    assert audit.retrieval_mode is RetrievalMode.EVIDENCE
    assert (audit.candidate_k, audit.final_k) == (7, 3)
    assert audit.rerank_enabled is True
    assert audit.rerank_model == "fake/bge-reranker-v2-m3"
    assert audit.rerank_revision == "test-rev"
    assert audit.embedding_model == "fake/bge-m3"
    assert audit.collection == TEST_COLLECTION
    assert audit.timestamp
    assert audit.max_chunks_per_document == 3
    service.close()


def test_reranker_receives_the_normalized_query(client) -> None:
    reranker = FakeReranker(score=0.5)
    service = build(client, reranker)
    service.retrieve("  artificial   intelligence \n in higher education ", mode=RetrievalMode.GENERAL)
    assert reranker.calls[0][0] == "artificial intelligence in higher education"
    service.close()


def test_gpu_reranker_wrapper_contract_without_torch() -> None:
    """The BGE wrapper scores pairs in order and keeps identity (no model download)."""
    from app.reranking.bge_reranker import BgeReranker
    from app.reranking.models import RerankInput

    class Backend:
        max_length = 8192

        def __init__(self):
            self.seen: list[tuple[str, str]] = []

        def predict(self, pairs, batch_size=8, show_progress_bar=False, convert_to_numpy=True):
            import numpy as np

            self.seen = list(pairs)
            return np.asarray([[0.1], [0.9]], dtype="float32")

    backend = Backend()
    reranker = BgeReranker(device="cpu", batch_size=2, loader=lambda *a, **k: backend)
    scores = reranker.rerank(
        "query",
        [RerankInput(chunk_id="a", text="first"), RerankInput(chunk_id="b", text="second")],
    )
    assert [item.chunk_id for item in scores] == ["a", "b"]
    assert [item.score for item in scores] == pytest.approx([0.1, 0.9])
    assert backend.seen == [("query", "first"), ("query", "second")]
    assert reranker.spec.max_length == 8192
    assert reranker.spec.batch_size == 2
    assert reranker.count_truncated([("query", "first")]) == 0
