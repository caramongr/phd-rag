"""RAG-009 §2/§11: widening the search when duplicate copies starve the results.

The real corpus contains several physical copies of some published papers. A single
stage of ``candidate_k`` can then consist of nothing but copies of one or two works,
and the pipeline would return fewer than ``final_k`` chunks. These tests pin the
bounded top-up behaviour that fixes it, and the fact that a healthy query never
pays for an extra round.
"""

from __future__ import annotations

import pytest

from app.retrieval.models import RetrievalMode
from app.retrieval.service import RetrievalService
from tests.retrieval_support import (
    TEST_COLLECTION,
    FakeReranker,
    QueryEmbedder,
    memory_collection,
    point,
)


def saturated_corpus(copies: int = 6) -> list[dict]:
    """One work in many copies (identical passages) plus a few unrelated works.

    Angles place every copy of the work ahead of the other documents, so the first
    search round sees only duplicates.
    """
    points: list[dict] = []
    for copy in range(copies):
        for index in range(2):
            points.append(
                point(
                    f"copy{copy}-chunk{index}",
                    angle=2 + copy * 0.2 + index * 0.1,
                    document_id=f"doc-copy-{copy}",
                    filename=f"gregor-2013-copy{copy}.pdf",
                    title="Positioning and Presenting Design Science Research for Maximum Impact",
                    chunk_index=index,
                    text=f"design science research passage {index} (copy {copy})",
                )
            )
    for index, angle in enumerate((40.0, 45.0, 50.0, 55.0, 60.0)):
        points.append(
            point(
                f"other-{index}",
                angle=angle,
                document_id=f"doc-other-{index}",
                filename=f"other-{index}.pdf",
                title=f"Unrelated work number {index} on assessment design",
                text=f"assessment design passage {index}",
            )
        )
    return points


def build(client, **kwargs) -> RetrievalService:
    return RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(score_fn=lambda _q, cid, _t: 1.0 if cid.startswith("copy") else 0.5),
        client=client,
        collection=TEST_COLLECTION,
        candidate_k=5,
        final_k=6,
        **kwargs,
    )


def test_search_widens_until_final_k_is_reachable() -> None:
    """A saturated first round triggers additional rounds instead of a short list."""
    service = build(memory_collection(saturated_corpus()))
    response = service.retrieve("design science research", mode=RetrievalMode.GENERAL)
    assert response.stats.candidate_rounds > 1
    assert response.stats.candidates_retrieved > 5
    assert len(response.results) == 6
    assert response.audit.max_candidate_rounds == 3
    # the work cap still holds: at most 3 chunks of the duplicated work
    capped = [c for c in response.results if c.document_id.startswith("doc-copy")]
    assert len(capped) <= 3
    assert any(c.document_id.startswith("doc-other") for c in response.results)
    service.close()


def test_widening_stops_at_the_configured_cap() -> None:
    service = build(memory_collection(saturated_corpus(copies=20)), max_candidate_rounds=1)
    response = service.retrieve("design science research", mode=RetrievalMode.GENERAL)
    assert response.stats.candidate_rounds == 1
    assert response.stats.candidates_retrieved == 5
    assert len(response.results) <= 5
    service.close()


def test_no_extra_round_when_the_first_round_is_enough() -> None:
    """A normal query must not pay for a second search or rerank."""
    calls: list[int] = []

    class Counting:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def query_points(self, **kwargs):
            calls.append(kwargs["limit"])
            return self._inner.query_points(**kwargs)

    client = memory_collection(saturated_corpus(copies=1))
    service = RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(score_fn=lambda _q, cid, _t: 1.0 if cid.startswith("copy") else 0.5),
        client=client,
        collection=TEST_COLLECTION,
        candidate_k=5,
        final_k=4,
    )
    service.retriever._client = Counting(service.retriever._client)
    response = service.retrieve("design science research", mode=RetrievalMode.GENERAL)
    assert calls == [5]
    assert response.stats.candidate_rounds == 1
    assert len(response.results) == 4
    service.close()


def test_candidates_are_never_scored_twice() -> None:
    """The pool is merged by chunk id, so extra rounds do not duplicate work."""
    client = memory_collection(saturated_corpus())
    service = build(client)
    response = service.retrieve("design science research", mode=RetrievalMode.GENERAL)
    ids = [candidate.chunk_id for candidate in response.dense_candidates]
    assert len(ids) == len(set(ids))
    pool_ids = [item.chunk_id for item in response.dropped if item.reason == "beyond_final_k"]
    assert len(pool_ids) == len(set(pool_ids))
    service.close()


@pytest.mark.parametrize("order", [0, 1])
def test_widening_is_deterministic(order: int) -> None:
    points = saturated_corpus()
    client = memory_collection(points[::-1] if order else points)
    service = build(client)
    response = service.retrieve("design science research", mode=RetrievalMode.GENERAL)
    assert response.stats.candidate_rounds > 1
    assert len(response.results) == 6
    service.close()
