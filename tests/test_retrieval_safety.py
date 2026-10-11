"""RAG-009 checks 11-13 and 21: evidence-mode safety.

The point of these tests is that non-evidence material cannot be reached in
EVIDENCE mode, no matter how well it matches the query - including the real
corpus case of ``second-paper-draft-final.docx``, which exists under both
``literature`` and ``my_work`` but resolves to ``my_work``.
"""

from __future__ import annotations

import pytest

from app.retrieval.filters import (
    REASON_INELIGIBLE,
    REASON_MODE,
    REASON_REFERENCES,
    is_allowed,
    rejection_reason,
)
from app.retrieval.models import RetrievalFilters, RetrievalMode
from app.retrieval.service import RetrievalService
from tests.retrieval_support import TEST_COLLECTION, QueryEmbedder, corpus_points, memory_collection


@pytest.fixture()
def service():
    client = memory_collection(corpus_points())
    service = RetrievalService(
        embedder=QueryEmbedder(), client=client, collection=TEST_COLLECTION, final_k=10
    )
    yield service
    service.close()


NON_EVIDENCE_ROLES = ("my_work", "summaries", "supervision", "rules", "guidance", "future_chapters")


def test_evidence_mode_never_returns_a_non_evidence_role(service) -> None:
    """Checks 4/11/12: the requirement is corpus-wide, not per query."""
    for query in (
        "artificial intelligence",
        "my own draft",
        "virtual reality",
        "historical development of AI",
        "referencing rules",
        "supervisor methodology",
        "future chapters guidance",
        "my summaries",
    ):
        response = service.retrieve(query, mode=RetrievalMode.EVIDENCE, final_k=10)
        leaked = [candidate for candidate in response.results if not candidate.evidence_eligible]
        assert leaked == [], f"query {query!r} leaked {[c.filename for c in leaked]}"
        assert all(
            candidate.corpus_role not in NON_EVIDENCE_ROLES for candidate in response.results
        )


def test_second_paper_draft_final_can_never_appear_in_evidence_mode(service) -> None:
    """Check 11: the RAG-007 duplicate-role case."""
    for query in ("second paper draft", "my own unpublished draft", "my draft claims"):
        response = service.retrieve(query, mode=RetrievalMode.EVIDENCE, final_k=10)
        assert all(
            candidate.filename != "second-paper-draft-final.docx"
            for candidate in response.results
        ), f"query {query!r} returned the author's own draft as evidence"

    my_work = service.retrieve("second paper draft", mode=RetrievalMode.MY_WORK, final_k=10)
    assert {candidate.filename for candidate in my_work.results} == {"second-paper-draft-final.docx"}


def test_future_chapters_are_excluded_from_evidence_mode(service) -> None:
    """Check 12."""
    response = service.retrieve(
        "Chapter 3 historical development of AI", mode=RetrievalMode.EVIDENCE, final_k=10
    )
    assert all(candidate.corpus_role != "future_chapters" for candidate in response.results)
    assert all(candidate.evidence_eligible for candidate in response.results)


def test_reference_points_are_defensively_excluded(service) -> None:
    """Check 13: even if a collection contained reference chunks, they are refused."""
    response = service.retrieve(
        "Adams reference entry", mode=RetrievalMode.GENERAL, final_k=10, candidate_k=25
    )
    assert all(candidate.chunk_id != "ref-1" for candidate in response.results)
    assert all(candidate.content_type != "references" for candidate in response.results)

    reference_payload = {
        "content_type": "references",
        "retrieval_eligible": True,
        "evidence_eligible": True,
        "corpus_role": "literature",
    }
    assert rejection_reason(reference_payload, RetrievalMode.EVIDENCE) == REASON_REFERENCES
    assert is_allowed(reference_payload, RetrievalMode.EVIDENCE) is False


def test_defensive_predicate_rejects_mode_mismatches_and_filters() -> None:
    own = {
        "content_type": "body",
        "retrieval_eligible": True,
        "evidence_eligible": False,
        "corpus_role": "my_work",
        "year": 2025,
    }
    assert rejection_reason(own, RetrievalMode.EVIDENCE) == REASON_MODE
    assert rejection_reason(own, RetrievalMode.MY_WORK) is None
    assert rejection_reason(own, RetrievalMode.MY_WORK, RetrievalFilters(year_max=2020)) == (
        "filter_mismatch"
    )
    ineligible = dict(own, retrieval_eligible=False)
    assert rejection_reason(ineligible, RetrievalMode.MY_WORK) == REASON_INELIGIBLE


def test_qdrant_level_filter_is_the_primary_defence(service) -> None:
    """The mode condition must reach Qdrant, not only the local re-check."""
    calls: list[object] = []

    class Recording:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def query_points(self, **kwargs):
            calls.append(kwargs.get("query_filter"))
            return self._inner.query_points(**kwargs)

    service.retriever._client = Recording(service.retriever._client)
    service.retrieve("anything", mode=RetrievalMode.EVIDENCE, final_k=5)
    assert calls and calls[0] is not None
    conditions = [getattr(condition, "key", None) for condition in calls[0].must]
    assert "evidence_eligible" in conditions


def test_evidence_mode_drop_reasons_are_recorded(service) -> None:
    """A filtered-out candidate is reported, not silently discarded."""
    response = service.retrieve("artificial intelligence higher education", mode=RetrievalMode.EVIDENCE)
    assert response.stats.candidates_retrieved == len(response.dense_candidates) + (
        response.stats.dropped_by_post_filter
    )
    for item in response.dropped:
        if item.reason in (REASON_MODE, REASON_REFERENCES, REASON_INELIGIBLE):
            assert item.chunk_id
