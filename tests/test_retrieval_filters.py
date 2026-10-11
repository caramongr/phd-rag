"""RAG-009 checks 4-10: retrieval modes and explicit filters."""

from __future__ import annotations

import pytest
from qdrant_client import models

from app.retrieval.filters import build_qdrant_filter
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


def roles(response) -> set[str | None]:
    return {candidate.corpus_role for candidate in response.results}


def ids(response) -> list[str]:
    return response.chunk_ids


def test_evidence_mode_returns_only_evidence_eligible_points(service) -> None:
    """Check 4."""
    response = service.retrieve("artificial intelligence higher education", mode=RetrievalMode.EVIDENCE)
    assert response.results
    assert all(candidate.evidence_eligible for candidate in response.results)
    assert roles(response) == {"literature"}


def test_my_work_mode_returns_only_my_work(service) -> None:
    """Check 5."""
    response = service.retrieve("my own draft", mode=RetrievalMode.MY_WORK)
    assert response.results
    assert roles(response) == {"my_work"}
    assert all(candidate.evidence_eligible is False for candidate in response.results)


def test_supervision_mode_returns_only_supervision(service) -> None:
    """Check 6."""
    response = service.retrieve("supervisor methodology comments", mode=RetrievalMode.SUPERVISION)
    assert response.results
    assert roles(response) == {"supervision"}


def test_guidance_mode_returns_expected_non_evidence_roles(service) -> None:
    """Check 7."""
    response = service.retrieve("guidance material", mode=RetrievalMode.GUIDANCE, final_k=10)
    assert response.results
    assert roles(response) <= {"rules", "guidance", "future_chapters", "summaries"}
    assert all(candidate.evidence_eligible is False for candidate in response.results)


def test_general_mode_can_see_every_role_and_preserves_it(service) -> None:
    response = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL, final_k=10)
    assert "literature" in roles(response)
    assert all(candidate.corpus_role for candidate in response.results)


def test_year_range_filter_works(service) -> None:
    """Check 8."""
    response = service.retrieve(
        "anything",
        mode=RetrievalMode.EVIDENCE,
        filters=RetrievalFilters(year_min=2020, year_max=2025),
        final_k=10,
    )
    assert response.results
    assert all(2020 <= candidate.year <= 2025 for candidate in response.results)

    empty = service.retrieve(
        "anything",
        mode=RetrievalMode.EVIDENCE,
        filters=RetrievalFilters(year_min=1990, year_max=1995),
    )
    assert empty.results == []
    assert empty.stats.candidates_retrieved == 0


def test_content_type_filter_works(service) -> None:
    """Check 9."""
    response = service.retrieve(
        "anything",
        mode=RetrievalMode.EVIDENCE,
        filters=RetrievalFilters(content_types=("abstract",)),
        final_k=10,
    )
    assert response.results == []
    assert RetrievalFilters(content_types=("body",)) is not None
    with pytest.raises(ValueError):
        RetrievalFilters(content_types=("nonsense",))


def test_document_id_and_file_hash_filters_work(service) -> None:
    """Check 10."""
    by_document = service.retrieve(
        "anything",
        mode=RetrievalMode.EVIDENCE,
        filters=RetrievalFilters(document_id="doc-lit-b"),
        final_k=10,
    )
    assert {candidate.document_id for candidate in by_document.results} == {"doc-lit-b"}

    by_hash = service.retrieve(
        "anything",
        mode=RetrievalMode.EVIDENCE,
        filters=RetrievalFilters(file_hash="hash-lit-b1"),
        final_k=10,
    )
    assert [candidate.chunk_id for candidate in by_hash.results] == ["lit-b1"]


def test_ocr_and_doi_and_publication_type_filters_are_available(service) -> None:
    response = service.retrieve(
        "anything",
        mode=RetrievalMode.EVIDENCE,
        filters=RetrievalFilters(ocr_involved=False, publication_type="journal-article"),
        final_k=10,
    )
    assert response.results
    assert all(candidate.ocr_involved is False for candidate in response.results)

    none_by_doi = service.retrieve(
        "anything", mode=RetrievalMode.EVIDENCE, filters=RetrievalFilters(doi="10.0/nothing")
    )
    assert none_by_doi.results == []


def test_filters_reject_impossible_year_ranges() -> None:
    with pytest.raises(ValueError):
        RetrievalFilters(year_min=2025, year_max=2000)


def test_filter_construction_is_explicit_and_empty_by_default() -> None:
    assert build_qdrant_filter(RetrievalMode.GENERAL, RetrievalFilters()) is not None
    assert RetrievalFilters().is_empty
    assert RetrievalFilters(year_min=2000).describe() == {"year_min": 2000}


def test_filter_conditions_encode_the_mode() -> None:
    evidence = build_qdrant_filter(RetrievalMode.EVIDENCE, RetrievalFilters())
    assert evidence is not None
    keys = [condition.key for condition in evidence.must if isinstance(condition, models.FieldCondition)]
    assert "evidence_eligible" in keys
    assert "retrieval_eligible" in keys
    assert evidence.must_not, "references must be excluded explicitly"

    guidance = build_qdrant_filter(RetrievalMode.GUIDANCE, RetrievalFilters())
    assert guidance is not None
    role_condition = [
        condition
        for condition in guidance.must
        if isinstance(condition, models.FieldCondition) and condition.key == "corpus_role"
    ][0]
    assert set(role_condition.match.any) == {
        "rules",
        "guidance",
        "future_chapters",
        "summaries",
    }
