"""Combined local + external discovery (RAG-010 §30 items 22-24, §21, §28).

The property under test: local evidence and external discovery stay in two separate
result sets, and an Internet outage never damages local retrieval.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.discovery.models import LocalMatchKind
from app.discovery.render import format_combined
from app.discovery.research import research_discover
from app.retrieval.models import Candidate, RetrievalMode
from tests.discovery_support import (
    FakeResponse,
    QueuedTransport,
    RoutingTransport,
    build_service,
    crossref_message,
    crossref_search_payload,
    local_payload,
    make_local_index,
)


class _Response:
    """The only part of a RAG-009 response the combined service reads."""

    def __init__(self, results: Sequence[Candidate]) -> None:
        self.results = list(results)


class FakeRetrievalService:
    """A RAG-009 stand-in returning fixed candidates (and optionally failing)."""

    def __init__(
        self,
        candidates: Sequence[Candidate] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.collection = "phd_rag_chunks_v1"
        self.candidate_k = 25
        self._candidates = candidates
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> _Response:
        self.calls.append({"query": query, **kwargs})
        if self._error is not None:
            raise self._error
        return _Response(self._candidates)

    def close(self) -> None:
        pass


def _candidate(
    chunk_id: str = "c1",
    *,
    title: str = "A Local Paper",
    document_id: str = "doc-1",
) -> Candidate:
    """A local retrieval candidate."""
    return Candidate(
        chunk_id=chunk_id,
        document_id=document_id,
        file_hash="hash-1",
        filename="local.pdf",
        chunk_index=0,
        dense_score=0.8,
        rerank_score=0.9,
        dense_rank=1,
        final_rank=1,
        text="Local passage text.",
        retrieval_text="Local passage text.",
        title=title,
        first_author="Jane Doe",
        year=2021,
        doi="10.1234/local",
        page_start=3,
        page_end=4,
        section_title="Introduction",
        content_type="body",
        corpus_role="literature",
        evidence_eligible=True,
    )


def _external_service(items: list[dict] | None = None, *, failing: bool = False):
    """A discovery service over a fake Crossref."""
    if failing:
        transport: Any = QueuedTransport([FakeResponse(429, {}, text="quota")])
    else:
        transport = RoutingTransport(
            lambda url, params: FakeResponse(
                200, crossref_search_payload(items or [crossref_message(doi="10.1234/ext")])
            )
        )
    return build_service(transport, provider_names=("crossref",))


# --- 22. the two result sets remain separate --------------------------------


def test_local_and_external_results_stay_in_two_separate_sections() -> None:
    """Local evidence and external discoveries are never fused into one list."""
    combined = research_discover(
        "AI governance",
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=_external_service(),
    )

    assert [chunk["chunk_id"] for chunk in combined.local.chunks] == ["c1"]
    assert [record.doi for record in combined.external.records] == ["10.1234/ext"]
    # The external record is not presented as trusted evidence anywhere.
    assert all(not record.is_evidence for record in combined.external.records)
    assert "LOCAL CORPUS" in format_combined(combined)
    assert "NEW EXTERNAL DISCOVERY" in format_combined(combined)


def test_local_half_records_the_retrieval_configuration() -> None:
    """The local section records which collection and mode produced it."""
    combined = research_discover(
        "AI governance",
        local_k=5,
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=_external_service(),
    )

    assert combined.local.collection == "phd_rag_chunks_v1"
    assert combined.local.mode == RetrievalMode.EVIDENCE.value
    assert combined.local.final_k == 5
    assert combined.local.seconds >= 0.0
    assert combined.local.error is None


# --- 23. external failure does not break local retrieval --------------------


def test_external_failure_leaves_local_retrieval_usable() -> None:
    """Every provider down: local results still come back, with a clear failure."""
    combined = research_discover(
        "AI governance",
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=_external_service(failing=True),
    )

    assert len(combined.local.chunks) == 1
    assert combined.local.error is None
    assert combined.external_available is False
    assert combined.external.audit.all_providers_failed is True
    assert combined.external.audit.failures[0].provider == "crossref"
    rendered = format_combined(combined)
    assert "external discovery unavailable" in rendered
    assert "Local passage text." in rendered


def test_local_retrieval_failure_is_reported_without_losing_external_results() -> None:
    """A local failure is recorded as such, and external discovery still returns."""
    combined = research_discover(
        "AI governance",
        retrieval_service=FakeRetrievalService(error=RuntimeError("index unavailable")),
        discovery_service=_external_service(),
    )

    assert combined.local.chunks == ()
    assert combined.local.error is not None
    assert "index unavailable" in combined.local.error
    assert len(combined.external.records) == 1


def test_missing_retrieval_service_is_not_fatal() -> None:
    """A missing local service leaves the external half working."""
    combined = research_discover(
        "AI governance",
        retrieval_service=None,
        discovery_service=_external_service(),
    )

    assert combined.local.chunks == ()
    assert combined.local.error is not None
    assert len(combined.external.records) == 1


# --- 24. already-owned paper is shown as local, not new ---------------------


def test_already_owned_paper_is_shown_as_local_not_new() -> None:
    """A paper already in the corpus is grouped as local, in both halves."""
    index = make_local_index(
        [
            local_payload(
                document_id="doc-1",
                filename="known.pdf",
                doi="10.1234/known",
                title="AI Governance",
            )
        ]
    )
    service = build_service(
        RoutingTransport(
            lambda url, params: FakeResponse(
                200, crossref_search_payload([crossref_message(doi="10.1234/known")])
            )
        ),
        local_index=index,
        provider_names=("crossref",),
    )

    combined = research_discover(
        "ai governance",
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=service,
    )

    record = combined.external.records[0]
    assert record.local_match.kind is LocalMatchKind.ALREADY_IN_CORPUS
    assert combined.external.audit.new_records == 0
    assert combined.external_available is True


def test_new_only_filters_out_already_owned_records() -> None:
    """``new_only`` hides what the corpus already has."""
    index = make_local_index(
        [local_payload(document_id="doc-1", doi="10.1234/known", title="AI Governance")]
    )
    service = build_service(
        RoutingTransport(
            lambda url, params: FakeResponse(
                200,
                crossref_search_payload(
                    [
                        crossref_message(doi="10.1234/known", title="AI Governance"),
                        crossref_message(doi="10.1234/fresh", title="A Brand New Study"),
                    ]
                ),
            )
        ),
        local_index=index,
        provider_names=("crossref",),
    )

    combined = research_discover(
        "ai governance",
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=service,
        new_only=True,
    )

    assert [record.doi for record in combined.external.records] == ["10.1234/fresh"]


def test_year_filter_reaches_the_external_half() -> None:
    """The combined service passes the year window through to discovery."""
    service = _external_service([crossref_message(doi="10.1234/old", year=2015)])

    combined = research_discover(
        "ai",
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=service,
        year_min=2024,
    )

    assert combined.external.records == ()
    assert combined.external.audit.filters["year_min"] == 2024


def test_no_prose_answer_is_generated() -> None:
    """The combined result contains retrieval and discovery data only."""
    combined = research_discover(
        "AI governance",
        retrieval_service=FakeRetrievalService([_candidate()]),
        discovery_service=_external_service(),
    )

    fields = set(combined.model_dump())
    assert fields <= {"query", "local", "external", "timestamp"}
    rendered = format_combined(combined)
    # A generated answer would be prose without evidence markers; the renderer only
    # prints labelled metadata and quoted passage previews.
    assert "Answer:" not in rendered
    assert "LOCAL CORPUS" in rendered
