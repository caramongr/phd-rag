"""Discovery service tests (RAG-010 §30 items 16-18, §27, §28).

These exercise the whole pipeline offline: providers -> merge -> status -> local
comparison -> rank -> audit, plus provider isolation and the no-Internet path.
"""

from __future__ import annotations

from pathlib import Path

from app.discovery.models import DiscoveryStatus, LocalMatchKind
from tests.discovery_support import (
    FakeResponse,
    QueuedTransport,
    RoutingTransport,
    build_service,
    crossref_message,
    crossref_search_payload,
    local_payload,
    make_cache,
    make_local_index,
    openalex_search_payload,
    openalex_work,
)


def _both_providers(
    crossref_items: list[dict], openalex_items: list[dict]
) -> RoutingTransport:
    """A transport answering Crossref and OpenAlex searches separately."""

    def handler(url: str, params: dict | None) -> FakeResponse:
        if "crossref.org" in url:
            return FakeResponse(200, crossref_search_payload(crossref_items))
        if "openalex.org" in url:
            return FakeResponse(200, openalex_search_payload(openalex_items))
        raise AssertionError(f"unexpected url {url}")

    return RoutingTransport(handler)


# --- 16-18. status model -----------------------------------------------------


def test_discovery_only_record_stays_unverified() -> None:
    """One provider, no corroboration: the record is DISCOVERY ONLY."""
    transport = _both_providers([crossref_message()], [])
    result = build_service(transport, provider_names=("crossref",)).discover("ai")

    record = result.records[0]
    assert record.discovery_status is DiscoveryStatus.EXTERNAL_DISCOVERY
    assert record.fulltext_verified is False
    assert record.is_evidence is False


def test_two_providers_agreeing_reach_verified_metadata_status() -> None:
    """DOI + title + authors + year from two providers is verified metadata."""
    transport = _both_providers(
        [
            crossref_message(
                doi="10.1234/abc",
                title="AI Governance",
                authors=[("Jane", "Doe")],
                year=2024,
                venue="Policy Journal",
            )
        ],
        [
            openalex_work(
                doi="10.1234/abc",
                title="AI Governance",
                authors=["Jane Doe"],
                year=2024,
                venue="Policy Journal",
            )
        ],
    )
    result = build_service(transport).discover("ai governance")

    record = result.records[0]
    assert record.discovery_status is DiscoveryStatus.EXTERNAL_METADATA_VERIFIED
    assert record.identity_confidence >= 0.80


def test_fulltext_status_only_after_real_verification() -> None:
    """An open-access URL alone is not 'verified'; a real fetch is required by design."""
    transport = _both_providers([], [openalex_work(is_oa=True, pdf_url="https://example.org/p.pdf")])
    result = build_service(transport).discover("ai")

    record = result.records[0]
    assert record.fulltext_verified is False
    assert record.fulltext_sha256 is None
    assert record.fulltext_url == "https://example.org/p.pdf"


# --- provider isolation ------------------------------------------------------


def test_one_provider_failing_does_not_lose_the_other() -> None:
    """A Crossref 429 still leaves the OpenAlex results usable."""

    def handler(url: str, params: dict | None) -> FakeResponse:
        if "crossref.org" in url:
            return FakeResponse(429, {}, text="rate limited")
        return FakeResponse(200, openalex_search_payload([openalex_work(doi="10.1234/oa")]))

    result = build_service(RoutingTransport(handler)).discover("ai")

    assert result.audit.providers_succeeded == ("openalex",)
    assert result.audit.providers_failed == ("crossref",)
    assert len(result.records) == 1
    assert result.records[0].doi == "10.1234/oa"
    assert result.audit.all_providers_failed is False


def test_all_providers_failing_is_reported_as_such_not_as_no_results() -> None:
    """Every provider down is a provider problem, with an explicit flag."""
    transport = QueuedTransport(
        [FakeResponse(429, {}, text="quota"), FakeResponse(429, {}, text="quota")]
    )
    result = build_service(transport).discover("ai")

    assert result.records == ()
    assert result.audit.all_providers_failed is True
    assert len(result.audit.failures) == 2
    assert result.audit.providers_succeeded == ()


def test_provider_failure_is_reported_with_provider_words() -> None:
    """The failure message keeps what the provider actually said."""
    transport = QueuedTransport(
        [FakeResponse(429, {}, text="daily budget exhausted", headers={"Retry-After": "3600"})]
    )
    result = build_service(transport, provider_names=("crossref",)).discover("ai")

    failure = result.audit.failures[0]
    assert failure.provider == "crossref"
    assert failure.status_code == 429
    assert failure.retry_after == 3600.0
    assert "daily budget exhausted" in failure.message
    assert failure.counted_as_quota is True


# --- §27 audit trail ---------------------------------------------------------


def test_audit_trail_records_the_whole_request() -> None:
    """The audit carries everything the spec requires (§27)."""
    transport = _both_providers([crossref_message(year=2021)], [])
    result = build_service(transport, provider_names=("crossref",)).discover(
        "  AI   Governance ", year_min=2020
    )

    audit = result.audit
    assert audit.raw_query == "  AI   Governance "
    assert audit.normalized_query == "AI Governance"
    assert audit.operation == "search"
    assert audit.providers_requested == ("crossref",)
    assert audit.providers_succeeded == ("crossref",)
    assert audit.filters["year_min"] == 2020
    assert audit.provider_records == 1
    assert audit.merged_records == 1
    assert audit.timestamp
    assert audit.queries[0].provider == "crossref"
    assert audit.queries[0].ok is True
    assert audit.queries[0].http_requests == 1


def test_empty_query_is_refused_before_any_request() -> None:
    """A blank query raises instead of searching for nothing."""
    import pytest

    from app.discovery.service import DiscoveryError

    transport = QueuedTransport([])
    service = build_service(transport)

    with pytest.raises(DiscoveryError):
        service.discover("   ")
    assert transport.call_count == 0


def test_unknown_provider_name_fails_loudly() -> None:
    """A typo in a provider name does not silently search nothing."""
    import pytest

    from app.discovery.service import DiscoveryError

    service = build_service(QueuedTransport([]))

    with pytest.raises(DiscoveryError):
        service.discover("ai", providers=("crossreff",))


# --- local comparison through the service -----------------------------------


def test_local_doi_match_is_marked_already_in_corpus() -> None:
    """A paper the corpus already holds is reported as present, not new."""
    index = make_local_index(
        [
            local_payload(
                document_id="doc-1",
                filename="known.pdf",
                doi="10.1234/abc",
                title="AI Governance",
                corpus_role="literature",
            )
        ]
    )
    transport = _both_providers([crossref_message(doi="10.1234/abc", title="AI Governance")], [])
    result = build_service(transport, local_index=index, provider_names=("crossref",)).discover("ai")

    record = result.records[0]
    assert record.local_match.kind is LocalMatchKind.ALREADY_IN_CORPUS
    assert record.local_match.local_filename == "known.pdf"
    assert result.audit.local_matches == 1
    assert result.audit.new_records == 0


def test_own_draft_doi_match_is_not_reported_as_new_or_evidence() -> None:
    """The author's own draft is neither 'new' nor scholarly evidence."""
    index = make_local_index(
        [
            local_payload(
                document_id="doc-own",
                filename="second-paper-draft-final.docx",
                doi="10.1234/own",
                title="My Own Paper",
                corpus_role="my_work",
                evidence_eligible=False,
            )
        ]
    )
    transport = _both_providers(
        [crossref_message(doi="10.1234/own", title="My Own Paper")], []
    )
    result = build_service(transport, local_index=index, provider_names=("crossref",)).discover("my paper")

    record = result.records[0]
    assert record.local_match.kind is LocalMatchKind.POSSIBLE_MATCH
    assert record.local_match.local_evidence_eligible is False
    assert record.discovery_status is not DiscoveryStatus.LOCAL_MATCH
    assert result.audit.possible_matches == 1


def test_new_paper_is_reported_as_new() -> None:
    """An unknown DOI is new."""
    transport = _both_providers([crossref_message(doi="10.9999/new")], [])
    result = build_service(transport, provider_names=("crossref",)).discover("ai")

    assert result.audit.new_records == 1
    assert result.records[0].local_match.kind is LocalMatchKind.NEW


# --- filters ----------------------------------------------------------------


def test_year_filters_are_rechecked_locally() -> None:
    """A record outside the requested window is dropped even if a provider sent it."""
    transport = _both_providers(
        [
            crossref_message(doi="10.1234/old", year=2015),
            crossref_message(doi="10.1234/new", year=2025),
        ],
        [],
    )
    result = build_service(transport, provider_names=("crossref",)).discover(
        "ai", year_min=2024
    )

    assert [record.doi for record in result.records] == ["10.1234/new"]


def test_author_filter_is_rechecked_locally() -> None:
    """The author filter holds even when a provider ignores it."""
    transport = _both_providers(
        [
            crossref_message(doi="10.1234/a", authors=[("Jane", "Doe")]),
            crossref_message(doi="10.1234/b", authors=[("Peter", "Novak")]),
        ],
        [],
    )
    result = build_service(transport, provider_names=("crossref",)).discover(
        "ai", author="Doe"
    )

    assert [record.doi for record in result.records] == ["10.1234/a"]


# --- lookup operations -------------------------------------------------------


def test_doi_lookup_returns_one_verified_record() -> None:
    """A DOI lookup is a single high-confidence record."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, {"message": crossref_message(doi="10.1234/abc")})
    )
    result = build_service(transport, provider_names=("crossref",)).lookup_doi("10.1234/abc")

    assert result.audit.operation == "doi"
    assert len(result.records) == 1
    assert result.records[0].doi == "10.1234/abc"
    assert result.records[0].identity_confidence >= 0.80
    assert result.records[0].discovery_status is DiscoveryStatus.EXTERNAL_METADATA_VERIFIED


def test_title_lookup_discards_unrelated_candidates() -> None:
    """Only candidates that really are this title survive."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(
            200,
            crossref_search_payload(
                [
                    crossref_message(doi="10.1234/right", title="AI Governance in Universities"),
                    crossref_message(doi="10.1234/wrong", title="Soil Erosion in Semi-Arid Chile"),
                ]
            ),
        )
    )
    result = build_service(transport, provider_names=("crossref",)).lookup_title(
        "AI Governance in Universities"
    )

    assert [record.doi for record in result.records] == ["10.1234/right"]
    assert result.audit.provider_records == 2


def test_invalid_doi_lookup_is_reported_not_crashed() -> None:
    """A non-DOI string yields a provider failure, not an exception."""
    transport = QueuedTransport([])
    result = build_service(transport, provider_names=("crossref",)).lookup_doi("hello")

    assert result.records == ()
    assert result.audit.failures
    assert result.audit.failures[0].kind == "invalid_request"


# --- find_related (§23) -----------------------------------------------------


def test_find_related_uses_local_metadata_and_excludes_the_paper_itself() -> None:
    """Related search is deterministic and starts after the local paper's year."""
    index = make_local_index(
        [
            local_payload(
                document_id="doc-1",
                title="AI Governance",
                doi="10.1234/abc",
                year=2022,
            )
        ]
    )
    transport = _both_providers(
        [crossref_message(doi="10.1234/next", year=2025, title="AI Governance Revisited")],
        [],
    )
    result = build_service(transport, local_index=index, provider_names=("crossref",)).find_related(
        document_id="doc-1", max_results=5
    )

    assert result.audit.filters["year_min"] == 2023
    assert result.audit.filters["sort"] == "recent"
    assert [record.doi for record in result.records] == ["10.1234/next"]


def test_find_related_without_identity_is_refused() -> None:
    """Related search needs something to search for."""
    import pytest

    from app.discovery.service import DiscoveryError

    service = build_service(QueuedTransport([]), local_index=make_local_index([]))

    with pytest.raises(DiscoveryError):
        service.find_related()


# --- describe ----------------------------------------------------------------


def test_describe_reports_configuration_and_missing_contact_identity(tmp_path: Path) -> None:
    """The report says plainly whether a polite-pool identity is configured."""
    service = build_service(QueuedTransport([]), cache=make_cache(tmp_path))
    described = service.describe()

    assert described["providers"] == ["crossref", "openalex"]
    assert described["polite_pool_identity_configured"] is False
    assert described["cache"]["provider_request_version"]
    assert described["ranking_weights"]["title_relevance"] == 0.55
    # Citation count must not be a ranking weight.
    assert "cited_by_count" not in described["ranking_weights"]
