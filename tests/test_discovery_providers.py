"""Provider adapter tests (RAG-010 §30 items 1-7).

Every test here is offline: provider responses come from a fake transport, so the
suite proves the *normalization* and the *failure handling*, never the live API.
"""

from __future__ import annotations

from app.discovery.models import DiscoveryFilters, FulltextStatus
from app.discovery.providers.crossref import CrossrefProvider, work_from_message
from app.discovery.providers.openalex import OpenAlexProvider, work_from_work
from app.discovery.providers.semanticscholar import SemanticScholarProvider
from app.metadata.doi import doi_cache_key
from tests.discovery_support import (
    FakeResponse,
    QueuedTransport,
    RoutingTransport,
    crossref_message,
    crossref_search_payload,
    make_http,
    openalex_search_payload,
    openalex_work,
)


# --- 1. Crossref normalization ----------------------------------------------


def test_crossref_message_normalizes_all_core_fields() -> None:
    """A Crossref message becomes a complete external record."""
    work = work_from_message(
        crossref_message(
            doi="10.1234/AbC",
            title="Artificial Intelligence in Higher Education",
            authors=[("Jane", "Doe"), ("John", "Roe")],
            year=2021,
            venue="Journal of AI",
            publisher="AI Press",
            abstract="<jats:p>An abstract.</jats:p>",
        )
    )

    assert work.doi == "10.1234/AbC"
    # DOI comparison is case-insensitive even though the published casing is kept.
    assert doi_cache_key(work.doi) == doi_cache_key("10.1234/abc")
    assert work.title == "Artificial Intelligence in Higher Education"
    assert work.authors == ("Jane Doe", "John Roe")
    assert work.year == 2021
    assert work.venue == "Journal of AI"
    assert work.publisher == "AI Press"
    assert work.publication_type == "journal-article"
    assert work.abstract == "An abstract."
    assert work.providers == ("crossref",)
    assert work.field_provenance["title"] == ("crossref",)
    assert work.missing_fields == ()


def test_crossref_states_no_open_access_or_citation_count() -> None:
    """Fields Crossref does not supply stay empty rather than being inferred."""
    work = work_from_message(crossref_message())

    assert work.open_access is None
    assert work.open_access_status is None
    assert work.fulltext_status is FulltextStatus.UNKNOWN


def test_crossref_fulltext_only_when_openly_licensed() -> None:
    """A PDF link without an open licence is not claimed as open full text."""
    closed = work_from_message(
        crossref_message(pdf_link="https://example.org/paper.pdf")
    )
    assert closed.fulltext_status is FulltextStatus.LANDING_PAGE_ONLY

    licensed = work_from_message(
        crossref_message(
            license_="https://creativecommons.org/licenses/by/4.0/",
            pdf_link="https://example.org/paper.pdf",
        )
    )
    assert licensed.fulltext_status is FulltextStatus.FULLTEXT_OPEN
    assert licensed.fulltext_url == "https://example.org/paper.pdf"


def test_crossref_accepts_a_pdf_url_declared_as_unspecified() -> None:
    """Several publishers leave ``content-type`` as 'unspecified' for a real PDF URL."""
    message = crossref_message(license_="https://creativecommons.org/licenses/by/4.0/")
    message["link"] = [
        {
            "URL": "https://www.mdpi.com/2078-2489/16/4/296/pdf",
            "content-type": "unspecified",
        }
    ]

    work = work_from_message(message)

    assert work.fulltext_status is FulltextStatus.FULLTEXT_OPEN
    assert work.fulltext_url == "https://www.mdpi.com/2078-2489/16/4/296/pdf"


def test_crossref_does_not_treat_a_tdm_endpoint_as_a_pdf() -> None:
    """An XML/plain-text API endpoint is not an open PDF, even with a CC licence."""
    message = crossref_message(license_="http://creativecommons.org/licenses/by/4.0/")
    message["link"] = [
        {
            "URL": "https://api.elsevier.com/content/article/PII:S0360?httpAccept=text/xml",
            "content-type": "text/xml",
        }
    ]

    work = work_from_message(message)

    assert work.fulltext_status is FulltextStatus.LANDING_PAGE_ONLY
    assert work.fulltext_url is None


def test_crossref_search_sends_bibliographic_query_and_filters() -> None:
    """Search builds the documented Crossref query, including polite mailto."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, crossref_search_payload([]))
    )
    provider = CrossrefProvider(make_http(transport), mailto="someone@example.org")

    provider.search(
        "AI governance",
        limit=7,
        filters=DiscoveryFilters(year_min=2024, year_max=2026, author="Doe"),
    )

    params = transport.calls[0]["params"]
    assert params["query.bibliographic"] == "AI governance"
    assert params["rows"] == 7
    assert params["query.author"] == "Doe"
    assert "from-pub-date:2024-01-01" in params["filter"]
    assert "until-pub-date:2026-12-31" in params["filter"]
    assert params["mailto"] == "someone@example.org"
    assert "mailto:someone@example.org" in transport.calls[0]["headers"]["User-Agent"]


def test_crossref_search_declares_empty_query_instead_of_requesting() -> None:
    """An empty query is an invalid request, not a network round trip."""
    transport = QueuedTransport([])
    response = CrossrefProvider(make_http(transport)).search("   ")

    assert not response.ok
    assert response.failure is not None
    assert response.failure.kind == "invalid_request"
    assert transport.call_count == 0


def test_crossref_doi_lookup_rejects_a_non_doi_without_a_request() -> None:
    """A string that is not a DOI is reported, not sent to Crossref."""
    transport = QueuedTransport([])
    response = CrossrefProvider(make_http(transport)).lookup_doi("not a doi")

    assert response.failure is not None
    assert response.failure.kind == "invalid_request"
    assert transport.call_count == 0


# --- 2. OpenAlex normalization ----------------------------------------------


def test_openalex_work_normalizes_all_core_fields() -> None:
    """An OpenAlex work becomes a complete external record, abstract included."""
    work = work_from_work(
        openalex_work(
            doi="10.1234/AbC",
            title="Generative AI Assessment",
            authors=["Jane Doe", "John Roe"],
            year=2025,
            venue="Assessment & Evaluation",
            publisher="Elsevier",
            abstract_words=["Students", "use", "generative", "AI"],
            is_oa=True,
            pdf_url="https://example.org/paper.pdf",
            cited_by=42,
        )
    )

    assert work.doi == "10.1234/AbC"
    assert doi_cache_key(work.doi) == doi_cache_key("10.1234/abc")
    assert work.title == "Generative AI Assessment"
    assert work.authors == ("Jane Doe", "John Roe")
    assert work.year == 2025
    assert work.venue == "Assessment & Evaluation"
    assert work.publisher == "Elsevier"
    assert work.abstract == "Students use generative AI"
    assert work.cited_by_count == 42
    assert work.open_access is True
    assert work.open_access_status == "gold"
    assert work.fulltext_status is FulltextStatus.FULLTEXT_OPEN
    assert work.fulltext_url == "https://example.org/paper.pdf"
    assert work.raw_provider_ids["openalex"] == "W123"


def test_openalex_reconstructs_inverted_abstract_deterministically() -> None:
    """The inverted index is reassembled in position order."""
    work = work_from_work(
        {
            "id": "https://openalex.org/W1",
            "title": "T",
            "abstract_inverted_index": {"world": [1], "Hello": [0], "again": [2]},
        }
    )
    assert work.abstract == "Hello world again"


def test_openalex_recent_search_sorts_by_publication_date() -> None:
    """The 'recent' sort is sent to OpenAlex and the year filter is passed through."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, openalex_search_payload([]))
    )
    provider = OpenAlexProvider(make_http(transport), mailto="someone@example.org")

    provider.search("AI governance", limit=5, filters=DiscoveryFilters(year_min=2025, sort="recent"))

    params = transport.calls[0]["params"]
    assert params["search"] == "AI governance"
    assert params["per-page"] == 5
    assert params["sort"] == "publication_date:desc"
    assert params["filter"] == "from_publication_date:2025-01-01"
    assert params["mailto"] == "someone@example.org"


def test_openalex_sends_api_key_when_configured() -> None:
    """A configured API key is sent as a request parameter."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, openalex_search_payload([]))
    )
    provider = OpenAlexProvider(make_http(transport), api_key="secret-key")

    provider.search("AI")

    assert transport.calls[0]["params"]["api_key"] == "secret-key"


# --- 3. Missing fields stay missing -----------------------------------------


def test_missing_fields_are_recorded_not_invented() -> None:
    """A sparse record reports exactly which fields it lacks."""
    crossref = work_from_message({"DOI": "10.1/x", "title": []})
    openalex = work_from_work({"id": "https://openalex.org/W1"})

    assert crossref.title is None
    assert set(crossref.missing_fields) >= {"title", "authors", "year", "venue", "abstract"}
    assert openalex.title is None
    assert "doi" in openalex.missing_fields
    assert openalex.year is None
    assert openalex.cited_by_count is None


# --- 4. Timeout --------------------------------------------------------------


def test_http_timeout_is_reported_as_a_network_failure() -> None:
    """A transport timeout becomes a provider failure, never an empty success."""
    transport = QueuedTransport([TimeoutError("read timed out")])
    response = CrossrefProvider(make_http(transport)).search("ai governance")

    assert not response.ok
    assert response.records == []
    assert response.failure is not None
    assert response.failure.kind == "network_error"
    assert "TimeoutError" in response.failure.message


def test_http_timeout_is_retried_then_reported() -> None:
    """Retries are bounded, and the final outcome is the failure."""
    transport = QueuedTransport(
        [TimeoutError("timed out"), TimeoutError("timed out again")]
    )
    response = CrossrefProvider(make_http(transport, max_retries=1)).search("ai")

    assert not response.ok
    assert transport.call_count == 2


# --- 5. HTTP 429 -------------------------------------------------------------


def test_http_429_is_reported_with_the_provider_message_and_retry_after() -> None:
    """A rate limit keeps the provider's own words and its Retry-After."""
    transport = QueuedTransport(
        [
            FakeResponse(
                429,
                {"error": "budget exhausted"},
                text="The shared anonymous daily budget is exhausted.",
                headers={"Retry-After": "18406"},
            )
        ]
    )
    response = OpenAlexProvider(make_http(transport)).search("ai governance")

    assert not response.ok
    assert response.failure is not None
    assert response.failure.kind == "rate_limited"
    assert response.failure.status_code == 429
    assert response.failure.retry_after == 18406.0
    assert response.failure.counted_as_quota is True
    assert "daily budget" in response.failure.message
    # The wording must not turn a provider budget into a charge to the user.
    assert "cost" not in response.failure.message.lower()


def test_rate_limit_is_not_retried_into_a_long_stall() -> None:
    """A 429 is attempted at most ``max_retries + 1`` times."""
    transport = QueuedTransport(
        [FakeResponse(429, {}, text="slow down") for _ in range(4)]
    )
    response = CrossrefProvider(make_http(transport, max_retries=1)).search("ai")

    assert not response.ok
    assert transport.call_count == 2


# --- 6. 5xx retry ------------------------------------------------------------


def test_5xx_is_retried_and_can_recover() -> None:
    """A transient server error is retried, and a later 200 succeeds."""
    transport = QueuedTransport(
        [
            FakeResponse(503, {}, text="unavailable"),
            FakeResponse(200, crossref_search_payload([crossref_message()])),
        ]
    )
    response = CrossrefProvider(make_http(transport, max_retries=2)).search("ai")

    assert response.ok
    assert len(response.records) == 1
    assert transport.call_count == 2


def test_5xx_that_never_recovers_is_reported() -> None:
    """Exhausted retries end in an http_error, not an exception."""
    transport = QueuedTransport([FakeResponse(500, {}, text="boom") for _ in range(5)])
    response = CrossrefProvider(make_http(transport, max_retries=2)).search("ai")

    assert not response.ok
    assert response.failure is not None
    assert response.failure.kind == "http_error"
    assert response.failure.status_code == 500


# --- 7. Malformed response ---------------------------------------------------


def test_malformed_json_body_is_reported_not_raised() -> None:
    """A body that is not JSON ends as a provider failure."""
    transport = QueuedTransport([FakeResponse(200, None, text="<html>nope</html>")])
    response = CrossrefProvider(make_http(transport)).search("ai")

    assert not response.ok
    assert response.failure is not None
    assert "invalid JSON" in response.failure.message


def test_unexpected_payload_shape_yields_no_records_and_no_crash() -> None:
    """A JSON body of the wrong shape yields zero records."""
    transport = QueuedTransport([FakeResponse(200, {"message": {"items": "oops"}})])
    response = CrossrefProvider(make_http(transport)).search("ai")

    assert response.ok
    assert response.records == []


def test_non_dict_items_are_skipped() -> None:
    """A list containing junk does not break normalization."""
    transport = QueuedTransport(
        [FakeResponse(200, crossref_search_payload(["not a dict", crossref_message()]))]
    )
    response = CrossrefProvider(make_http(transport)).search("ai")

    assert response.ok
    assert len(response.records) == 1


# --- optional provider -------------------------------------------------------


def test_semantic_scholar_is_optional_and_degrades_on_rate_limit() -> None:
    """Semantic Scholar is not required: a 429 is just another provider failure."""
    transport = QueuedTransport([FakeResponse(429, {}, text="Too Many Requests")])
    response = SemanticScholarProvider(make_http(transport)).search("ai governance")

    assert not response.ok
    assert response.failure is not None
    assert response.failure.provider == "semanticscholar"
    assert response.records == []


def test_crossref_doi_lookup_does_not_send_select() -> None:
    """Crossref's single-work route rejects ``select`` (HTTP 400), so it is not sent."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(200, {"message": crossref_message()})
    )
    provider = CrossrefProvider(make_http(transport))

    response = provider.lookup_doi("10.1234/abc")

    assert response.ok
    assert transport.calls[0]["params"] == {}
    assert "/works/10.1234/abc" in transport.calls[0]["url"]


def test_crossref_doi_lookup_reports_a_400_with_the_provider_message() -> None:
    """A validation failure from Crossref is surfaced with its own words."""
    transport = QueuedTransport(
        [
            FakeResponse(
                400,
                None,
                text='{"status":"failed","message-type":"validation-failure"}',
            )
        ]
    )
    response = CrossrefProvider(make_http(transport)).lookup_doi("10.1234/abc")

    assert not response.ok
    assert response.failure is not None
    assert response.failure.status_code == 400
    assert "validation-failure" in response.failure.message


def test_crossref_doi_lookup_reports_a_missing_doi_as_not_found() -> None:
    """A 404 becomes an explicit 'no such work', not a generic HTTP error."""
    transport = QueuedTransport([FakeResponse(404, {}, text="Not Found")])
    response = CrossrefProvider(make_http(transport)).lookup_doi("10.1234/missing")

    assert not response.ok
    assert response.failure is not None
    assert response.failure.kind == "not_found"


def test_crossref_records_are_provider_attributed() -> None:
    """Provider identity travels with the record, for provenance."""
    transport = RoutingTransport(
        lambda url, params: FakeResponse(
            200, crossref_search_payload([crossref_message(doi="10.1/one")])
        )
    )
    response = CrossrefProvider(make_http(transport)).search("ai")

    assert response.records[0].providers == ("crossref",)
    assert response.records[0].source_provenance == ("crossref:journal-article",)
    assert response.http_requests == 1
