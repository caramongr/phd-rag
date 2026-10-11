"""RAG-006 tests 16-22: external enrichment, matching, caching, offline, duplicates."""

from __future__ import annotations

from pathlib import Path

from metadata_support import (
    FakeResponse,
    FakeTransport,
    build_test_enricher,
    build_test_stack,
    crossref_doi_handler,
    crossref_message,
    crossref_search_handler,
)

from app.metadata.enums import (
    Confidence,
    CorpusRole,
    ExternalMatchStatus,
    MetadataSource,
)
from app.metadata.external.cache import JsonCache
from app.metadata.models import AcademicMetadata, Author, MetadataValue


def _doi_metadata(doi: str = "10.1000/XYZ", *, file_hash: str | None = None) -> AcademicMetadata:
    return AcademicMetadata(
        filename="paper.pdf",
        source_path="paper.pdf",
        extension=".pdf",
        corpus_role=CorpusRole.LITERATURE,
        file_hash=file_hash,
        doi=MetadataValue(
            value=doi,
            source=MetadataSource.DOCUMENT_TEXT,
            confidence=Confidence.HIGH,
            evidence="first-page DOI pattern",
        ),
        candidates={
            "doi": [
                MetadataValue(
                    value=doi,
                    source=MetadataSource.DOCUMENT_TEXT,
                    confidence=Confidence.HIGH,
                    evidence="first-page DOI pattern",
                )
            ]
        },
    )


def _title_metadata(title: str = "Attention Is All You Need") -> AcademicMetadata:
    return AcademicMetadata(
        filename="paper.pdf",
        source_path="paper.pdf",
        extension=".pdf",
        corpus_role=CorpusRole.LITERATURE,
        title=MetadataValue(
            value=title,
            source=MetadataSource.DOCUMENT_TEXT,
            confidence=Confidence.MEDIUM,
            evidence="prominent first-page text",
        ),
        authors=MetadataValue(
            value=[Author(full_name="Jane Doe", family_name="Doe")],
            source=MetadataSource.DOCUMENT_TEXT,
            confidence=Confidence.MEDIUM,
            evidence="first-page author line",
        ),
        candidates={
            "title": [
                MetadataValue(
                    value=title,
                    source=MetadataSource.DOCUMENT_TEXT,
                    confidence=Confidence.MEDIUM,
                    evidence="prominent first-page text",
                )
            ]
        },
    )


def test_exact_doi_crossref_match_accepted() -> None:
    transport = FakeTransport(
        crossref_doi_handler(crossref_message(doi="10.1000/xyz", title="A Verified Title"))
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    result = enricher.enrich(_doi_metadata("10.1000/XYZ"))

    assert result.external_match_status is ExternalMatchStatus.EXACT
    assert result.title is not None
    assert result.title.value == "A Verified Title"
    assert result.title.source is MetadataSource.CROSSREF
    assert result.title.confidence is Confidence.HIGH


def test_title_mismatch_rejected() -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [crossref_message(title="Something Completely Different", doi="10.9999/other")]
        )
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    result = enricher.enrich(_title_metadata())

    assert result.external_match_status is ExternalMatchStatus.NO_MATCH
    assert result.title is not None
    assert result.title.value == "Attention Is All You Need"
    assert all(
        candidate.source is not MetadataSource.CROSSREF
        for candidate in result.candidates["title"]
    )


def test_ambiguous_result_not_accepted() -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [
                crossref_message(title="Attention Is All You Need", doi="10.1/a"),
                crossref_message(title="Attention Is All You Need", doi="10.1/b"),
            ]
        )
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    result = enricher.enrich(_title_metadata())

    assert result.external_match_status is ExternalMatchStatus.AMBIGUOUS
    assert result.title is not None
    assert result.title.value == "Attention Is All You Need"
    assert all(
        candidate.source is not MetadataSource.CROSSREF
        for candidate in result.candidates["title"]
    )


def test_timeout_handled_cleanly() -> None:
    def handler(url: str, params: dict | None) -> FakeResponse:
        raise TimeoutError("connection timed out")

    transport = FakeTransport(handler)
    enricher = build_test_enricher(transport, crossref=True, openalex=False, max_retries=1)

    result = enricher.enrich(_doi_metadata("10.1000/xyz"))

    assert result.external_match_status is ExternalMatchStatus.NO_MATCH
    # One initial attempt plus one retry, then a clean give-up.
    assert len(transport.calls) == 2


def test_cache_prevents_duplicate_request(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path / "cache.json")
    handler = crossref_doi_handler(crossref_message(doi="10.1000/xyz"))

    first_transport = FakeTransport(handler)
    first = build_test_enricher(first_transport, cache=cache, crossref=True, openalex=False)
    first.enrich(_doi_metadata("10.1000/xyz", file_hash="hash-one"))

    second_transport = FakeTransport(handler)
    second = build_test_enricher(second_transport, cache=cache, crossref=True, openalex=False)
    second.enrich(_doi_metadata("10.1000/xyz", file_hash="hash-two"))

    assert len(first_transport.calls) == 1
    assert second_transport.calls == []


def test_offline_mode_performs_zero_external_requests() -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message()))
    enricher = build_test_enricher(transport, crossref=True, openalex=True, enabled=False)

    result = enricher.enrich(_doi_metadata("10.1000/xyz"))

    assert transport.calls == []
    assert result.external_match_status is None


def test_same_hash_enriched_once() -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message(doi="10.1000/xyz")))
    enricher = build_test_enricher(transport, crossref=True, openalex=False)
    metadata = _doi_metadata("10.1000/xyz", file_hash="same-hash")

    first = enricher.enrich(metadata)
    second = enricher.enrich(metadata)

    assert len(transport.calls) == 1
    assert first.title.value == second.title.value == "A Verified Title"


# --- DOI conflict safeguard (post-review correction) ------------------------


def _doi_with_local_title(doi: str, title: str) -> AcademicMetadata:
    doi_value = MetadataValue(
        value=doi,
        source=MetadataSource.DOCUMENT_TEXT,
        confidence=Confidence.HIGH,
        evidence="first-page DOI pattern",
    )
    title_value = MetadataValue(
        value=title,
        source=MetadataSource.DOCUMENT_TEXT,
        confidence=Confidence.MEDIUM,
        evidence="prominent first-page text",
    )
    return AcademicMetadata(
        filename="paper.pdf",
        source_path="paper.pdf",
        extension=".pdf",
        corpus_role=CorpusRole.LITERATURE,
        doi=doi_value,
        title=title_value,
        candidates={"doi": [doi_value], "title": [title_value]},
    )


def test_exact_doi_with_matching_local_title_accepted() -> None:
    transport = FakeTransport(
        crossref_doi_handler(crossref_message(doi="10.1000/xyz", title="A Verified Title"))
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    result = enricher.enrich(_doi_with_local_title("10.1000/xyz", "A Verified Title"))

    assert result.external_match_status is ExternalMatchStatus.EXACT
    assert result.conflicts == []
    assert result.title.value == "A Verified Title"
    assert result.title.source is MetadataSource.CROSSREF


def test_exact_doi_with_conflicting_local_title_is_conflict() -> None:
    external_title = (
        "Six Practical Recommendations Enabling Ethical Use of Predictive "
        "Learning Analytics in Distance Education"
    )
    transport = FakeTransport(
        crossref_doi_handler(crossref_message(doi="10.18608/jla.2023.7743", title=external_title))
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)
    local_title = (
        "Stakeholder Perspectives on the Ethics of AI in Distance-Based Higher Education"
    )

    result = enricher.enrich(_doi_with_local_title("10.18608/jla.2023.7743", local_title))

    assert result.external_match_status is ExternalMatchStatus.CONFLICT
    # The local title is preserved; the external metadata is not made preferred.
    assert result.title is not None
    assert result.title.value == local_title
    assert all(
        candidate.source is not MetadataSource.CROSSREF
        for candidate in result.candidates.get("title", [])
    )
    # Both candidates are preserved and the conflict is recorded.
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.field == "title"
    assert conflict.local_value == local_title
    assert conflict.external_value == external_title
    assert conflict.external_source is MetadataSource.CROSSREF
    assert conflict.similarity < 0.6


def test_exact_doi_with_minor_title_differences_accepted() -> None:
    external_title = (
        "Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, "
        "Opportunities and Challenges toward Responsible AI"
    )
    transport = FakeTransport(
        crossref_doi_handler(crossref_message(doi="10.1000/xyz", title=external_title))
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)
    local_title = (
        "Explainable Artificial Intelligence: Concepts, Taxonomies, Opportunities "
        "and Challenges"
    )

    result = enricher.enrich(_doi_with_local_title("10.1000/xyz", local_title))

    assert result.external_match_status is ExternalMatchStatus.EXACT
    assert result.conflicts == []
    assert result.title.value == external_title


def test_exact_doi_without_local_title_accepted() -> None:
    transport = FakeTransport(
        crossref_doi_handler(crossref_message(doi="10.1000/xyz", title="A Verified Title"))
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    result = enricher.enrich(_doi_metadata("10.1000/xyz"))

    assert result.external_match_status is ExternalMatchStatus.EXACT
    assert result.title is not None and result.title.value == "A Verified Title"


def test_weak_filename_title_does_not_raise_conflict() -> None:
    """A filename-derived title is too weak to declare a conflict."""
    transport = FakeTransport(
        crossref_doi_handler(crossref_message(doi="10.1000/xyz", title="A Verified Title"))
    )
    enricher = build_test_enricher(transport, crossref=True, openalex=False)
    metadata = _doi_with_local_title("10.1000/xyz", "unrelated filename token")
    weak_title = MetadataValue(
        value="unrelated filename token",
        source=MetadataSource.FILENAME,
        confidence=Confidence.LOW,
        evidence="author-like token in filename",
    )
    metadata = metadata.model_copy(update={"title": weak_title})

    result = enricher.enrich(metadata)

    assert result.external_match_status is ExternalMatchStatus.EXACT
    assert result.conflicts == []


# --- DOI-only mode and enrichment statistics (stage 1) ----------------------


def test_doi_only_mode_performs_no_title_lookup(tmp_path: Path) -> None:
    transport = FakeTransport(lambda url, params: FakeResponse(200, {}))
    enricher, _components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=True,
        title_lookup_enabled=False,
    )

    result = enricher.enrich(_title_metadata())

    assert transport.calls == []
    assert result.external_match_status is None
    assert enricher.stats.title_lookups == 0
    assert enricher.stats.eligible_no_doi == 1


def test_doi_only_mode_never_issues_title_search(tmp_path: Path) -> None:
    transport = FakeTransport(lambda url, params: FakeResponse(404, {}))
    enricher, _components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=False,
        title_lookup_enabled=False,
    )

    result = enricher.enrich(_doi_metadata("10.1000/xyz"))

    assert result.external_match_status is ExternalMatchStatus.NO_MATCH
    assert all("/works/" in call["url"] for call in transport.calls)
    assert all(call["params"] is None for call in transport.calls)
    assert enricher.stats.title_lookups == 0


def test_enrichment_stats_track_accepted_and_conflict(tmp_path: Path) -> None:
    def handler(url: str, params: dict | None) -> FakeResponse:
        if "10.1000/ok" in url:
            return FakeResponse(
                200, {"message": crossref_message(doi="10.1000/ok", title="A Verified Title")}
            )
        if "10.1000/bad" in url:
            return FakeResponse(
                200,
                {
                    "message": crossref_message(
                        doi="10.1000/bad",
                        title="A Totally Different Work About Widgets",
                    )
                },
            )
        return FakeResponse(404, {})

    transport = FakeTransport(handler)
    enricher, components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=False,
        title_lookup_enabled=False,
    )

    enricher.enrich(_doi_with_local_title("10.1000/ok", "A Verified Title"))
    enricher.enrich(_doi_with_local_title("10.1000/bad", "Stakeholder Perspectives on Ethics"))
    enricher.enrich(_title_metadata())

    stats = enricher.stats
    assert stats.eligible_doi == 2
    assert stats.crossref_exact == 2
    assert stats.accepted == 1
    assert stats.conflicts == 1
    assert stats.accepted_by_source == {"CROSSREF": 1}
    assert stats.conflicts_by_source == {"CROSSREF": 1}
    assert stats.title_lookups == 0
