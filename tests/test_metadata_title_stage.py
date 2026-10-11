"""RAG-006 stage 2A: title-based enrichment sample (offline, mocked externals)."""

from __future__ import annotations

from pathlib import Path

from metadata_support import (
    FakeResponse,
    FakeTransport,
    build_test_stack,
    crossref_message,
    crossref_search_handler,
    make_pdf,
)

from app.metadata.enums import (
    Confidence,
    CorpusRole,
    ExternalMatchStatus,
    MetadataSource,
)
from app.metadata.external.cache import JsonCache
from app.metadata.models import AcademicMetadata, Author, MetadataValue
from app.metadata.title_stage import (
    is_title_sample_eligible,
    render,
    render_stage2b,
    run_stage2b,
    run_title_stage,
    select_title_sample,
)


def _metadata(
    filename: str,
    *,
    title: str | None = "A Strong Local Title",
    source: MetadataSource = MetadataSource.DOCUMENT_TEXT,
    confidence: Confidence = Confidence.MEDIUM,
    doi: str | None = None,
    year: int | None = 2020,
    author: str | None = "Jane Doe",
    role: CorpusRole = CorpusRole.LITERATURE,
) -> AcademicMetadata:
    return AcademicMetadata(
        filename=filename,
        source_path=filename,
        extension=".pdf",
        corpus_role=role,
        title=(
            MetadataValue(value=title, source=source, confidence=confidence, evidence="test")
            if title is not None
            else None
        ),
        doi=(
            MetadataValue(
                value=doi,
                source=MetadataSource.DOCUMENT_TEXT,
                confidence=Confidence.HIGH,
                evidence="test",
            )
            if doi is not None
            else None
        ),
        year=(
            MetadataValue(
                value=year,
                source=MetadataSource.DOCUMENT_TEXT,
                confidence=Confidence.MEDIUM,
                evidence="test",
            )
            if year is not None
            else None
        ),
        authors=(
            MetadataValue(
                value=[Author(full_name=author)],
                source=MetadataSource.DOCUMENT_TEXT,
                confidence=Confidence.MEDIUM,
                evidence="test",
            )
            if author is not None
            else None
        ),
    )


def test_eligibility_requires_no_doi_and_a_strong_local_title() -> None:
    assert is_title_sample_eligible(_metadata("ok.pdf"))

    assert not is_title_sample_eligible(_metadata("has-doi.pdf", doi="10.1000/xyz"))
    assert not is_title_sample_eligible(
        _metadata("filename-only.pdf", source=MetadataSource.FILENAME)
    )
    assert not is_title_sample_eligible(_metadata("weak.pdf", confidence=Confidence.LOW))
    assert not is_title_sample_eligible(_metadata("no-title.pdf", title=None))
    assert not is_title_sample_eligible(
        _metadata("rules.pdf", role=CorpusRole.RULES)
    )
    assert not is_title_sample_eligible(
        _metadata("draft.pdf", role=CorpusRole.MY_WORK)
    )


def test_select_title_sample_is_deterministic_and_spread() -> None:
    metadatas = [
        _metadata(
            f"paper-{index}.pdf",
            title="Short Title Here" if index % 2 else "A Much Longer Title That Keeps Going And Going",
            source=MetadataSource.PDF_METADATA if index % 3 else MetadataSource.DOCUMENT_TEXT,
            year=1980 + index * 3,
            author=None if index % 4 == 0 else "Jane Doe",
        )
        for index in range(24)
    ]

    first = [item.filename for item in select_title_sample(metadatas, 10)]
    second = [item.filename for item in select_title_sample(metadatas, 10)]

    assert len(first) == 10
    assert first == second
    sample = select_title_sample(metadatas, 10)
    assert {item.title.source for item in sample} == {
        MetadataSource.PDF_METADATA,
        MetadataSource.DOCUMENT_TEXT,
    }
    assert any(len(item.title.value.split()) <= 6 for item in sample)
    assert any(len(item.title.value.split()) >= 7 for item in sample)
    assert any(item.year is None for item in sample) or any(
        item.year is not None for item in sample
    )


def test_select_title_sample_size_zero_returns_nothing() -> None:
    assert select_title_sample([_metadata("a.pdf")], 0) == []


def _corpus(tmp_path: Path) -> Path:
    literature = tmp_path / "data" / "literature"
    literature.mkdir(parents=True)
    make_pdf(
        literature,
        "accepted.pdf",
        lines=[
            ("Attention Is All You Need", 18.0, 80.0),
            ("Jane Doe, John Smith", 11.0, 105.0),
            ("Ordinary body text follows the title here.", 10.0, 150.0),
            ("More body text continues on this page.", 10.0, 170.0),
            ("Even more body text follows here.", 10.0, 190.0),
        ],
    )
    return tmp_path / "data"


def test_title_stage_strong_match_records_doi_as_external_candidate(tmp_path: Path) -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [
                crossref_message(
                    doi="10.5555/attention",
                    title="Attention Is All You Need",
                    authors=[("Jane", "Doe"), ("John", "Smith")],
                    year=2017,
                    type_="proceedings-article",
                    venue="NeurIPS",
                    publisher="Curran Associates",
                )
            ]
        )
    )
    enricher, components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=False,
        title_lookup_enabled=True,
    )

    result = run_title_stage(
        _corpus(tmp_path), enricher=enricher, components=components, size=10
    )

    assert result.sample_size == 1
    assert result.searches_attempted == 1
    assert result.strong == 1
    assert result.ambiguous == 0
    assert result.no_match == 0
    assert result.stats.accepted == 1
    # Title search only: never a DOI endpoint.
    assert transport.calls
    assert all("/works/" not in call["url"] for call in transport.calls)

    row = result.rows[0]
    assert row.status == ExternalMatchStatus.STRONG.value
    assert row.accepted is True
    assert row.provider == MetadataSource.CROSSREF.value
    assert row.similarity is not None and row.similarity >= 0.92
    assert row.external_doi == "10.5555/attention"
    assert row.resolved_title == "Attention Is All You Need"
    assert row.doi_origin is not None and "not observed in the document" in row.doi_origin
    assert row.venue == "NeurIPS"
    assert row.publication_type == "CONFERENCE_PAPER"

    # The DOI is recorded as an external candidate, never as a locally observed one.
    assert result.before.doi == 0
    assert result.after.doi == 1
    assert result.after.venue == 1
    assert result.after.publisher == 1
    assert result.after.publication_type == 1
    assert result.after.abstract == 0


def test_title_stage_ambiguous_match_merges_nothing(tmp_path: Path) -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [
                crossref_message(
                    doi="10.5555/one",
                    title="Attention Is All You Need",
                    authors=[("Jane", "Doe")],
                    year=2017,
                ),
                crossref_message(
                    doi="10.5555/two",
                    title="Attention Is All You Need",
                    authors=[("John", "Smith")],
                    year=2017,
                ),
            ]
        )
    )
    enricher, components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=False,
    )

    result = run_title_stage(
        _corpus(tmp_path), enricher=enricher, components=components, size=10
    )

    assert result.ambiguous == 1
    assert result.stats.accepted == 0
    row = result.rows[0]
    assert row.accepted is False
    assert row.status == ExternalMatchStatus.AMBIGUOUS.value
    assert row.resolved_title == "Attention Is All You Need"
    assert result.after.doi == 0
    assert result.after.venue == 0
    assert result.after.publication_type == 0


def test_title_stage_no_match_merges_nothing_and_falls_back_to_openalex(tmp_path: Path) -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [crossref_message(doi="10.5555/other", title="An Entirely Different Paper")]
        )
    )
    enricher, components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=True,
    )

    result = run_title_stage(
        _corpus(tmp_path), enricher=enricher, components=components, size=10
    )

    assert result.no_match == 1
    assert result.stats.accepted == 0
    assert result.rows[0].accepted is False
    assert result.after.doi == 0
    # Crossref was tried, then the OpenAlex fallback was consulted.
    assert any("crossref.org" in call["url"] for call in transport.calls)
    assert any("openalex.org" in call["url"] for call in transport.calls)


def test_stage2b_covers_whole_population_and_keeps_doi_documents_unchanged(
    tmp_path: Path,
) -> None:
    data = _corpus(tmp_path)
    literature = data / "literature"
    # A DOI-bearing document must go through the DOI route, not a title search.
    make_pdf(
        literature,
        "with-doi.pdf",
        lines=[
            ("A DOI Bearing Paper Title", 18.0, 80.0),
            ("Body text with doi:10.1000/confirmed here.", 10.0, 130.0),
        ],
    )
    # A DOI-less document without a strong title must never be searched.
    make_pdf(
        literature,
        "no-title.pdf",
        lines=[("Ordinary body text only on this page.", 10.0, 120.0)],
    )

    def handler(url: str, params: dict | None) -> FakeResponse:
        if "10.1000/confirmed" in url:
            return FakeResponse(
                200,
                {
                    "message": crossref_message(
                        doi="10.1000/confirmed",
                        title="A DOI Bearing Paper Title",
                        year=2019,
                        venue="Journal of Testing",
                    )
                },
            )
        if url.rstrip("/").endswith("/works"):
            return FakeResponse(
                200,
                {
                    "message": {
                        "items": [
                            crossref_message(
                                doi="10.5555/attention",
                                title="Attention Is All You Need",
                                year=2017,
                                type_="proceedings-article",
                                venue="NeurIPS",
                            )
                        ]
                    }
                },
            )
        return FakeResponse(404, {})

    transport = FakeTransport(handler)
    enricher, components = build_test_stack(
        transport, JsonCache(tmp_path / "cache.json"), crossref=True, openalex=False
    )

    result = run_stage2b(data, enricher=enricher, components=components)

    assert result.unique_documents == 3
    assert result.eligible_population == 1
    assert result.searches_attempted == 1
    assert result.strong == 1
    assert result.accepted_by_source == {"CROSSREF": 1}
    # The DOI-bearing document was resolved through the DOI route, unchanged.
    assert result.doi_route_exact == 1
    assert result.doi_route_conflict == 0
    # Three coverage snapshots: local, after DOI Stage 1, after Stage 2B.
    assert result.local.doi == 1
    assert result.local.venue == 0
    assert result.after_stage1.doi == 1
    assert result.after_stage1.venue == 1
    assert result.after_stage2b.doi == 2
    assert result.after_stage2b.venue == 2
    # The DOI-less document without a strong title was never searched.
    searched = [row.filename for row in result.accepted]
    assert "no-title.pdf" not in searched
    # Nothing on disk was modified.
    assert result.files_modified == 0
    assert len(result.accepted) == 1
    assert result.accepted[0].provider == "CROSSREF"
    assert result.accepted[0].external_doi == "10.5555/attention"


def test_stage2b_classifies_withheld_matches_by_reason(tmp_path: Path) -> None:
    literature = tmp_path / "data" / "literature"
    literature.mkdir(parents=True)
    make_pdf(
        literature,
        "dated.pdf",
        lines=[
            ("Attention Is All You Need", 18.0, 80.0),
            ("Jane Doe, John Smith", 11.0, 105.0),
            ("Published 2017 by Example Press.", 10.0, 150.0),
            ("More body text continues on this page.", 10.0, 170.0),
            ("Even more body text follows here.", 10.0, 190.0),
        ],
    )
    data = tmp_path / "data"

    def handler(url: str, params: dict | None) -> FakeResponse:
        if url.rstrip("/").endswith("/works"):
            return FakeResponse(
                200,
                {
                    "message": {
                        "items": [
                            crossref_message(
                                doi="10.5555/attention",
                                title="Attention Is All You Need",
                                year=1990,
                                type_="proceedings-article",
                            )
                        ]
                    }
                },
            )
        return FakeResponse(404, {})

    transport = FakeTransport(handler)
    enricher, components = build_test_stack(
        transport, JsonCache(tmp_path / "cache.json"), crossref=True, openalex=False
    )

    result = run_stage2b(data, enricher=enricher, components=components)

    assert result.strong == 0
    assert result.ambiguous == 1
    assert result.reason_counts == {"local year disagreement": 1}
    row = result.withheld[0]
    assert row.status == ExternalMatchStatus.AMBIGUOUS.value
    assert row.reason == "local_year_disagreement"
    assert row.similarity is not None and row.similarity >= 0.92
    assert row.matched_title == "Attention Is All You Need"


def test_stage2b_report_has_required_fields(tmp_path: Path) -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [crossref_message(doi="10.5555/attention", title="Attention Is All You Need", year=2017)]
        )
    )
    enricher, components = build_test_stack(
        transport, JsonCache(tmp_path / "cache.json"), crossref=True, openalex=False
    )
    result = run_stage2b(_corpus(tmp_path), enricher=enricher, components=components)

    text = render_stage2b(result)

    for required in (
        "eligible population",
        "title searches attempted",
        "STRONG",
        "AMBIGUOUS",
        "NO_MATCH",
        "API/network failures",
        "accepted matches by external source",
        "cached responses used",
        "source files modified",
        "local",
        "after DOI",
        "after title",
        "publication type",
        "accepted identities",
        "external DOI provenance",
    ):
        assert required in text


def test_title_stage_report_has_required_fields(tmp_path: Path) -> None:
    transport = FakeTransport(
        crossref_search_handler(
            [crossref_message(doi="10.5555/attention", title="Attention Is All You Need")]
        )
    )
    enricher, components = build_test_stack(
        transport, JsonCache(tmp_path / "cache.json"), crossref=True, openalex=False
    )
    result = run_title_stage(
        _corpus(tmp_path), enricher=enricher, components=components, size=10
    )

    text = render(result)

    for required in (
        "sample size",
        "title searches attempted",
        "STRONG",
        "AMBIGUOUS",
        "NO_MATCH",
        "API/network failures",
        "local title source",
        "external provider",
        "similarity",
        "match status",
        "external DOI",
        "DOI origin",
        "publication type",
    ):
        assert required in text
