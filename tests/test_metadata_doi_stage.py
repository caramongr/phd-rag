"""RAG-006 stage 1: DOI-only corpus stage runner (offline, mocked externals)."""

from __future__ import annotations

from pathlib import Path

from metadata_support import (
    FakeResponse,
    FakeTransport,
    build_test_stack,
    crossref_message,
    make_pdf,
)

from app.metadata.doi_stage import run_doi_stage
from app.metadata.enums import ExternalMatchStatus
from app.metadata.external.cache import JsonCache


def _corpus(tmp_path: Path) -> Path:
    literature = tmp_path / "data" / "literature"
    literature.mkdir(parents=True)
    make_pdf(
        literature,
        "accepted.pdf",
        lines=[
            ("A Prominent First Page Title", 18.0, 80.0),
            ("Body text with doi:10.1000/xyz here.", 10.0, 130.0),
        ],
    )
    make_pdf(
        literature,
        "unresolved.pdf",
        lines=[
            ("Another Prominent First Page Title", 18.0, 80.0),
            ("Body text with doi:10.1000/missing here.", 10.0, 130.0),
        ],
    )
    return tmp_path / "data"


def test_run_doi_stage_reports_metrics_and_coverage(tmp_path: Path) -> None:
    def handler(url: str, params: dict | None) -> FakeResponse:
        if "10.1000/xyz" in url:
            return FakeResponse(
                200,
                {
                    "message": crossref_message(
                        doi="10.1000/xyz",
                        title="A Prominent First Page Title",
                        year=2019,
                        venue="Journal of Testing",
                        type_="journal-article",
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

    result = run_doi_stage(_corpus(tmp_path), enricher=enricher, components=components)

    assert result.unique_documents == 2
    assert result.with_local_doi == 2
    assert result.eligible_for_lookup == 2
    assert result.stats.crossref_exact == 1
    assert result.stats.accepted == 1
    assert result.stats.no_match == 1
    assert result.stats.title_lookups == 0
    # Only DOI endpoints were hit — no title searches.
    assert all("/works/" in call["url"] for call in transport.calls)
    assert len(result.unresolved) == 1
    assert result.unresolved[0].status == ExternalMatchStatus.NO_MATCH.value
    # Venue improves from 0 to 1; there was no venue locally.
    assert result.before.venue == 0
    assert result.after.venue == 1
    assert result.after.doi == 2
    assert result.http_requests == len(transport.calls)


def test_run_doi_stage_skips_non_literature_roles(tmp_path: Path) -> None:
    rules = tmp_path / "data" / "rules"
    rules.mkdir(parents=True)
    make_pdf(
        rules,
        "style.pdf",
        lines=[
            ("House Style Guidance", 18.0, 80.0),
            ("Body text with doi:10.1000/styles here.", 10.0, 130.0),
        ],
    )

    transport = FakeTransport(lambda url, params: FakeResponse(200, {"message": {}}))
    enricher, components = build_test_stack(
        transport,
        JsonCache(tmp_path / "cache.json"),
        crossref=True,
        openalex=False,
        title_lookup_enabled=False,
    )

    result = run_doi_stage(tmp_path / "data", enricher=enricher, components=components)

    assert result.with_local_doi == 1
    assert result.eligible_for_lookup == 0
    assert result.skipped_not_enrichable == 1
    assert transport.calls == []
