"""Discovery CLI tests (RAG-010 §24-§26).

The CLIs are exercised offline by stubbing the service/settings, so no test opens a
socket or reads the real index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.discovery import render, search
from app.discovery.models import (
    DiscoveryAudit,
    DiscoveryResult,
    DiscoveryStatus,
    ExternalWork,
    FieldConflict,
    FulltextStatus,
    LocalMatch,
    LocalMatchKind,
)


def _record(**kwargs: Any) -> ExternalWork:
    """An external record with a readable default shape."""
    base: dict[str, Any] = {
        "title": "AI Governance in Universities",
        "authors": ("Jane Doe", "John Roe"),
        "year": 2025,
        "doi": "10.1234/abc",
        "venue": "Policy Journal",
        "publisher": "Spring",
        "publication_type": "journal-article",
        "abstract": "An abstract about governance.",
        "providers": ("crossref", "openalex"),
        "discovery_status": DiscoveryStatus.EXTERNAL_METADATA_VERIFIED,
        "identity_confidence": 0.85,
    }
    base.update(kwargs)
    return ExternalWork(**base)  # type: ignore[arg-type]


def _audit(**kwargs: Any) -> DiscoveryAudit:
    """A minimal audit trail for rendering tests."""
    base: dict[str, Any] = {
        "raw_query": "ai governance",
        "normalized_query": "ai governance",
        "providers_requested": ("crossref",),
        "providers_succeeded": ("crossref",),
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    base.update(kwargs)
    return DiscoveryAudit(**base)  # type: ignore[arg-type]


def _result(records: list[ExternalWork], **kwargs: Any) -> DiscoveryResult:
    """A discovery result for rendering tests."""
    return DiscoveryResult(audit=_audit(**kwargs), records=tuple(records))


# --- 26. user-facing metadata ----------------------------------------------


def test_rendered_record_shows_the_metadata_a_reader_needs() -> None:
    """Title, authors, year, venue, DOI, providers and status are all shown."""
    rendered = render.format_record(_record(), index=1)

    assert "AI Governance in Universities" in rendered
    assert "Jane Doe, John Roe" in rendered
    assert "2025" in rendered
    assert "Policy Journal" in rendered
    assert "10.1234/abc" in rendered
    assert "crossref, openalex" in rendered
    assert "verification" in rendered


def test_abstract_is_hidden_unless_requested() -> None:
    """The default view stays compact; ``--show-abstract`` adds the preview."""
    record = _record()

    assert "An abstract about governance." not in render.format_record(record)
    assert "An abstract about governance." in render.format_record(
        record, show_abstract=True
    )


def test_conflicts_are_shown_with_both_values() -> None:
    """A conflict is visible in the default output, not hidden in JSON."""
    record = _record(
        discovery_status=DiscoveryStatus.CONFLICT,
        conflicts=(
            FieldConflict(
                field_name="title",
                values=("One Title", "Another Title"),
                providers=("crossref", "openalex"),
            ),
        ),
    )

    rendered = render.format_record(record)

    assert "CONFLICT" in rendered
    assert "One Title" in rendered
    assert "Another Title" in rendered


def test_missing_fields_are_stated_plainly() -> None:
    """A sparse record says what its provider did not supply."""
    rendered = render.format_record(_record(abstract=None, missing_fields=("abstract",)))

    assert "missing      : abstract" in rendered


# --- 24. grouping of the default output -------------------------------------


def test_result_is_grouped_into_readable_sections() -> None:
    """Already-have, verified, discovery-only and conflicts are separate groups."""
    local = _record(
        doi="10.1234/local",
        local_match=LocalMatch(
            already_in_corpus=True,
            kind=LocalMatchKind.ALREADY_IN_CORPUS,
            local_filename="known.pdf",
            local_corpus_role="literature",
            local_evidence_eligible=True,
            match_reason="doi_identity",
        ),
    )
    verified = _record(doi="10.1234/verified")
    unverified = _record(
        doi="10.1234/new",
        discovery_status=DiscoveryStatus.EXTERNAL_DISCOVERY,
        missing_fields=("abstract",),
    )
    conflicting = _record(
        doi="10.1234/conflict",
        discovery_status=DiscoveryStatus.CONFLICT,
        conflicts=(
            FieldConflict(field_name="year", values=("2020", "2024"), providers=("crossref",)),
        ),
    )

    rendered = render.format_result(
        _result([local, verified, unverified, conflicting])
    )

    assert "ALREADY IN LOCAL CORPUS (1)" in rendered
    assert "NEW / VERIFIED METADATA (1)" in rendered
    assert "DISCOVERY ONLY (identity not verified) (1)" in rendered
    assert "CONFLICTS (providers disagree) (1)" in rendered


def test_new_only_hides_records_already_held_locally() -> None:
    """``--new-only`` keeps the local half out of view."""
    local = _record(
        doi="10.1234/local",
        local_match=LocalMatch(
            already_in_corpus=True,
            kind=LocalMatchKind.ALREADY_IN_CORPUS,
            local_filename="known.pdf",
        ),
    )
    fresh = _record(doi="10.1234/fresh")

    rendered = render.format_result(_result([local, fresh]), new_only=True)

    assert "10.1234/fresh" in rendered
    assert "10.1234/local" not in rendered


def test_audit_header_reports_counts_and_cache() -> None:
    """The audit header shows what happened, including cache hits."""
    rendered = render.format_result(
        _result(
            [_record()],
            provider_records=12,
            merged_records=9,
            duplicates_merged=3,
            new_records=8,
            cache_hits=2,
            cache_misses=1,
        )
    )

    assert "provider_records=12" in rendered
    assert "duplicates_merged=3" in rendered
    assert "hits=2 misses=1" in rendered


def test_provider_failure_is_reported_in_the_audit() -> None:
    """A provider outage is visible in the output rather than looking like 'nothing found'."""
    from app.discovery.models import ProviderFailure

    rendered = render.format_result(
        _result(
            [],
            providers_succeeded=(),
            providers_failed=("openalex",),
            failures=(
                ProviderFailure(
                    provider="openalex",
                    kind="rate_limited",
                    status_code=429,
                    message="provider rate limit reached (HTTP 429)",
                    retry_after=3600.0,
                    counted_as_quota=True,
                ),
            ),
            all_providers_failed=True,
        )
    )

    assert "all_providers_failed" not in rendered  # internal flag, not user text
    assert "openalex [rate_limited] HTTP 429" in rendered
    assert "retry_after=3600s" in rendered
    assert "every external provider failed" in rendered


# --- CLI argument handling ---------------------------------------------------


def test_parse_args_defaults() -> None:
    """The documented flags exist with sane defaults."""
    args = search.parse_args(["ai governance"])

    assert args.query == "ai governance"
    assert args.limit == 20
    assert args.provider is None
    assert args.sort == "relevance"
    assert args.refresh is False
    assert args.verify_fulltext is False


def test_parse_args_accepts_the_documented_flags() -> None:
    """``--year-min/--year-max/--provider/--new-only`` parse as documented."""
    args = search.parse_args(
        [
            "generative AI academic integrity",
            "--year-min",
            "2025",
            "--provider",
            "crossref",
            "--provider",
            "openalex",
            "--new-only",
            "--refresh",
        ]
    )

    assert args.year_min == 2025
    assert args.provider == ["crossref", "openalex"]
    assert args.new_only is True
    assert args.refresh is True


def test_empty_query_is_refused() -> None:
    """Running with no query, DOI or title exits with a message."""
    with pytest.raises(SystemExit):
        search.run(search.parse_args([]), service=object())  # type: ignore[arg-type]


def test_cli_json_output_is_machine_readable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--json`` emits the audit and the records, with no prose."""

    class _Service:
        provider_names = ()

        def discover(self, query: str, **kwargs: Any) -> DiscoveryResult:
            return _result([_record()])

        def lookup_doi(self, doi: str, **kwargs: Any) -> DiscoveryResult:
            raise AssertionError("unexpected DOI lookup")

        def lookup_title(self, title: str, **kwargs: Any) -> DiscoveryResult:
            raise AssertionError("unexpected title lookup")

    monkeypatch.setattr(search, "build_service", lambda _args: _Service())
    exit_code = search.main(["ai governance", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["records"][0]["doi"] == "10.1234/abc"
    assert payload["audit"]["normalized_query"] == "ai governance"


def test_cli_reports_a_total_provider_outage_clearly(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every provider fails the CLI says so instead of printing an empty list."""

    class _Service:
        provider_names = ("crossref",)

        def discover(self, query: str, **kwargs: Any) -> DiscoveryResult:
            return _result(
                [],
                providers_succeeded=(),
                providers_failed=("crossref",),
                all_providers_failed=True,
            )

        def lookup_doi(self, doi: str, **kwargs: Any) -> DiscoveryResult:
            raise AssertionError("unexpected DOI lookup")

        def lookup_title(self, title: str, **kwargs: Any) -> DiscoveryResult:
            raise AssertionError("unexpected title lookup")

    monkeypatch.setattr(search, "build_service", lambda _args: _Service())
    exit_code = search.main(["ai governance"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "every requested provider failed" in output
    assert "all_providers_failed" not in output


def test_local_only_disables_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--local-only`` builds a service with no providers at all."""
    monkeypatch.setenv("PHD_RAG_DISCOVERY_CACHE_PATH", str(tmp_path / "cache.json"))
    args = search.parse_args(["ai", "--local-only"])

    service = search.build_service(args)

    assert service.provider_names == ()


def test_provider_option_overrides_the_default_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--provider`` replaces the default provider list."""
    monkeypatch.setenv("PHD_RAG_DISCOVERY_CACHE_PATH", str(tmp_path / "cache.json"))
    args = search.parse_args(["ai", "--provider", "crossref"])

    service = search.build_service(args)

    assert service.provider_names == ("crossref",)


def test_status_counts_summarizes_a_result_set() -> None:
    """The report helper counts each status and local-match kind."""
    counts = render.status_counts(
        [
            _record(),
            _record(
                local_match=LocalMatch(
                    already_in_corpus=True,
                    kind=LocalMatchKind.ALREADY_IN_CORPUS,
                )
            ),
        ]
    )

    assert counts["EXTERNAL_METADATA_VERIFIED"] == 2
    assert counts["local:ALREADY_IN_CORPUS"] == 1
    assert counts["local:NEW"] == 1


def test_fulltext_status_is_rendered_when_present() -> None:
    """Open-access and verification status are visible for each record."""
    rendered = render.format_record(
        _record(
            open_access=True,
            open_access_status="gold",
            fulltext_url="https://example.org/p.pdf",
            fulltext_status=FulltextStatus.FULLTEXT_OPEN,
            cited_by_count=31,
        )
    )

    assert "open access  : yes (gold)" in rendered
    assert "fulltext url : https://example.org/p.pdf" in rendered
    assert "cited by     : 31" in rendered
    assert "not a score" in rendered


def test_unverified_fulltext_explains_why() -> None:
    """A failed verification states the reason instead of looking successful."""
    rendered = render.format_record(
        _record(
            fulltext_status=FulltextStatus.LANDING_PAGE_ONLY,
            fulltext_url="https://example.org/landing",
            fulltext_note="provider did not state an open full text; not fetched",
        )
    )

    assert "fulltext=LANDING_PAGE_ONLY (provider did not state an open full text" in rendered
    assert "fulltext sha" not in rendered


def test_verified_fulltext_shows_the_fetch_evidence() -> None:
    """A verified record shows the hash, size, type and timestamp (inspectable)."""
    rendered = render.format_record(
        _record(
            fulltext_status=FulltextStatus.FULLTEXT_OPEN,
            fulltext_url="https://example.org/p.pdf",
            fulltext_verified=True,
            fulltext_sha256="a" * 64,
            fulltext_bytes=1234,
            fulltext_content_type="application/pdf",
            fulltext_fetched_at="2026-01-01T00:00:00+00:00",
        )
    )

    assert "fulltext=verified" in rendered
    assert f"fulltext sha : {'a' * 64} (1234 bytes, application/pdf" in rendered
