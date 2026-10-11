"""Deduplication, provenance, conflicts and determinism (RAG-010 §30 items 8-11, 29-30)."""

from __future__ import annotations

from app.discovery.merge import (
    identity_confidence,
    merge_conflicts,
    merge_records,
    same_work,
    status_for,
)
from app.discovery.models import DiscoveryStatus, ExternalWork, FieldConflict


def _work(**kwargs: object) -> ExternalWork:
    """An external record with test-friendly defaults."""
    base: dict[str, object] = {
        "title": "Artificial Intelligence in Higher Education",
        "authors": ("Jane Doe",),
        "year": 2021,
        "doi": None,
        "providers": ("crossref",),
    }
    base.update(kwargs)
    return ExternalWork(**base)  # type: ignore[arg-type]


# --- 8. same DOI from two providers merges ----------------------------------


def test_same_doi_from_two_providers_merges_into_one_record() -> None:
    """Crossref and OpenAlex records for one DOI become one record with both providers."""
    crossref = _work(
        doi="10.1234/abc",
        title="Artificial Intelligence in Higher Education",
        venue="Journal of AI",
        providers=("crossref",),
    )
    openalex = _work(
        doi="https://doi.org/10.1234/ABC",
        title="Artificial intelligence in higher education",
        cited_by_count=31,
        open_access=True,
        providers=("openalex",),
    )

    merged, merged_away = merge_records([crossref, openalex])

    assert merged_away == 1
    assert len(merged) == 1
    record = merged[0]
    assert record.providers == ("crossref", "openalex")
    assert record.venue == "Journal of AI"
    assert record.cited_by_count == 31
    assert record.open_access is True
    # Provenance says who supplied each field; nothing is attributed to a guess.
    assert record.field_provenance["venue"] == ("crossref",)
    assert record.field_provenance["cited_by_count"] == ("openalex",)
    assert set(record.field_provenance["title"]) == {"crossref", "openalex"}


# --- 9. equivalent DOI forms merge ------------------------------------------


def test_equivalent_doi_url_and_plain_forms_merge() -> None:
    """``10.x/y``, ``https://doi.org/10.x/y`` and ``doi:10.x/y`` are one DOI."""
    forms = [
        "10.1234/abc",
        "https://doi.org/10.1234/abc",
        "http://dx.doi.org/10.1234/abc",
        "doi:10.1234/ABC",
    ]
    records = [
        _work(doi=form, providers=(f"provider{index}",), title=f"Title {index}")
        for index, form in enumerate(forms)
    ]

    merged, merged_away = merge_records(records)

    assert merged_away == 3
    assert len(merged) == 1
    assert merged[0].doi == "10.1234/abc"


def test_different_dois_never_merge_even_with_identical_titles() -> None:
    """Two different DOIs are two different papers."""
    left = _work(doi="10.1/a")
    right = _work(doi="10.1/b")

    assert not same_work(left, right)
    merged, merged_away = merge_records([left, right])
    assert merged_away == 0
    assert len(merged) == 2


# --- 10. strong same-title merge when DOI is absent -------------------------


def test_strongly_identical_titles_merge_without_a_doi() -> None:
    """A near-identical normalized title is enough to merge when no DOI exists."""
    left = _work(title="Artificial Intelligence in Higher Education", providers=("crossref",))
    right = _work(
        title="Artificial  intelligence in higher   education",
        providers=("openalex",),
    )

    merged, merged_away = merge_records([left, right])

    assert merged_away == 1
    assert len(merged) == 1


def test_merely_similar_titles_do_not_merge() -> None:
    """A different paper with a similar title stays a separate record."""
    left = _work(title="Artificial Intelligence in Higher Education")
    right = _work(title="Artificial Intelligence in Medical Education")

    assert not same_work(left, right)
    merged, _ = merge_records([left, right])
    assert len(merged) == 2


# --- 11. materially conflicting records remain auditable --------------------


def test_materially_conflicting_titles_stay_conflict_and_keep_both_values() -> None:
    """Two providers describing the same DOI with different titles is a CONFLICT."""
    crossref = _work(doi="10.1/x", title="Deep Learning for Diagnostics", providers=("crossref",))
    openalex = _work(
        doi="10.1/x",
        title="A Completely Different Study of Soil Erosion",
        providers=("openalex",),
    )

    merged, _ = merge_records([crossref, openalex])

    assert len(merged) == 1
    record = merged[0]
    assert record.discovery_status is DiscoveryStatus.CONFLICT
    assert len(record.conflicts) == 1
    conflict = record.conflicts[0]
    assert conflict.field_name == "title"
    assert set(conflict.values) == {
        "Deep Learning for Diagnostics",
        "A Completely Different Study of Soil Erosion",
    }
    assert set(conflict.providers) == {"crossref", "openalex"}


def test_conflicting_years_beyond_tolerance_are_preserved() -> None:
    """A two-year disagreement is material and both years are recorded."""
    left = _work(doi="10.1/x", year=2019, providers=("crossref",))
    right = _work(doi="10.1/x", year=2023, providers=("openalex",))

    merged, _ = merge_records([left, right])

    record = merged[0]
    assert record.discovery_status is DiscoveryStatus.CONFLICT
    assert [conflict.field_name for conflict in record.conflicts] == ["year"]
    assert set(record.conflicts[0].values) == {"2019", "2023"}


def test_conflicting_first_author_families_are_preserved() -> None:
    """Different first-author families for one DOI are recorded, not averaged."""
    left = _work(doi="10.1/x", authors=("Jane Doe",), providers=("crossref",))
    right = _work(doi="10.1/x", authors=("John Smith",), providers=("openalex",))

    merged, _ = merge_records([left, right])

    record = merged[0]
    assert record.discovery_status is DiscoveryStatus.CONFLICT
    assert [conflict.field_name for conflict in record.conflicts] == ["authors"]
    assert set(record.conflicts[0].values) == {"doe", "smith"}


def test_minor_title_punctuation_differences_are_not_a_conflict() -> None:
    """Punctuation and spacing variation is not a material disagreement."""
    left = _work(
        doi="10.1/x",
        title="Artificial Intelligence: A Review of the Literature",
        providers=("crossref",),
    )
    right = _work(
        doi="10.1/x",
        title="Artificial intelligence - a review of the literature",
        providers=("openalex",),
    )

    merged, _ = merge_records([left, right])

    assert merged[0].conflicts == ()
    assert merged[0].discovery_status is not DiscoveryStatus.CONFLICT


# --- 29-30. determinism ------------------------------------------------------


def test_same_responses_produce_the_same_merged_order() -> None:
    """Merging is a pure function of the input set."""
    records = [
        _work(doi="10.1/b", title="Beta Study"),
        _work(doi="10.1/a", title="Alpha Study"),
        _work(doi="10.1/c", title="Gamma Study"),
    ]

    first, _ = merge_records(list(records))
    second, _ = merge_records(list(records))

    assert [record.doi for record in first] == [record.doi for record in second]


def test_provider_response_order_does_not_change_identity_grouping() -> None:
    """Reversing the input order gives the same groups, fields and order."""
    forward = [
        _work(doi="10.1/a", venue="Venue A", providers=("crossref",)),
        _work(doi="10.1/a", cited_by_count=5, providers=("openalex",)),
        _work(doi="10.1/b", title="Second Paper", providers=("crossref",)),
    ]
    backward = list(reversed(forward))

    merged_forward, away_forward = merge_records(forward)
    merged_backward, away_backward = merge_records(backward)

    assert away_forward == away_backward == 1
    assert [record.model_dump() for record in merged_forward] == [
        record.model_dump() for record in merged_backward
    ]


def test_merge_conflicts_wrapper_is_also_order_independent() -> None:
    """The DOI-lookup variant has the same property."""
    records = [
        _work(doi="10.1/a", providers=("openalex",)),
        _work(doi="10.1/a", providers=("crossref",)),
    ]

    forward, _ = merge_conflicts(records)
    backward, _ = merge_conflicts(list(reversed(records)))

    assert forward[0].model_dump() == backward[0].model_dump()
    assert forward[0].providers == ("crossref", "openalex")


# --- identity confidence / status -------------------------------------------


def test_identity_confidence_rewards_agreement() -> None:
    """Two providers agreeing on DOI, title, authors and year is high confidence."""
    strong = _work(
        doi="10.1/x",
        providers=("crossref", "openalex"),
        field_provenance={
            "doi": ("crossref", "openalex"),
            "title": ("crossref", "openalex"),
            "authors": ("crossref", "openalex"),
            "year": ("crossref", "openalex"),
        },
    )
    weak = ExternalWork(providers=("crossref",), title="Some title")

    assert identity_confidence(strong) > identity_confidence(weak)
    assert 0.0 <= identity_confidence(weak) <= 1.0


def test_status_requires_verification_evidence() -> None:
    """A discovery-only record never claims to be verified (§30 items 16-17)."""
    plain = _work(doi=None, title=None, authors=(), year=None)
    plain = plain.model_copy(update={"title": "An Unverified Title"})
    assert status_for(plain)[0] is DiscoveryStatus.EXTERNAL_DISCOVERY

    verified_metadata = _work(
        doi="10.1/x",
        providers=("crossref", "openalex"),
        venue="Journal of AI",
        publication_type="journal-article",
    )
    status, confidence = status_for(verified_metadata)

    assert status is DiscoveryStatus.EXTERNAL_METADATA_VERIFIED
    assert confidence >= 0.80


def test_record_without_title_or_doi_is_unavailable() -> None:
    """A record with no identity at all is reported as unavailable."""
    empty = ExternalWork(providers=("crossref",), title=None, doi=None)

    assert status_for(empty)[0] is DiscoveryStatus.UNAVAILABLE


def test_conflict_status_is_preserved_by_apply_status() -> None:
    """A conflict is never silently downgraded to a verified status."""
    conflicting = ExternalWork(
        doi="10.1/x",
        title="T",
        providers=("crossref", "openalex"),
        conflicts=(
            FieldConflict(
                field_name="year",
                values=("2019", "2023"),
                providers=("crossref", "openalex"),
            ),
        ),
    )

    status, confidence = status_for(conflicting)
    assert status is DiscoveryStatus.CONFLICT
    # Confidence and conflict are independent facts: a conflict is not "low confidence",
    # it is a known disagreement that the caller must see.
    assert confidence > 0.0


def test_doi_lookup_route_raises_confidence() -> None:
    """A direct DOI lookup is stronger evidence of identity than a search hit."""
    work = _work(doi="10.1/x", title=None, authors=(), year=None)

    assert identity_confidence(work, doi_lookup=True) > identity_confidence(work)
