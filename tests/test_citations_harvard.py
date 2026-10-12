"""Harvard citation and reference list tests (RAG-012 §9-§12, §17)."""

from __future__ import annotations

from app.citations.harvard import (
    MISSING_AUTHOR,
    MISSING_DOI,
    MISSING_TITLE,
    MISSING_YEAR,
    author_label,
    build_reference,
    build_reference_list,
    citation_for_ids,
    citation_map,
    format_reference,
    reference_identity,
    split_author_string,
)
from tests.citations_support import block


def reference_for(**kwargs):
    """One reference entry for a block built with the given metadata."""
    return build_reference(block("S1", **kwargs))


# --- author parsing ------------------------------------------------------------


def test_single_author_in_text_and_reference():
    entry = reference_for(first_author="Jane Doe", year=2024, title="A Paper", doi=None)
    assert entry.in_text == "(Doe, 2024)"
    assert entry.authors_display == "Doe, J."
    assert entry.year_label == "2024"


def test_two_authors_are_joined_with_and():
    entry = reference_for(first_author="Jane Doe and John Smith")
    assert entry.in_text == "(Doe and Smith, 2024)"
    assert entry.authors_display == "Doe, J. and Smith, J."


def test_three_authors_use_et_al_in_text_and_all_names_in_the_reference():
    entry = reference_for(first_author="Jane Doe, John Smith, Anna Lee")
    assert entry.in_text == "(Doe et al., 2024)"
    assert entry.authors_display == "Doe, J., Smith, J. and Lee, A."


def test_inverted_author_strings_are_parsed_as_two_authors():
    entry = reference_for(first_author="Doe, J. and Smith, J.")
    assert entry.in_text == "(Doe and Smith, 2024)"


def test_inverted_single_author_keeps_the_given_name_as_initials():
    entry = reference_for(first_author="Doe, Jane A.")
    assert entry.in_text == "(Doe, 2024)"
    assert entry.authors_display == "Doe, J.A."


def test_existing_et_al_is_preserved():
    entry = reference_for(first_author="Merchant et al.", year=2014)
    assert entry.in_text == "(Merchant et al., 2014)"
    assert entry.authors_display == "Merchant et al."


def test_name_particles_stay_with_the_surname():
    entry = reference_for(first_author="Jan van der Berg")
    assert entry.in_text == "(van der Berg, 2024)"
    assert entry.authors_display == "van der Berg, J."


def test_split_author_string_handles_empty_and_garbage():
    assert split_author_string(None) == ()
    assert split_author_string("   ") == ()
    assert [author.surname for author in split_author_string("Doe")] == ["Doe"]


# --- fallbacks -----------------------------------------------------------------


def test_missing_year_uses_n_d_and_records_the_fallback():
    entry = reference_for(first_author="Jane Doe", year=None)
    assert entry.in_text == "(Doe, n.d.)"
    assert MISSING_YEAR in entry.metadata_warning
    assert "[metadata_warning: missing_year" in entry.formatted


def test_missing_author_falls_back_to_the_title():
    entry = reference_for(first_author=None, title="Adaptive Learning in Higher Education")
    assert entry.in_text == '("Adaptive Learning in Higher Education", 2024)'
    assert MISSING_AUTHOR in entry.metadata_warning
    label, fallbacks, warnings = author_label(
        block("S1", first_author=None, title="Adaptive Learning in Higher Education")
    )
    assert fallbacks == (MISSING_AUTHOR,)
    assert label == '"Adaptive Learning in Higher Education"'
    assert warnings


def test_missing_author_and_title_use_a_defined_placeholder():
    entry = reference_for(first_author=None, title=None)
    assert entry.in_text == "(Untitled work, 2024)"
    assert set(entry.metadata_warning) >= {MISSING_AUTHOR, MISSING_TITLE}


def test_unusable_author_metadata_falls_back_to_the_title():
    entry = reference_for(first_author="ssrn", title="The Impact of Virtual Reality")
    assert entry.in_text == '("The Impact of Virtual Reality", 2024)'
    assert MISSING_AUTHOR in entry.metadata_warning
    _, _fallbacks, warnings = author_label(
        block("S1", first_author="ssrn", title="The Impact of Virtual Reality")
    )
    assert any("unusable" in warning for warning in warnings)


def test_single_letter_author_metadata_is_rejected():
    assert split_author_string("a") == ()
    assert split_author_string("J") == ()
    assert [author.surname for author in split_author_string("Doe")] == ["Doe"]


def test_reference_without_an_author_puts_the_title_first():
    entry = reference_for(
        first_author=None,
        title="Information Technology Students' Perceptions",
        doi="10.3991/ijim.v17i07.37211",
        pages="6-7",
    )
    assert entry.formatted.startswith(
        "\"Information Technology Students' Perceptions\" (2024)"
    )
    assert "doi:10.3991/ijim.v17i07.37211" in entry.formatted


def test_missing_doi_is_recorded_as_a_metadata_warning():
    entry = reference_for(doi=None)
    assert MISSING_DOI in entry.metadata_warning
    assert entry.identity_kind == "title"


def test_doi_is_normalized_to_its_canonical_form():
    entry = reference_for(doi="https://doi.org/10.1234/AbC-d4")
    assert entry.doi == "10.1234/AbC-d4"
    assert entry.identity_kind == "doi"
    assert "doi:10.1234/AbC-d4" in entry.formatted


def test_unusable_doi_is_reported_rather_than_used():
    entry = reference_for(doi="not-a-doi")
    assert entry.doi is None
    assert entry.identity_kind == "title"
    assert MISSING_DOI in entry.metadata_warning


# --- identity and deduplication -------------------------------------------------


def test_chunks_of_one_paper_collapse_into_one_reference():
    blocks = [
        block("S1", doi="10.1234/shared", pages="4-6", chunk_id="chunk-a"),
        block("S2", doi="https://doi.org/10.1234/SHARED", pages="7-9", chunk_id="chunk-b"),
        block("S3", doi="10.1234/shared", pages="10", chunk_id="chunk-c"),
    ]
    entries = build_reference_list(blocks)
    assert len(entries) == 1
    assert entries[0].source_ids == ("S1", "S2", "S3")
    assert entries[0].chunk_ids == ("chunk-a", "chunk-b", "chunk-c")
    assert entries[0].pages == "4-6"


def test_chunks_of_one_paper_without_a_doi_collapse_by_title():
    blocks = [
        block("S1", doi=None, title="A Paper About AI Governance", chunk_id="chunk-a"),
        block("S2", doi=None, title="A Paper about AI Governance ", chunk_id="chunk-b"),
    ]
    entries = build_reference_list(blocks)
    assert len(entries) == 1
    assert entries[0].identity_kind == "title"


def test_different_works_stay_separate():
    blocks = [
        block("S1", doi="10.1234/a", title="First"),
        block("S2", doi="10.1234/b", title="Second"),
    ]
    assert len(build_reference_list(blocks)) == 2


def test_reference_identity_prefers_doi_then_title_then_document():
    assert reference_identity(block("S1", doi="10.1234/a"))[1] == "doi"
    assert reference_identity(block("S1", doi=None, title="A Title"))[1] == "title"
    assert reference_identity(block("S1", doi=None, title=None))[1] == "document"


# --- ordering and formatting ---------------------------------------------------


def test_citation_order_is_alphabetical_then_by_year():
    blocks = [
        block("S1", first_author="Zoe Adams", year=2020, doi="10.1234/z"),
        block("S2", first_author="Bob Chan", year=2024, doi="10.1234/c"),
        block("S3", first_author="Bob Chan", year=2019, doi="10.1234/c2"),
    ]
    entries = build_reference_list(blocks)
    assert [entry.author_label for entry in entries] == [
        "Adams",
        "Chan",
        "Chan",
    ]
    assert [entry.year for entry in entries] == [2020, 2019, 2024]


def test_multi_source_in_text_citation_is_deterministically_ordered():
    blocks = [
        block("S7", first_author="Sara Smith", year=2023, doi="10.1234/s"),
        block("S1", first_author="David Brown", year=2022, doi="10.1234/b"),
        block("S4", first_author="Chris Chan", year=2024, doi="10.1234/c"),
    ]
    entries = build_reference_list(blocks)
    by_source = {source_id: entry for entry in entries for source_id in entry.source_ids}
    assert citation_for_ids(["S7", "S1", "S4"], by_source) == (
        "(Brown, 2022; Chan, 2024; Smith, 2023)"
    )


def test_two_markers_of_one_work_are_cited_once():
    blocks = [
        block("S1", doi="10.1234/shared", pages="3-4"),
        block("S2", doi="10.1234/shared", pages="5-6"),
    ]
    entries = build_reference_list(blocks)
    by_source = {source_id: entry for entry in entries for source_id in entry.source_ids}
    assert citation_for_ids(["S1", "S2"], by_source) == "(Doe, 2024)"


def test_citation_for_unknown_ids_is_empty():
    assert citation_for_ids(["S9"], {}) == ""


def test_page_formatting():
    assert "pp. 10-25" in format_reference(reference_for(pages="10-25"))
    assert "p. 7" in format_reference(reference_for(pages="7"))
    assert "pp." not in format_reference(reference_for(pages=None))


def test_journal_style_reference_uses_only_available_metadata():
    entry = reference_for(
        first_author="Jane Doe and John Smith",
        year=2024,
        title="Governing AI in universities",
        venue="Journal of Higher Education Policy",
        pages="10-25",
        doi="10.1234/abc",
    )
    assert entry.formatted == (
        "Doe, J. and Smith, J. (2024) 'Governing AI in universities', "
        "Journal of Higher Education Policy, pp. 10-25. doi:10.1234/abc"
    )


def test_reference_without_a_venue_or_doi_stays_safe():
    entry = reference_for(
        first_author="Jane Doe",
        year=2024,
        title="A Paper About AI Governance",
        venue=None,
        pages="4-6",
        doi=None,
    )
    assert entry.formatted == (
        "Doe, J. (2024) 'A Paper About AI Governance', pp. 4-6. "
        "[metadata_warning: missing_doi]"
    )


def test_no_volume_issue_or_publisher_is_ever_invented():
    formatted = reference_for(venue=None, doi="10.1234/abc").formatted
    for invented in ("publisher", "vol.", "no.", "Issue", "Available at"):
        assert invented not in formatted


def test_citation_map_records_the_source_chunk_and_doi():
    context_map = {
        "S1": {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "title": "A Paper",
        }
    }
    entries = build_reference_list([block("S1", doi="10.1234/abc")])
    by_source = {source_id: entry for entry in entries for source_id in entry.source_ids}
    mapped = citation_map(context_map, by_source)
    assert [entry.source_id for entry in mapped] == ["S1"]
    assert mapped[0].in_text == "(Doe, 2024)"
    assert mapped[0].chunk_id == "chunk-1"
    assert mapped[0].reference_key == "doi:10.1234/abc"
    assert mapped[0].doi == "10.1234/abc"
