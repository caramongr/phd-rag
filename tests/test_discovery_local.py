"""Local-corpus comparison tests (RAG-010 §30 items 12-15, 22-24).

The important property under test is that an external paper resembling the author's own
work is *not* reported as "already in the corpus as scholarly evidence".
"""

from __future__ import annotations

from app.discovery.local import build_index, local_documents_from_payloads
from app.discovery.models import ExternalWork, LocalMatchKind
from tests.discovery_support import local_index_with, local_payload


def _work(**kwargs: object) -> ExternalWork:
    """An external record with test-friendly defaults."""
    base: dict[str, object] = {
        "title": "A Verified Title",
        "authors": ("Jane Doe",),
        "year": 2019,
        "providers": ("crossref",),
    }
    base.update(kwargs)
    return ExternalWork(**base)  # type: ignore[arg-type]


# --- 12. DOI already in corpus ----------------------------------------------


def test_doi_already_in_corpus_is_local_match() -> None:
    """An external DOI that the corpus already holds is recognized as present."""
    index = local_index_with(doi="10.1234/abc", corpus_role="literature")

    match = index.match(_work(doi="10.1234/abc"))

    assert match.already_in_corpus
    assert match.kind is LocalMatchKind.ALREADY_IN_CORPUS
    assert match.match_reason == "doi_identity"
    assert match.local_corpus_role == "literature"
    assert match.local_evidence_eligible is True


def test_doi_match_ignores_spelling_and_case_differences() -> None:
    """``https://doi.org/10.x/Y`` matches a locally stored ``10.x/y``."""
    index = local_index_with(doi="10.1234/AbC")

    for spelling in ("10.1234/abc", "https://doi.org/10.1234/ABC", "doi:10.1234/abc"):
        match = index.match(_work(doi=spelling))
        assert match.already_in_corpus, spelling


# --- 13. title/year/author identity -----------------------------------------


def test_normalized_identical_title_is_a_local_match() -> None:
    """An exact normalized title identity counts, even without a DOI."""
    index = local_index_with(
        title="Artificial Intelligence in Higher Education: A Systematic Review",
        doi=None,
    )

    match = index.match(
        _work(
            title="Artificial intelligence in higher education - a systematic review",
            year=2019,
            doi=None,
        )
    )

    assert match.already_in_corpus
    assert match.kind is LocalMatchKind.ALREADY_IN_CORPUS
    assert match.match_reason == "normalized_title_identity"
    assert match.year_delta == 0


def test_strong_title_with_year_and_author_agreement_is_a_possible_match() -> None:
    """A strong-but-not-identical title plus consistent year/author is POSSIBLE_MATCH."""
    index = local_index_with(
        title="Generative AI and Assessment in Higher Education",
        year=2021,
        first_author="Jane Doe",
        doi=None,
    )

    match = index.match(
        _work(
            title="Generative AI and Assessment in Higher Education Today",
            year=2021,
            authors=("Jane Doe",),
            doi=None,
        )
    )

    assert match.already_in_corpus
    assert match.kind is LocalMatchKind.POSSIBLE_MATCH
    assert match.match_reason == "title_year_author_consistency"
    assert match.author_consistent is True
    assert match.year_delta == 0
    assert 0.90 <= (match.title_similarity or 0) < 0.97


def test_year_disagreement_prevents_a_strong_title_match() -> None:
    """A nine-year gap means these are not the same document."""
    index = local_index_with(
        title="Generative AI and Assessment in Higher Education",
        year=2015,
        doi=None,
    )

    match = index.match(
        _work(
            title="Generative AI and Assessment in Higher Education Today",
            year=2024,
            doi=None,
        )
    )

    assert not match.already_in_corpus


def test_identical_title_from_a_different_year_is_not_swallowed_as_local() -> None:
    """An identical title nine years apart is a different paper, not 'already here'."""
    index = local_index_with(
        title="Artificial Intelligence in Higher Education: A Systematic Review",
        year=2015,
        doi=None,
    )

    match = index.match(
        _work(
            title="Artificial Intelligence in Higher Education: A Systematic Review",
            year=2024,
            doi=None,
        )
    )

    assert match.already_in_corpus is False
    assert match.kind is LocalMatchKind.NEW


def test_year_tolerance_of_one_is_accepted() -> None:
    """A one-year difference (online-first vs issue year) is within tolerance."""
    index = local_index_with(
        title="Artificial Intelligence in Higher Education: A Systematic Review",
        year=2020,
        doi=None,
    )

    match = index.match(
        _work(
            title="Artificial Intelligence in Higher Education: A Systematic Review",
            year=2021,
            doi=None,
        )
    )

    assert match.already_in_corpus
    assert match.year_delta == 1


def test_author_disagreement_prevents_a_possible_match() -> None:
    """A different first-author family blocks a title-similarity match."""
    index = local_index_with(
        title="Generative AI and Assessment in Higher Education",
        year=2024,
        first_author="Jane Doe",
        doi=None,
    )

    match = index.match(
        _work(
            title="Generative AI and Assessment in Higher Education Today",
            year=2024,
            authors=("Peter Novak",),
            doi=None,
        )
    )

    assert not match.already_in_corpus


# --- 14. genuinely new paper -------------------------------------------------


def test_unrelated_paper_is_new() -> None:
    """A different title/year is a new paper."""
    index = local_index_with(title="Something Completely Else", year=2015)

    match = index.match(
        _work(title="Soil Erosion Modelling in Semi-Arid Regions", year=2024)
    )

    assert match.kind is LocalMatchKind.NEW
    assert match.already_in_corpus is False
    assert match.match_reason == "no_local_match"


# --- 15. own work is not external evidence ----------------------------------


def test_own_work_is_not_reported_as_local_scholarly_evidence() -> None:
    """A DOI hit on the author's own draft is a POSSIBLE_MATCH, not 'already have it'."""
    index = local_index_with(
        doi="10.1234/own",
        filename="second-paper-draft-final.docx",
        corpus_role="my_work",
        evidence_eligible=False,
    )

    match = index.match(_work(doi="10.1234/own"))

    assert match.already_in_corpus is True  # the file is genuinely here
    assert match.kind is LocalMatchKind.POSSIBLE_MATCH  # but it is not evidence
    assert match.local_evidence_eligible is False
    assert match.local_corpus_role == "my_work"
    assert "local role: my_work" in match.match_reason


def test_own_work_title_match_does_not_become_evidence() -> None:
    """A title match against a non-evidence role stays POSSIBLE_MATCH."""
    index = local_index_with(
        title="My Own Chapter On AI Governance",
        corpus_role="future_chapters",
        evidence_eligible=False,
        doi=None,
    )

    match = index.match(_work(title="My Own Chapter On AI Governance", doi=None))

    assert match.kind is LocalMatchKind.POSSIBLE_MATCH
    assert match.local_evidence_eligible is False


def test_non_evidence_role_never_gains_evidence_eligibility() -> None:
    """Evidence eligibility requires both the literature role and the flag."""
    index = build_index(
        [
            local_payload(
                document_id="d1",
                filename="draft.docx",
                corpus_role="my_work",
                evidence_eligible=True,  # inconsistent upstream metadata
                doi="10.1234/own",
            )
        ]
    )

    match = index.match(_work(doi="10.1234/own"))

    assert match.kind is LocalMatchKind.POSSIBLE_MATCH
    assert match.local_evidence_eligible is False


# --- index construction ------------------------------------------------------


def test_index_keeps_one_entry_per_document_id() -> None:
    """Chunk payloads collapse to one document entry."""
    payloads = [
        local_payload(document_id="d1", title="One", doi="10.1234/a"),
        local_payload(document_id="d1", title="One", doi="10.1234/a"),
        local_payload(document_id="d2", title="Two", doi="10.1234/b"),
    ]

    documents = local_documents_from_payloads(payloads)
    index = build_index(payloads)

    assert len(documents) == 2
    assert index.document_count == 2
    assert index.doi_count == 2
    assert index.title_count == 2


def test_payload_without_document_id_is_ignored() -> None:
    """A payload with no document id cannot become a local document."""
    assert local_documents_from_payloads([{"filename": "x.pdf"}]) == []


def test_match_is_deterministic_for_equal_candidates() -> None:
    """Two equally similar candidates always resolve to the same one."""
    payloads = [
        local_payload(
            document_id="d-b",
            title="Artificial Intelligence in Higher Education",
            doi=None,
        ),
        local_payload(
            document_id="d-a",
            title="Artificial Intelligence in Higher Education",
            doi=None,
            file_hash="hash-2",
        ),
    ]
    work = _work(title="Artificial Intelligence in Higher Education", doi=None)

    first = build_index(payloads).match(work)
    second = build_index(list(reversed(payloads))).match(work)

    # The lowest document_id wins, so the answer cannot depend on payload order.
    assert first.local_document_id == "d-a"
    assert first == second
