"""RAG-006 tests 10-13: conservative first-page title candidate extraction."""

from __future__ import annotations

from pathlib import Path

from metadata_support import make_pdf

from app.metadata.enums import MetadataSource
from app.metadata.extractor import extract_metadata
from app.metadata.titles import is_meaningful_title


def test_strong_first_page_title_identified(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("A Prominent First Page Title", 20.0, 80.0),
            ("Ordinary body text follows the title here.", 10.0, 130.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "A Prominent First Page Title"
    assert metadata.title.source is MetadataSource.DOCUMENT_TEXT


def test_journal_header_not_selected_as_title(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("IEEE Transactions on Visualization, Vol. 25, No. 5, pp. 1-10", 9.0, 60.0),
            ("Towards Foveated Rendering for Gaze-Tracked VR", 18.0, 100.0),
            ("Ordinary body text follows the title here.", 10.0, 150.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "Towards Foveated Rendering for Gaze-Tracked VR"


def test_doi_line_not_selected_as_title(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("https://doi.org/10.1000/xyz.123", 16.0, 80.0),
            ("Ordinary body text follows here.", 10.0, 140.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is None


def test_ambiguous_candidate_remains_unresolved(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("First Possible Title Of The Work", 18.0, 80.0),
            ("small body line", 10.0, 95.0),
            ("Second Possible Title Of The Work", 18.0, 110.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is None


def test_legitimate_short_title_not_over_cleaned() -> None:
    assert is_meaningful_title("Deep Learning")
    assert is_meaningful_title("On the Origin of Species")


def test_artefact_titles_rejected() -> None:
    assert not is_meaningful_title("untitled")
    assert not is_meaningful_title("Microsoft Word - document1.docx")
    assert not is_meaningful_title("Adobe Acrobat")


def test_title_not_merged_with_author_line(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("A Prominent First Page Title", 16.0, 80.0),
            ("Jane Doe and John Smith", 15.0, 105.0),
            ("Ordinary body text follows the title here.", 10.0, 150.0),
            ("More body text continues on this page.", 10.0, 170.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "A Prominent First Page Title"


def test_two_line_title_joined(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("A Very Long Title That Wraps", 16.0, 80.0),
            ("Across Two Lines", 16.0, 100.0),
            ("Ordinary body text follows the title here.", 10.0, 140.0),
            ("More body text continues on this page.", 10.0, 160.0),
            ("Even more body text follows here.", 10.0, 180.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "A Very Long Title That Wraps Across Two Lines"


def test_arxiv_margin_stamp_not_selected_as_title(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("arXiv:1910.10045v2 [cs.AI] 26 Dec 2019", 14.0, 60.0),
            ("Ordinary body text follows here.", 10.0, 120.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is None


# --- title-extraction correction: artefact rejection classes (RAG-006) -------


def test_journal_masthead_above_article_title_is_rejected(tmp_path: Path) -> None:
    article = "Learning Analytics for Primary and Secondary Schools"
    path = make_pdf(
        tmp_path,
        lines=[
            ("Computers & Education", 14.0, 50.0),
            ("Volume 150, March 2020, 103778", 9.0, 74.0),
            (article, 16.0, 120.0),
            ("Ordinary body text follows here.", 10.0, 200.0),
            ("More body text continues on this page.", 10.0, 220.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == article


def test_education_sciences_style_masthead_rejected(tmp_path: Path) -> None:
    article = "Adaptive Learning in E-Learning: A Literature Review"
    path = make_pdf(
        tmp_path,
        lines=[
            ("education sciences", 14.0, 50.0),
            ("Volume 13, Issue 12, 1216", 9.0, 74.0),
            (article, 16.0, 130.0),
            ("Ordinary body text follows here.", 10.0, 200.0),
            ("More body text continues on this page.", 10.0, 220.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == article


def test_production_identifiers_rejected() -> None:
    for text in (
        "DSJ_2300028 1..24",
        "APC873088 1..17",
        "feduc-2021-645758 1..9",
        "JMI-18219SSR 11..11 ++",
        "1 - 7543-Article Text-35739-1-11-20210714 - PROOF II",
    ):
        assert not is_meaningful_title(text), text


def test_template_labels_rejected() -> None:
    assert not is_meaningful_title("Proceedings Template - WORD")
    assert not is_meaningful_title("SIGCHI Conference Proceedings Format")


def test_reprint_notice_rejected() -> None:
    assert not is_meaningful_title(
        "This is an electronic reprint of the original article. This reprint may "
        "differ from the original in pagination and typographic detail."
    )


def test_actual_title_selected_after_false_candidate_rejected(tmp_path: Path) -> None:
    article = "A Real Article Title About Design Science"
    path = make_pdf(
        tmp_path,
        lines=[
            ("DSJ_2300028 1..24", 15.0, 50.0),
            (article, 17.0, 110.0),
            ("Ordinary body text follows here.", 10.0, 180.0),
            ("More body text continues on this page.", 10.0, 200.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == article


def test_legitimate_short_title_not_rejected(tmp_path: Path) -> None:
    assert is_meaningful_title("Deep Learning")
    path = make_pdf(
        tmp_path,
        lines=[
            ("Deep Learning", 18.0, 80.0),
            ("Volume 3, Issue 1", 9.0, 60.0),
            ("Ordinary body text follows here.", 10.0, 200.0),
            ("More body text continues on this page.", 10.0, 220.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "Deep Learning"


def test_title_with_journal_like_words_not_rejected_by_vocabulary(tmp_path: Path) -> None:
    article = "Research on Applied Sciences Education"
    assert is_meaningful_title(article)
    path = make_pdf(
        tmp_path,
        lines=[
            (article, 18.0, 80.0),
            ("Volume 5, Issue 2", 9.0, 60.0),
            ("Ordinary body text follows here.", 10.0, 200.0),
            ("More body text continues on this page.", 10.0, 220.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == article


def test_local_title_extraction_is_offline_and_deterministic(tmp_path: Path, monkeypatch) -> None:
    import socket

    def _fail(*args, **kwargs):
        raise AssertionError("local title extraction must not use the network")

    monkeypatch.setattr(socket, "create_connection", _fail)

    path = make_pdf(
        tmp_path,
        lines=[
            ("education sciences", 14.0, 50.0),
            ("Volume 13, Issue 12, 1216", 9.0, 74.0),
            ("A Genuine Article Title In The Journal", 16.0, 130.0),
            ("Ordinary body text follows here.", 10.0, 200.0),
            ("More body text continues on this page.", 10.0, 220.0),
        ],
    )

    first = extract_metadata(path)
    second = extract_metadata(path)

    assert first.title is not None
    assert first.title.value == "A Genuine Article Title In The Journal"
    assert first.title.model_dump() == second.title.model_dump()


def test_embedded_venue_masthead_title_discarded_for_page_title(tmp_path: Path) -> None:
    """A journal name stamped into PDF metadata must not beat the real title."""
    path = make_pdf(
        tmp_path,
        metadata={"title": "Journal of Applied Learning and Teaching"},
        lines=[
            ("Machine Learning Support for Distance Learners", 17.0, 90.0),
            ("Ordinary body text follows the title here.", 10.0, 150.0),
            ("More body text continues on this page.", 10.0, 170.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "Machine Learning Support for Distance Learners"
    assert all(
        candidate.value != "Journal of Applied Learning and Teaching"
        for candidate in metadata.candidates["title"]
    )


def test_embedded_title_kept_without_page_title_candidate(tmp_path: Path) -> None:
    """Without a page-one title the embedded metadata title is still used."""
    path = make_pdf(
        tmp_path,
        metadata={"title": "Journal of Applied Learning and Teaching"},
        lines=[
            ("Ordinary body text follows here.", 10.0, 120.0),
            ("More body text continues on this page.", 10.0, 140.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "Journal of Applied Learning and Teaching"


def test_author_line_above_title_is_not_part_of_title(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("Jane Doe, John Smith, Ann Lee & Peter Hall", 15.0, 80.0),
            ("A Real Article Title Follows Here", 16.0, 100.0),
            ("Ordinary body text follows the title here.", 10.0, 150.0),
            ("More body text continues on this page.", 10.0, 170.0),
            ("Even more body text follows here.", 10.0, 190.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "A Real Article Title Follows Here"
    assert metadata.authors is not None
    assert [author.full_name for author in metadata.authors.value] == [
        "Jane Doe",
        "John Smith",
        "Ann Lee",
        "Peter Hall",
    ]


def test_all_caps_author_names_accepted_below_title(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("A Prominent First Page Title", 20.0, 80.0),
            ("MUHAMMAD ASIF HASAN", 10.0, 130.0),
            ("Department of Computer Science", 7.0, 150.0),
            ("Ordinary body text follows here.", 8.0, 200.0),
            ("More body text continues on this page.", 8.0, 220.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.authors is not None
    assert [author.full_name for author in metadata.authors.value] == ["MUHAMMAD ASIF HASAN"]


def test_repository_boilerplate_not_treated_as_authors(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("A Prominent First Page Title", 20.0, 80.0),
            ("Document Version", 10.0, 130.0),
            ("Peer reviewed version", 10.0, 145.0),
            ("Ordinary body text follows here.", 8.0, 200.0),
            ("More body text continues on this page.", 8.0, 220.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.authors is None
