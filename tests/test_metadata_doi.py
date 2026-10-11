"""RAG-006 tests 5-9: DOI detection and normalization."""

from __future__ import annotations

from pathlib import Path

from metadata_support import make_pdf, make_pdf_pages

from app.metadata.doi import find_doi, normalize_doi
from app.metadata.enums import MetadataSource
from app.metadata.extractor import extract_metadata


def test_doi_extracted_from_first_page_text(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("Some ordinary body text on the first page.", 10.0, 120.0),
            ("doi:10.1234/abc.def", 9.0, 160.0),
        ],
    )
    metadata = extract_metadata(path)

    assert metadata.doi is not None
    assert metadata.doi.value == "10.1234/abc.def"
    assert metadata.doi.source is MetadataSource.DOCUMENT_TEXT
    assert metadata.doi.confidence.value == "HIGH"


def test_https_doi_org_normalized() -> None:
    assert normalize_doi("https://doi.org/10.5678/XYZ-123") == "10.5678/XYZ-123"
    assert normalize_doi("http://dx.doi.org/10.5678/XYZ-123") == "10.5678/XYZ-123"


def test_doi_prefix_normalized(tmp_path: Path) -> None:
    assert normalize_doi("doi:10.9999/xyz.1") == "10.9999/xyz.1"
    assert normalize_doi("DOI: 10.9999/xyz.1.") == "10.9999/xyz.1"


def test_invalid_doi_like_string_rejected() -> None:
    assert normalize_doi("10.1234") is None
    assert normalize_doi("10.abcd/xyz") is None
    assert find_doi("see 10.1234 for details") is None


def test_doi_extraction_deterministic() -> None:
    text = "first 10.1111/aaa then 10.2222/bbb"
    assert find_doi(text) == find_doi(text) == "10.1111/aaa"


def test_first_page_doi_wins_over_trailing_text(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        lines=[
            ("Body text with a DOI 10.3333/first here.", 10.0, 120.0),
            ("References mention 10.4444/second as well.", 10.0, 160.0),
        ],
    )
    metadata = extract_metadata(path)
    assert metadata.doi is not None
    assert metadata.doi.value == "10.3333/first"


def test_doi_with_trailing_non_ascii_garbage_trimmed() -> None:
    # A PDF extraction artefact must not be swallowed into the DOI.
    assert find_doi("see 10.1109/cloudnet55617.2022.9978863\ufffd\ufffd") == (
        "10.1109/cloudnet55617.2022.9978863"
    )


def test_reference_list_doi_is_not_adopted(tmp_path: Path) -> None:
    """A DOI that only appears in the reference list is a cited work's DOI."""
    page_one = [
        ("A Prominent First Page Title", 18.0, 80.0),
        ("Ordinary body text on the first page.", 10.0, 130.0),
    ]
    page_two = [
        ("References", 12.0, 60.0),
        (
            "Rets, I., Gillespie, A. (2023). Six Practical Recommendations Enabling "
            "Ethical Use of Predictive Learning Analytics. Journal of Learning Analytics.",
            9.0,
            90.0,
        ),
        ("https://doi.org/10.18608/jla.2023.7743", 9.0, 105.0),
        ("Holmes, W. (2023). Stakeholder Perspectives. IRRODL, 24(2).", 9.0, 140.0),
    ]
    path = make_pdf_pages(tmp_path, "refs.pdf", [page_one, page_two])

    metadata = extract_metadata(path)

    assert metadata.doi is None
    assert metadata.title is not None
    assert metadata.title.value == "A Prominent First Page Title"
