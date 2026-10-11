"""RAG-006: DOCX, Markdown, filename and arXiv local extraction."""

from __future__ import annotations

from pathlib import Path

from metadata_support import make_docx, make_markdown, make_pdf, utc

from app.metadata.enums import CorpusRole, MetadataSource, PublicationType
from app.metadata.extractor import extract_metadata
from app.metadata.identifiers import normalize_arxiv_id, parse_filename_metadata


def test_docx_core_properties_extracted(tmp_path: Path) -> None:
    path = make_docx(
        tmp_path,
        title="A Thesis Draft Title",
        author="John Smith",
        created=utc(2021),
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None
    assert metadata.title.value == "A Thesis Draft Title"
    assert metadata.title.source is MetadataSource.DOCX_PROPERTIES
    assert metadata.authors is not None
    assert metadata.authors.value[0].full_name == "John Smith"
    assert metadata.year is not None and metadata.year.value == 2021
    assert metadata.year.source is MetadataSource.DOCX_PROPERTIES


def test_docx_author_manuscript_publication_type(tmp_path: Path) -> None:
    path = make_docx(tmp_path, title="Current Paper Draft", author="Jane Doe")
    metadata = extract_metadata(path, corpus_role=CorpusRole.MY_WORK)

    assert metadata.publication_type is not None
    assert metadata.publication_type.value is PublicationType.AUTHOR_MANUSCRIPT


def test_markdown_title_extracted_without_text_doi(tmp_path: Path) -> None:
    path = make_markdown(
        tmp_path,
        text="# A Markdown Title\n\nSome body text with doi:10.7777/md.1 here.\n",
    )
    metadata = extract_metadata(path)

    assert metadata.title is not None and metadata.title.value == "A Markdown Title"
    # A DOI in DOCX/Markdown text is not adopted: unlike a PDF's first page, the
    # body and bibliography cannot be told apart, so a cited DOI must not be
    # mistaken for the document's own.
    assert metadata.doi is None


def test_markdown_reference_doi_not_adopted(tmp_path: Path) -> None:
    path = make_markdown(
        tmp_path,
        text=(
            "# A Markdown Title\n\n"
            "Some body text here.\n\n"
            "## References\n\n"
            "Doe, J. (2020). A cited work. doi:10.9999/cited.1\n"
        ),
    )
    metadata = extract_metadata(path)

    assert metadata.doi is None


def test_docx_reference_doi_not_adopted(tmp_path: Path) -> None:
    path = make_docx(
        tmp_path,
        title="A Thesis Draft Title",
        author="John Smith",
        paragraphs=[
            "Body text referring to a cited work.",
            "References",
            "Doe, J. (2020). A cited work. doi:10.9999/cited.1",
        ],
    )
    metadata = extract_metadata(path, corpus_role=CorpusRole.MY_WORK)

    assert metadata.doi is None
    assert metadata.publication_type is not None
    assert metadata.publication_type.value is PublicationType.AUTHOR_MANUSCRIPT


def test_filename_hints_arxiv_and_year() -> None:
    arxiv = parse_filename_metadata("Zhu-1703.10593v7.pdf")
    assert arxiv.arxiv_id == "1703.10593"
    assert arxiv.author_token == "Zhu"
    assert arxiv.years == []

    named = parse_filename_metadata("kerbl-2023.pdf")
    assert named.years == [2023]
    assert named.author_token == "kerbl"
    assert named.arxiv_id is None


def test_normalize_arxiv_id() -> None:
    assert normalize_arxiv_id("1703.10593v7") == "1703.10593"
    assert normalize_arxiv_id("arXiv:2308.04079") == "2308.04079"
    assert normalize_arxiv_id("not an id") is None


def test_pdf_filename_arxiv_identifier_preserved(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        name="Zhu-1703.10593v7.pdf",
        lines=[("Ordinary body text only on this page.", 10.0, 120.0)],
    )
    metadata = extract_metadata(path)

    assert metadata.arxiv_id is not None
    assert metadata.arxiv_id.value == "1703.10593"
    assert metadata.arxiv_id.source is MetadataSource.FILENAME
    assert metadata.arxiv_id.confidence.value == "LOW"


def test_filename_never_becomes_title(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path,
        name="kerbl-2023.pdf",
        lines=[("Ordinary body text only on this page.", 10.0, 120.0)],
    )
    metadata = extract_metadata(path)

    assert metadata.title is None
    assert metadata.year is not None and metadata.year.value == 2023
    assert metadata.year.source is MetadataSource.FILENAME
