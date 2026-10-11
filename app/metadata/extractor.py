"""Local academic-metadata extraction (RAG-006).

This module turns a single file into an :class:`AcademicMetadata` object using
only local information, following a deterministic priority chain: structured
embedded metadata, DOI detection, conservative text extraction, and finally weak
filename hints. External verification is a separate, opt-in step handled by
:mod:`app.metadata.enrichment`.

Nothing is fabricated: when no evidence supports a field the field is left
missing, which is a valid state.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from app.metadata.authors import (
    extract_authors_from_first_page,
    looks_like_affiliation,
    parse_author_string,
)
from app.metadata.corpus import (
    ROLE_PRECEDENCE,
    corpus_role_for_path,
    is_enrichable,
    resolve_corpus_role,
)
from app.metadata.docx_reader import read_docx_properties
from app.metadata.doi import find_doi
from app.metadata.enums import (
    Confidence,
    CorpusRole,
    MetadataSource,
    PublicationType,
)
from app.metadata.identifiers import find_arxiv_id, parse_filename_metadata
from app.metadata.layout import PageLayout
from app.metadata.models import (
    AcademicMetadata,
    Author,
    MetadataValue,
    PhysicalRole,
    RoleResolution,
)
from app.metadata.pdf_reader import read_pdf
from app.metadata.resolver import add_candidate, resolve_all
from app.metadata.text import collapse_whitespace
from app.metadata.titles import (
    extract_first_page_title,
    is_meaningful_title,
    looks_like_venue_masthead,
)
from app.metadata.year import (
    extract_year_from_text,
    year_from_datetime,
    year_from_pdf_date,
)
from app.parsing.docx_parser import parse_docx
from app.parsing.exceptions import UnsupportedDocumentFormat
from app.parsing.markdown_parser import parse_markdown
from app.parsing.models import ContentItemType, ParsedBlockType, ParsedDocument

#: A short line that marks the start of a bibliography in DOCX/Markdown.
_REFERENCES_LABEL_RE = re.compile(
    r"^\s*(references|bibliography|works cited|reference list|literature cited|"
    r"references cited|citations|bibliographical references)\s*:?\s*$",
    re.IGNORECASE,
)
_MAX_REFERENCE_LABEL_CHARS = 60


def _body_text(parsed: ParsedDocument) -> str:
    """Return document text *before* any reference list / bibliography.

    DOIs (and arXiv ids) in a bibliography belong to cited works, not to this
    document, so text-based identifier extraction must stop at the reference
    boundary rather than adopt a cited work's identifier. Tables are included
    because they are part of the body.
    """
    parts: list[str] = []
    for item in parsed.content_items:
        if item.item_type is ContentItemType.BLOCK and item.block is not None:
            text = item.block.text or ""
            if len(text) <= _MAX_REFERENCE_LABEL_CHARS and _REFERENCES_LABEL_RE.match(text):
                break
            parts.append(text)
        elif item.item_type is ContentItemType.TABLE and item.table is not None:
            for row in item.table.rows:
                parts.append(" ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_metadata(
    path: str | Path,
    *,
    file_hash: str | None = None,
    corpus_role: CorpusRole | None = None,
    physical_paths: Sequence[str | Path] | None = None,
) -> AcademicMetadata:
    """Extract local academic metadata for one file.

    The returned object carries the resolved value per field plus the full
    candidate history. No external service is contacted here.

    ``physical_paths`` are *all* known locations of the same logical document
    (identical SHA-256 bytes), including ``path`` itself. When several folders
    imply different roles, the most conservative role wins so a duplicate under
    ``literature`` cannot turn the author's own work into scholarly evidence; see
    :func:`~app.metadata.corpus.resolve_corpus_role`. An explicit ``corpus_role``
    overrides the resolution entirely.
    """
    path = Path(path)
    extension = path.suffix.lower()
    locations: list[str | Path] = list(physical_paths) if physical_paths else [path]
    if corpus_role is not None:
        resolution = RoleResolution(
            effective_role=corpus_role,
            reason="explicit corpus_role override",
            physical_roles=[
                PhysicalRole(path=str(location), role=corpus_role_for_path(location))
                for location in locations
            ],
            applied_precedence=list(ROLE_PRECEDENCE),
        )
    else:
        resolution = resolve_corpus_role(locations)
    role = resolution.effective_role

    candidates: dict[str, list[MetadataValue]] = {}
    page_count: int | None = None

    if extension == ".pdf":
        page_count = _extract_pdf(path, candidates, role)
    elif extension == ".docx":
        _extract_docx(path, candidates, role)
    elif extension == ".md":
        _extract_markdown(path, candidates, role)
    elif extension:
        # Unsupported-but-known format: filename hints only.
        _extract_filename(path, candidates)
    else:
        raise UnsupportedDocumentFormat(f"Unsupported document format for: {path}")

    resolved = resolve_all(candidates)
    metadata = AcademicMetadata(
        file_hash=file_hash,
        filename=path.name,
        source_path=str(path),
        extension=extension,
        corpus_role=role,
        role_resolution=resolution,
        title=resolved.get("title"),
        authors=resolved.get("authors"),
        year=resolved.get("year"),
        doi=resolved.get("doi"),
        venue=resolved.get("venue"),
        publisher=resolved.get("publisher"),
        publication_type=resolved.get("publication_type"),
        url=resolved.get("url"),
        abstract=resolved.get("abstract"),
        arxiv_id=resolved.get("arxiv_id"),
        candidates=candidates,
    )
    return metadata


def _add(
    candidates: dict[str, list[MetadataValue]],
    field_name: str,
    value,
    source: MetadataSource,
    confidence: Confidence,
    evidence: str,
) -> None:
    if value is None:
        return
    add_candidate(
        candidates,
        field_name,
        MetadataValue(value=value, source=source, confidence=confidence, evidence=evidence),
    )


def _add_doi(
    candidates: dict[str, list[MetadataValue]],
    doi: str,
    source: MetadataSource,
    confidence: Confidence,
    evidence: str,
) -> None:
    _add(candidates, "doi", doi, source, confidence, evidence)
    _add(
        candidates,
        "url",
        f"https://doi.org/{doi}",
        source,
        Confidence.MEDIUM,
        f"derived from DOI resolved from {evidence}",
    )


def _first_page_text(page: PageLayout | None) -> str:
    if page is None:
        return ""
    return "\n".join(line.text for line in page.lines)


# --- PDF ------------------------------------------------------------------


def _extract_pdf(
    path: Path, candidates: dict[str, list[MetadataValue]], role: CorpusRole
) -> int | None:
    result = read_pdf(path)
    embedded = result.embedded

    title_candidates: list[str] = []

    first_page = result.first_page
    first_page_text = _first_page_text(first_page)
    page_title: str | None = None
    if first_page is not None:
        page_title = extract_first_page_title(first_page)

    # Journal templates routinely stamp the journal name into the embedded
    # "title" field, so an embedded title that reads as a venue masthead while
    # page one yields a materially different article title is discarded.
    embedded_title = embedded.title
    if embedded_title and not is_meaningful_title(embedded_title):
        embedded_title = None
    elif embedded_title and looks_like_venue_masthead(embedded_title, page_title=page_title):
        embedded_title = None

    if embedded_title:
        _add(
            candidates,
            "title",
            embedded_title,
            MetadataSource.PDF_METADATA,
            Confidence.MEDIUM,
            'PDF metadata field "title"',
        )
        title_candidates.append(embedded_title)

    if page_title is not None:
        _add(
            candidates,
            "title",
            page_title,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.MEDIUM,
            "prominent first-page text (font-size based candidate)",
        )
        title_candidates.append(page_title)

    # Authors: embedded field first, then a conservative first-page line.
    if embedded.author and not looks_like_affiliation(embedded.author):
        parsed = parse_author_string(embedded.author)
        if parsed:
            _add(
                candidates,
                "authors",
                parsed,
                MetadataSource.PDF_METADATA,
                Confidence.MEDIUM,
                'PDF metadata field "author"',
            )

    skip_title = page_title or embedded.title
    if first_page is not None:
        page_authors = extract_authors_from_first_page(first_page, skip_title)
        if page_authors:
            _add(
                candidates,
                "authors",
                page_authors,
                MetadataSource.DOCUMENT_TEXT,
                Confidence.MEDIUM,
                "first-page author line",
            )

    # DOI: an explicit identifier on the first page is the strongest local
    # signal, then the PDF metadata fields, then the filename. Trailing pages are
    # intentionally *not* scanned: their DOIs overwhelmingly sit inside the
    # reference list and therefore belong to cited works, not to this document.
    metadata_text = " ".join(
        filter(
            None,
            [embedded.subject, embedded.keywords, embedded.creator, embedded.producer],
        )
    )
    doi = find_doi(first_page_text)
    if doi is not None:
        _add_doi(
            candidates,
            doi,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.HIGH,
            "first-page DOI pattern",
        )
    else:
        doi = find_doi(metadata_text)
        if doi is not None:
            _add_doi(
                candidates,
                doi,
                MetadataSource.PDF_METADATA,
                Confidence.MEDIUM,
                'PDF metadata field "subject"/"keywords"',
            )

    # Year: publication line from page one, then weak file-creation date.
    published_year = extract_year_from_text(first_page_text)
    if published_year is not None:
        _add(
            candidates,
            "year",
            published_year,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.MEDIUM,
            "publication/copyright year on first page",
        )

    creation_year = year_from_pdf_date(embedded.creation_date)
    if creation_year is not None:
        _add(
            candidates,
            "year",
            creation_year,
            MetadataSource.PDF_METADATA,
            Confidence.LOW,
            'PDF metadata "creationDate" (file creation, weak)',
        )

    _extract_arxiv(first_page_text, path.name, candidates)
    _extract_filename(path, candidates)
    _add_role_publication_type(candidates, role)
    return result.page_count


# --- DOCX -----------------------------------------------------------------


def _extract_docx(
    path: Path, candidates: dict[str, list[MetadataValue]], role: CorpusRole
) -> None:
    properties = read_docx_properties(path)
    parsed = parse_docx(path)

    core_title = properties.title
    if core_title and is_meaningful_title(core_title):
        _add(
            candidates,
            "title",
            core_title,
            MetadataSource.DOCX_PROPERTIES,
            Confidence.MEDIUM,
            'DOCX core property "title"',
        )

    styled_title = parsed.title
    if (
        styled_title
        and is_meaningful_title(styled_title)
        and collapse_whitespace(styled_title).casefold()
        != collapse_whitespace(core_title or "").casefold()
    ):
        _add(
            candidates,
            "title",
            styled_title,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.MEDIUM,
            'DOCX Title-styled paragraph',
        )

    if properties.author and not looks_like_affiliation(properties.author):
        parsed_authors = parse_author_string(properties.author)
        if parsed_authors:
            _add(
                candidates,
                "authors",
                parsed_authors,
                MetadataSource.DOCX_PROPERTIES,
                Confidence.MEDIUM,
                'DOCX core property "author"',
            )

    created_year = year_from_datetime(properties.created)
    if created_year is not None:
        _add(
            candidates,
            "year",
            created_year,
            MetadataSource.DOCX_PROPERTIES,
            Confidence.LOW,
            'DOCX core property "created" (file creation, weak)',
        )

    # No DOI is extracted from DOCX text: a draft's bibliography is
    # indistinguishable from its body, and a cited work's DOI must never be
    # adopted as the document's own (see RAG-006 report §20/§21).
    _extract_arxiv(_body_text(parsed), path.name, candidates)
    _extract_filename(path, candidates)
    _add_role_publication_type(candidates, role)


# --- Markdown -------------------------------------------------------------


def _extract_markdown(
    path: Path, candidates: dict[str, list[MetadataValue]], role: CorpusRole
) -> None:
    parsed = parse_markdown(path)

    if parsed.title and is_meaningful_title(parsed.title):
        _add(
            candidates,
            "title",
            parsed.title,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.MEDIUM,
            "top-level Markdown heading",
        )

    body_text = _body_text(parsed)

    # No DOI is extracted from Markdown text for the same reason as DOCX: the
    # document body and its bibliography cannot be reliably told apart.
    published_year = extract_year_from_text(body_text)
    if published_year is not None:
        _add(
            candidates,
            "year",
            published_year,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.LOW,
            "year in document text",
        )

    _extract_arxiv(body_text, path.name, candidates)
    _extract_filename(path, candidates)
    _add_role_publication_type(candidates, role)


# --- Shared helpers -------------------------------------------------------


def _extract_arxiv(
    text: str, filename: str, candidates: dict[str, list[MetadataValue]]
) -> None:
    arxiv = find_arxiv_id(text)
    if arxiv is not None:
        _add(
            candidates,
            "arxiv_id",
            arxiv,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.MEDIUM,
            "arXiv identifier in document text",
        )
        _add(
            candidates,
            "publication_type",
            PublicationType.PREPRINT,
            MetadataSource.DOCUMENT_TEXT,
            Confidence.LOW,
            "arXiv identifier suggests a preprint",
        )


def _extract_filename(
    path: Path, candidates: dict[str, list[MetadataValue]]
) -> None:
    hints = parse_filename_metadata(path.name)

    if hints.arxiv_id is not None:
        _add(
            candidates,
            "arxiv_id",
            hints.arxiv_id,
            MetadataSource.FILENAME,
            Confidence.LOW,
            "arXiv identifier in filename",
        )

    for year in hints.years:
        _add(
            candidates,
            "year",
            year,
            MetadataSource.FILENAME,
            Confidence.LOW,
            "year-like token in filename",
        )

    token = hints.author_token
    if token:
        _add(
            candidates,
            "authors",
            [Author(full_name=token)],
            MetadataSource.FILENAME,
            Confidence.LOW,
            "author-like token in filename",
        )


def _add_role_publication_type(
    candidates: dict[str, list[MetadataValue]], role: CorpusRole
) -> None:
    """Add the folder-context manuscript classification for the author's work."""
    if role is CorpusRole.MY_WORK:
        _add(
            candidates,
            "publication_type",
            PublicationType.AUTHOR_MANUSCRIPT,
            MetadataSource.FOLDER,
            Confidence.LOW,
            "author's own work under data/my_work",
        )


def should_enrich(metadata: AcademicMetadata) -> bool:
    """Return whether ``metadata`` may be sent to an external service."""
    return is_enrichable(
        metadata.corpus_role, has_doi=metadata.doi is not None
    )
