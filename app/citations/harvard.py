"""Harvard in-text citations and reference list (RAG-012 §9-§12).

Everything here is built from metadata that is actually present on the source block. The
rules are deliberately literal:

* ``(Smith, 2024)`` for one author, ``(Smith and Jones, 2024)`` for two,
  ``(Smith et al., 2024)`` for three or more;
* a missing year becomes ``n.d.`` and a missing author falls back to the title, with the
  fallback recorded per citation rather than invented;
* citations for one sentence are ordered alphabetically by author, then by year, so the
  same claim always renders the same way;
* one reference entry per **work**, not per chunk: identity is the DOI first, then the
  normalised title, then the document. Eight chunks of one paper produce one entry.

The author list available to this stage is ``SourceBlock.first_author``, which the
metadata layer fills with the first author's full name. When that string itself carries
several names (``"A. Smith and B. Jones"``, ``"Doe, J. and Smith, J."``), they are
parsed as separate authors; when it carries ``et al.``, the citation keeps ``et al.``.
Nothing is guessed beyond that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.citations.models import CitationMapEntry, ReferenceEntry
from app.generation.models import SourceBlock
from app.metadata.doi import normalize_doi

#: Name particles that belong to the surname.
_PARTICLES = frozenset(
    {"van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "du", "la", "le", "bin", "ibn", "st", "mac", "mc"}
)
_AUTHOR_SEPARATORS = re.compile(r"\s*(?:;|\band\b|&)\s*", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
#: ``Smith et al.`` keeps its ``et al.``: the writer already said there are more authors.
_ET_AL_RE = re.compile(r"\s*\bet\s*al\.?\s*$", re.IGNORECASE)

#: Fallback labels, each recorded in ``fallbacks`` so a reader can see what was missing.
YEAR_FALLBACK = "n.d."
TITLE_FALLBACK = "Untitled work"
MISSING_AUTHOR = "missing_author"
MISSING_YEAR = "missing_year"
MISSING_TITLE = "missing_title"
MISSING_DOI = "missing_doi"
DENORMALIZED_DOI = "doi_not_normalized"


@dataclass(frozen=True)
class AuthorName:
    """One parsed author."""

    raw: str
    surname: str
    initials: str = ""
    et_al: bool = False

    def reference_form(self) -> str:
        """``Doe, J.`` — the form used in a reference list."""
        if self.initials:
            return f"{self.surname}, {self.initials}"
        return self.surname


def _initials(given: str) -> str:
    """``"Jane A."`` to ``"J.A."``; existing initials are preserved."""
    letters: list[str] = []
    for token in given.replace(".", " ").split():
        lowered = token.casefold()
        if lowered in _PARTICLES:
            continue
        if token[:1].isalpha():
            letters.append(f"{token[0].upper()}.")
    return "".join(letters)


def _from_name(name: str) -> AuthorName:
    """Parse a ``Given Surname`` string (particles kept with the surname)."""
    cleaned = " ".join(name.split()).strip(" ,;.")
    if not cleaned:
        return AuthorName(raw=name, surname="")
    tokens = cleaned.split()
    if len(tokens) == 1:
        return AuthorName(raw=name, surname=tokens[0], initials="")
    particle_index = next(
        (index for index, token in enumerate(tokens[:-1]) if token.casefold() in _PARTICLES),
        None,
    )
    if particle_index is not None:
        return AuthorName(
            raw=name,
            surname=" ".join(tokens[particle_index:]),
            initials=_initials(" ".join(tokens[:particle_index])),
        )
    return AuthorName(
        raw=name, surname=tokens[-1], initials=_initials(" ".join(tokens[:-1]))
    )


def _parse_part(part: str) -> list[AuthorName]:
    """Parse one author segment, which may be inverted or a list of full names."""
    segments = [segment.strip() for segment in part.split(",") if segment.strip()]
    if len(segments) > 1:
        # "Doe, J." / "Doe, Jane A." -> one inverted name.
        if len(segments[0].split()) == 1:
            return [
                AuthorName(
                    raw=part,
                    surname=segments[0],
                    initials=_initials(" ".join(segments[1:])),
                )
            ]
        # "Jane Doe, John Smith" -> two full names.
        if all(len(segment.split()) >= 2 for segment in segments):
            return [_from_name(segment) for segment in segments]
    return [_from_name(part)]


def _usable_author(author: AuthorName) -> bool:
    """Whether one parsed author can be used in a citation.

    The corpus metadata is extracted from PDFs, so an "author" is sometimes a watermark
    (``ssrn``), a stray single letter (from ``Japheth Kodua Wiredu a``), or a fragment with
    digits. Such a value is rejected and the documented title fallback is used instead with
    the fallback recorded: a citation must never invent an author, and ``(ssrn, n.d.)`` is
    not one.

    The metadata layer's own ``is_plausible_author_name`` is deliberately not reused here:
    it validates raw extracted strings, and would reject a valid reconstructed form such as
    ``A. Lee``.
    """
    surname = author.surname.strip()
    if len(surname) < 2:
        return False
    if not any(character.isalpha() for character in surname):
        return False
    if any(character.isdigit() for character in surname):
        return False
    # A lone lower-case token is a watermark, a journal stub or a file name.
    if len(surname.split()) == 1 and surname.islower():
        return False
    return True


def split_author_string(raw: str | None) -> tuple[AuthorName, ...]:
    """Parse the author string of a block into usable authors, or an empty tuple."""
    text = " ".join((raw or "").split()).strip()
    if not text:
        return ()
    et_al = bool(_ET_AL_RE.search(text))
    base = _ET_AL_RE.sub("", text).strip(" ,;")
    authors: list[AuthorName] = []
    for part in _AUTHOR_SEPARATORS.split(base):
        if part.strip():
            authors.extend(_parse_part(part.strip()))
    authors = [author for author in authors if author.surname]
    authors = [author for author in authors if _usable_author(author)]
    if not authors:
        return ()
    if et_al:
        first = authors[0]
        authors = [AuthorName(raw=first.raw, surname=first.surname, initials=first.initials, et_al=True)]
    return tuple(authors)


def short_title(title: str | None, *, words: int = 5) -> str:
    """The first ``words`` words of a title, for the missing-author fallback."""
    tokens = (title or "").split()
    if not tokens:
        return TITLE_FALLBACK
    return " ".join(tokens[:words]).rstrip(".,;:")


def author_label(block: SourceBlock) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """The alphabetical label for in-text citations, plus the fallbacks recorded."""
    raw = (block.first_author or "").strip()
    authors = split_author_string(block.first_author)
    if not authors:
        reason = (
            f"author metadata unusable ({raw!r})" if raw else "no author metadata"
        )
        if not (block.title or "").strip():
            return (
                TITLE_FALLBACK,
                (MISSING_AUTHOR, MISSING_TITLE),
                (reason, "no title metadata"),
            )
        return (
            f'"{short_title(block.title)}"',
            (MISSING_AUTHOR,),
            (f"{reason}: the short title is used instead",),
        )
    if len(authors) == 1:
        if authors[0].et_al:
            return f"{authors[0].surname} et al.", (), ()
        return authors[0].surname, (), ()
    if len(authors) == 2:
        return f"{authors[0].surname} and {authors[1].surname}", (), ()
    return f"{authors[0].surname} et al.", (), ()


def authors_display(block: SourceBlock) -> tuple[str, tuple[str, ...]]:
    """The author list as it appears in a reference entry, plus any warnings."""
    authors = split_author_string(block.first_author)
    if not authors:
        return "", ("no author metadata",)
    if len(authors) == 1:
        if authors[0].et_al:
            return f"{authors[0].surname} et al.", ()
        return authors[0].reference_form(), ()
    if len(authors) == 2:
        return f"{authors[0].reference_form()} and {authors[1].reference_form()}", ()
    head = ", ".join(author.reference_form() for author in authors[:-1])
    return f"{head} and {authors[-1].reference_form()}", ()


def year_label(block: SourceBlock) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """``"2024"`` or the ``n.d.`` fallback, with the fallback recorded."""
    if block.year:
        return str(block.year), (), ()
    return YEAR_FALLBACK, (MISSING_YEAR,), ("no year metadata",)


def normalized_source_doi(block: SourceBlock) -> tuple[str | None, tuple[str, ...]]:
    """The normalized DOI, or ``None`` when absent or unusable."""
    raw = (block.doi or "").strip()
    if not raw:
        return None, ()
    normalized = normalize_doi(raw)
    if normalized is None:
        return None, (f"doi not usable: {raw}",)
    if normalized != raw:
        return normalized, ()
    return normalized, ()


def source_metadata_warnings(block: SourceBlock) -> tuple[str, ...]:
    """The deterministic metadata checks of §3, as human-readable notes."""
    notes: list[str] = []
    if not (block.first_author or "").strip():
        notes.append("no author metadata")
    if not block.year:
        notes.append("no year metadata")
    if not (block.title or "").strip():
        notes.append("no title metadata")
    raw_doi = (block.doi or "").strip()
    if not raw_doi:
        notes.append("no doi")
    elif normalize_doi(raw_doi) is None:
        notes.append(f"doi not usable: {raw_doi}")
    if block.pages:
        notes.append(f"page provenance: {block.pages}")
    return tuple(notes)


def block_in_text(block: SourceBlock) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    """The in-text citation for one block: ``(Doe, 2024)``.

    Returns ``(in_text, author_label, year_label, fallbacks, warnings)``.
    """
    label, author_fallbacks, author_warnings = author_label(block)
    year, year_fallbacks, year_warnings = year_label(block)
    fallbacks = tuple(author_fallbacks) + tuple(year_fallbacks)
    warnings = tuple(author_warnings) + tuple(year_warnings)
    return f"({label}, {year})", label, year, fallbacks, warnings


def reference_identity(block: SourceBlock) -> tuple[str, str, str]:
    """``(key, identity_kind, identity_value)`` — DOI, then title, then work identity."""
    doi, _warnings = normalized_source_doi(block)
    if doi:
        return f"doi:{doi.casefold()}", "doi", doi
    title = (block.title or "").strip()
    if title:
        return f"title:{_NON_ALNUM.sub('', title.casefold())}", "title", title
    if block.document_id:
        return f"doc:{block.document_id}", "document", block.document_id
    if block.chunk_id:
        return f"chunk:{block.chunk_id}", "chunk", block.chunk_id
    return f"source:{block.source_id}", "source", block.source_id


def format_pages(pages: str | None) -> str:
    """``"4-6"`` to ``"pp. 4-6"``, ``"4"`` to ``"p. 4"``."""
    if not pages:
        return ""
    text = pages.strip()
    if not text:
        return ""
    if "-" in text or "\u2013" in text:
        return f"pp. {text}"
    return f"p. {text}"


def format_reference(entry: ReferenceEntry) -> str:
    """Render one Harvard reference line from the metadata the entry actually has."""
    parts: list[str] = []
    if entry.authors_display:
        parts.append(f"{entry.authors_display} ({entry.year_label})")
        if entry.title:
            parts.append(f"'{entry.title}'")
    elif entry.title:
        # No author: Harvard puts the title in the author position (§11). Double quotes
        # keep an apostrophe inside the title unambiguous.
        parts.append(f'"{entry.title}" ({entry.year_label})')
    else:
        parts.append(f"({entry.year_label})")
    tail = ", ".join(item for item in (entry.venue,) if item)
    pages = format_pages(entry.pages)
    if pages:
        tail = f"{tail}, {pages}" if tail else pages
    line = " ".join(parts)
    if tail:
        line = f"{line}, {tail}"
    if entry.doi:
        line = f"{line}. doi:{entry.doi}"
    elif entry.url:
        line = f"{line}. Available at: {entry.url}"
    else:
        line = f"{line}."
    if entry.metadata_warning:
        line = f"{line} [metadata_warning: {', '.join(entry.metadata_warning)}]"
    return line


def build_reference(block: SourceBlock) -> ReferenceEntry:
    """Build the reference entry for one block (not yet deduplicated)."""
    key, identity_kind, identity_value = reference_identity(block)
    label, author_fallbacks, author_warnings = author_label(block)
    display, display_warnings = authors_display(block)
    year, year_fallbacks, year_warnings = year_label(block)
    doi, doi_warnings = normalized_source_doi(block)
    warnings = list(author_warnings) + list(display_warnings) + list(year_warnings) + list(doi_warnings)
    metadata_warning: list[str] = []
    if MISSING_AUTHOR in author_fallbacks:
        metadata_warning.append(MISSING_AUTHOR)
    if MISSING_YEAR in year_fallbacks:
        metadata_warning.append(MISSING_YEAR)
    if not (block.title or "").strip():
        metadata_warning.append(MISSING_TITLE)
    if not doi:
        # No usable DOI: the entry is matched by title or work identity instead (§11).
        metadata_warning.append(MISSING_DOI)
    entry = ReferenceEntry(
        key=key,
        identity_kind=identity_kind,
        identity_value=identity_value,
        author_label=label,
        authors_display=display,
        year=block.year,
        year_label=year,
        title=(block.title or "").strip() or None,
        venue=(block.venue or "").strip() or None,
        pages=(block.pages or "").strip() or None,
        doi=doi,
        url=(block.external_source_url or "").strip() or None,
        source_ids=(block.source_id,),
        chunk_ids=(block.chunk_id,) if block.chunk_id else (),
        document_ids=(block.document_id,) if block.document_id else (),
        metadata_warning=tuple(metadata_warning),
    )
    in_text = f"({label}, {year})"
    return entry.model_copy(
        update={
            "formatted": format_reference(entry),
            "in_text": in_text,
            "metadata_warning": tuple(metadata_warning),
        }
    )


def _merge_reference(entry: ReferenceEntry, block: SourceBlock) -> ReferenceEntry:
    """Fold a second block of the same work into one entry (§12)."""
    doi, _doi_warnings = normalized_source_doi(block)
    merged = entry.model_copy(
        update={
            "source_ids": entry.source_ids + (block.source_id,),
            "chunk_ids": entry.chunk_ids + ((block.chunk_id,) if block.chunk_id else ()),
            "document_ids": entry.document_ids
            + ((block.document_id,) if block.document_id else ()),
            "pages": entry.pages or ((block.pages or "").strip() or None),
            "venue": entry.venue or ((block.venue or "").strip() or None),
            "title": entry.title or ((block.title or "").strip() or None),
            "year": entry.year or block.year,
            "doi": entry.doi or doi,
            "url": entry.url or ((block.external_source_url or "").strip() or None),
            "authors_display": entry.authors_display or authors_display(block)[0],
        }
    )
    return merged.model_copy(update={"formatted": format_reference(merged)})


def build_reference_list(blocks: list[SourceBlock] | tuple[SourceBlock, ...]) -> tuple[ReferenceEntry, ...]:
    """One deduplicated, deterministically ordered entry per cited work.

    Identity is the DOI first, then the normalised title, then the document: eight chunks
    of one paper collapse into one entry. Ordering is alphabetical by author label, then
    year, then key, so the list is stable for a given set of sources.
    """
    entries: dict[str, ReferenceEntry] = {}
    for block in blocks:
        key, _kind, _value = reference_identity(block)
        if key in entries:
            entries[key] = _merge_reference(entries[key], block)
        else:
            entries[key] = build_reference(block)
    return tuple(sorted(entries.values(), key=lambda entry: entry.sort_key))


def citation_for_ids(
    source_ids: list[str] | tuple[str, ...],
    reference_by_source: dict[str, ReferenceEntry],
) -> str:
    """``"(Brown, 2022; Chan, 2024)"`` for a marker group, in alphabetical order.

    Several markers of one group may point at chunks of the *same* work; the work is cited
    once, because the group yields one reference entry per work either way.
    """
    unique: dict[str, ReferenceEntry] = {}
    for source_id in source_ids:
        entry = reference_by_source.get(source_id)
        if entry is not None:
            unique.setdefault(entry.key, entry)
    ordered = sorted(unique.values(), key=lambda entry: entry.sort_key)
    if not ordered:
        return ""
    return "(" + "; ".join(entry.in_text.strip("()") for entry in ordered) + ")"


def citation_map(
    source_map: dict[str, dict[str, object]],
    reference_by_source: dict[str, ReferenceEntry],
) -> tuple[CitationMapEntry, ...]:
    """``S<n>`` to in-text citation, reference key and chunk provenance, in ``S`` order."""
    entries: list[CitationMapEntry] = []

    def source_number(source_id: str) -> int:
        return int(source_id[1:]) if source_id[1:].isdigit() else 0

    for source_id in sorted(reference_by_source, key=source_number):
        reference = reference_by_source[source_id]
        block = source_map.get(source_id, {})
        fallbacks = tuple(
            warning
            for warning in reference.metadata_warning
            if warning in (MISSING_AUTHOR, MISSING_YEAR)
        )
        entries.append(
            CitationMapEntry(
                source_id=source_id,
                in_text=reference.in_text,
                reference_key=reference.key,
                author_label=reference.author_label,
                year_label=reference.year_label,
                fallbacks=fallbacks,
                metadata_warning=reference.metadata_warning,
                title=reference.title,
                chunk_id=_as_text(block.get("chunk_id")),
                document_id=_as_text(block.get("document_id")),
                doi=reference.doi,
                pages=reference.pages,
            )
        )
    return tuple(entries)


def _as_text(value: object) -> str | None:
    """``str`` when a source-map value is present, else ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
