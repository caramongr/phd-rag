"""High-confidence bibliography-like detection at retrieval time (RAG-009).

RAG-009 found 142 indexed chunks whose upstream ``content_type`` is *not*
``references`` but whose text is a reference list, because RAG-007's
reference-heading detection does not fire for every document. Rebuilding the index
is out of scope here, so the guard is applied where the chunk is *read*: nothing in
Qdrant, the chunks, the embeddings or the sources is touched.

The detector is deliberately narrow. It is not a citation detector: an academic
paragraph that cites three sources is normal prose and must never be flagged. A
chunk is flagged only when several independent bibliographic signals agree that the
chunk *is* a list of reference entries:

* repeat ``Available at:`` with matching URLs (the Harvard/exported style);
* a labelled reference section (``References``/``Bibliography`` heading);
* four or more DOIs in one chunk;
* a long run of parenthetical publication years together with DOI/URL markers;
* five or more lines that start like a reference entry (``Surname, A.``);
* six or more numbered entries mixed with citation markers;
* very high marker density **plus** reference furniture (a number immediately
  followed by a year, as in ``Nat. Commun. 5, 4006 (2014)``) and no long prose line.

Every rule is a deterministic threshold on the chunk's own text; no model, no
learned weight, no corpus-wide statistics. The verdict and the rule that fired are
carried on the candidate so a result can always explain why it was withheld.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- thresholds ---------------------------------------------------------------
#: Rules are ordered; the first one that fires provides the reason.
MIN_WORDS_FOR_ENTRY_RULES = 30
MIN_WORDS_FOR_DOI_RULE = 60
AVAILABLE_AT_ENTRIES = 2
AVAILABLE_AT_DENSITY = 4.0
DOI_ENTRIES = 4
YEAR_ENTRIES = 8
YEAR_ENTRIES_WITH_LINKS = 2
ENTRY_LINES = 5
ENTRY_LINE_DENSITY = 4.0
NUMBERED_ENTRIES = 6
NUMBERED_ENTRY_MARKERS = 6
#: Structural list evidence that lets the density rule skip the prose-line guard.
DENSE_BYPASS_LIST_LINES = 3
DENSE_MARKERS = 6.0
DENSE_MAX_LONG_LINE_RATIO = 0.35
#: Furniture that reference entries carry and in-text citations do not: a number
#: followed by a year (``Nat. Commun. 5, 4006 (2014)``), a DOI, a URL or journal
#: volume/page markers. Prevents citation-heavy prose from being mistaken for a list.
DENSE_MIN_FURNITURE = 3
#: And at least this many signals that are *not* a bare link, so an ORCID/URL list is
#: never mistaken for a bibliography.
DENSE_MIN_BIBLIOGRAPHIC = 3
DENSE_MAX_LINES_FOR_SHORT_RULE = 3
#: A heading the *document itself* uses to label the section, plus enough list content
#: to be sure the chunk is the section and not a sentence that starts with the word.
HEADING_MIN_MARKERS = 3
HEADING_MIN_ENTRY_YEARS = 2
HEADING_MIN_DENSITY = 8.0
HEADING_MIN_ENTRY_LINES = 3

REASON_AVAILABLE_AT = "repeated_available_at_entries"
REASON_DOI = "doi_dense_entry_list"
REASON_YEARS = "citation_year_dense_list"
REASON_ENTRY_LINES = "author_initials_entry_lines"
REASON_NUMBERED = "numbered_entry_list"
REASON_DENSITY = "dense_reference_markers"
REASON_HEADING = "labelled_reference_section"

# --- patterns -----------------------------------------------------------------
#: A DOI, with or without a resolver URL, e.g. ``10.1016/j.compedu.2019.103778``.
_DOI = re.compile(r"(?:https?://doi\.org/|doi:\s*|10\.\d{4,9}/)\S+", re.IGNORECASE)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
#: ``Available at: https://…`` - the colon matters, because an ordinary sentence can say
#: "available at the point of use".
_AVAILABLE_AT = re.compile(r"available\s+(?:at|from)\s*:", re.IGNORECASE)
_ACCESSED = re.compile(r"\b(?:accessed|retrieved)\b", re.IGNORECASE)
#: A parenthetical publication year, the backbone of an APA/Harvard entry.
_YEAR = re.compile(r"\((?:19|20)\d{2}[a-z]?(?:[,;][^()]{0,12})?\)")
_ET_AL = re.compile(r"\bet\s+al\.?", re.IGNORECASE)
#: Journal/volume furniture that prose rarely carries: ``vol. 45``, ``12(3)``, ``pp. 45``.
#: Deliberately narrow: a bare ``1(2), 3`` sequence also occurs in mathematical prose, so
#: the volume(issue) form must follow a word.
_JOURNAL = re.compile(
    r"(?:\b(?:vol|no|pp?|issn|isbn)\.?\s*\d"
    r"|\b[A-Za-z][A-Za-z.\-]{1,24}\s+\d{1,4}\s*\(\d{1,3}\)\s*[,:]\s*\d"
    r"|\bdoi:\s*)",
    re.IGNORECASE,
)
#: A line that starts like a reference entry: ``Surname, A.`` or ``Surname, A.B.``
_ENTRY_LINE = re.compile(
    r"^\s*(?:[-•*]\s*)?[A-Z][A-Za-z'’\-]{1,24},\s*(?:[A-Z]\.|[A-Z][a-z]+|\d)",
    re.MULTILINE,
)
#: A numbered reference entry: ``[12] Author``, ``12. Author``.
_NUMBERED_LINE = re.compile(
    r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[.)])\s*(?=[A-Z]|\d)", re.MULTILINE
)
#: A heading that labels the section as a bibliography. Matched as a whole short line,
#: or as the first words of the first line (PDF chunks often run the heading into the
#: first entry).
_REFERENCE_HEADING_LINE = re.compile(
    r"^\s*(?:\d{1,2}[.)]?\s*)?"
    r"(?:references|reference list|references cited|bibliography|works cited|literature cited)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)
_REFERENCE_HEADING_PREFIX = re.compile(
    r"^\s*(?:\d{1,2}[.)]?\s*)?"
    r"(?:references|reference list|references cited|bibliography|works cited|literature cited)"
    r"\b[\s:.\-]*\S",
    re.IGNORECASE,
)
#: ``5, 4006 (2014)`` / ``58, 101862 (2020)`` - a journal number and year, which prose
#: citing a source in running text does not produce.
_ENTRY_YEAR = re.compile(r"\b\d{1,6}\s*\((?:19|20)\d{2}[a-z]?\)")
_WORDS = re.compile(r"\S+")


def _word_count(text: str) -> int:
    return len(_WORDS.findall(text))


@dataclass(frozen=True)
class BibliographyFeatures:
    """The measurements a verdict is based on (kept for reports and tests)."""

    words: int
    lines: int
    doi: int
    url: int
    available_at: int
    accessed: int
    parenthetical_years: int
    et_al: int
    journal_markers: int
    entry_lines: int
    numbered_lines: int
    entry_year_markers: int
    long_line_ratio: float
    reference_heading: bool = False

    @property
    def markers(self) -> int:
        """Weighted bibliographic marker count (links weigh more than words)."""
        return (
            3 * self.doi
            + 2 * self.url
            + 3 * self.available_at
            + self.accessed
            + self.parenthetical_years
            + self.et_al
            + self.journal_markers
        )

    @property
    def density(self) -> float:
        """Markers per 100 words - the size-independent signal."""
        return self.markers / max(self.words / 100.0, 1.0)

    @property
    def bibliographic_markers(self) -> int:
        """Markers that only reference furniture produces (no bare URLs)."""
        return self.doi + self.url + self.available_at + self.parenthetical_years

    @property
    def non_url_markers(self) -> int:
        """Bibliographic signals other than a bare link (guards link-only lists)."""
        return (
            self.doi
            + self.available_at
            + self.parenthetical_years
            + self.journal_markers
            + self.entry_year_markers
        )

    @property
    def furniture(self) -> int:
        """Reference-entry furniture: links, DOIs, journal markers, ``no., year``."""
        return self.doi + self.url + self.journal_markers + self.entry_year_markers


    def describe(self) -> str:
        """Compact single-line summary for an audit record."""
        return (
            f"words={self.words} lines={self.lines} doi={self.doi} url={self.url} "
            f"available_at={self.available_at} years={self.parenthetical_years} "
            f"entry_lines={self.entry_lines} numbered={self.numbered_lines} "
            f"entry_years={self.entry_year_markers} heading={self.reference_heading} "
            f"markers={self.markers} density={self.density:.2f}"
        )


@dataclass(frozen=True)
class BibliographyVerdict:
    """Whether a chunk is a reference list, and why."""

    is_bibliography_like: bool
    reason: str | None
    features: BibliographyFeatures

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.is_bibliography_like


def measure(text: str) -> BibliographyFeatures:
    """Compute the bibliographic features of ``text``."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_count = max(len(lines), 1)
    long_lines = sum(1 for line in lines if _word_count(line) >= 25)
    return BibliographyFeatures(
        words=_word_count(text),
        lines=line_count,
        doi=len(_DOI.findall(text)),
        url=len(_URL.findall(text)),
        available_at=len(_AVAILABLE_AT.findall(text)),
        accessed=len(_ACCESSED.findall(text)),
        parenthetical_years=len(_YEAR.findall(text)),
        et_al=len(_ET_AL.findall(text)),
        journal_markers=len(_JOURNAL.findall(text)),
        entry_lines=len(_ENTRY_LINE.findall(text)),
        numbered_lines=len(_NUMBERED_LINE.findall(text)),
        entry_year_markers=len(_ENTRY_YEAR.findall(text)),
        long_line_ratio=long_lines / line_count,
        reference_heading=(
            any(_REFERENCE_HEADING_LINE.match(line) for line in lines[:4])
            or bool(lines and _REFERENCE_HEADING_PREFIX.match(lines[0]))
        ),
    )


def classify(text: str) -> BibliographyVerdict:
    """Decide whether ``text`` is a high-confidence reference list.

    The rules require *agreement* between quantitative density and structural
    markers, so ordinary academic prose that cites several sources - even with DOIs
    and URLs - stays retrievable.
    """
    features = measure(text or "")
    words = features.words
    links = features.doi + features.url

    if words >= MIN_WORDS_FOR_ENTRY_RULES:
        if (
            features.available_at >= AVAILABLE_AT_ENTRIES
            and features.url >= AVAILABLE_AT_ENTRIES
            and features.density >= AVAILABLE_AT_DENSITY
        ):
            return BibliographyVerdict(True, REASON_AVAILABLE_AT, features)
        if (
            features.parenthetical_years >= YEAR_ENTRIES
            and links >= YEAR_ENTRIES_WITH_LINKS
        ):
            return BibliographyVerdict(True, REASON_YEARS, features)
        if features.entry_lines >= ENTRY_LINES and features.density >= ENTRY_LINE_DENSITY:
            return BibliographyVerdict(True, REASON_ENTRY_LINES, features)
        if (
            features.numbered_lines >= NUMBERED_ENTRIES
            and (
                features.available_at
                + features.doi
                + features.url
                + features.parenthetical_years
                + features.journal_markers
                + features.entry_year_markers
            )
            >= NUMBERED_ENTRY_MARKERS
        ):
            return BibliographyVerdict(True, REASON_NUMBERED, features)
        if (
            features.reference_heading
            and features.bibliographic_markers >= HEADING_MIN_MARKERS
            and (
                features.entry_lines >= HEADING_MIN_ENTRY_LINES
                or features.entry_year_markers >= HEADING_MIN_ENTRY_YEARS
                or features.density >= HEADING_MIN_DENSITY
            )
        ):
            return BibliographyVerdict(True, REASON_HEADING, features)
    if words >= MIN_WORDS_FOR_DOI_RULE and features.doi >= DOI_ENTRIES:
        return BibliographyVerdict(True, REASON_DOI, features)
    if (
        words >= MIN_WORDS_FOR_ENTRY_RULES
        and features.density >= DENSE_MARKERS
        and features.non_url_markers >= DENSE_MIN_BIBLIOGRAPHIC
        and features.furniture >= DENSE_MIN_FURNITURE
        and (
            features.long_line_ratio <= DENSE_MAX_LONG_LINE_RATIO
            or features.lines <= DENSE_MAX_LINES_FOR_SHORT_RULE
            or features.numbered_lines >= DENSE_BYPASS_LIST_LINES
            or features.entry_lines >= DENSE_BYPASS_LIST_LINES
        )
    ):
        return BibliographyVerdict(True, REASON_DENSITY, features)
    return BibliographyVerdict(False, None, features)


def is_bibliography_like(text: str) -> bool:
    """Convenience wrapper around :func:`classify`."""
    return classify(text).is_bibliography_like
