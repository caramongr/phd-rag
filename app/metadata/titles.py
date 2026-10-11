"""Conservative title validity checks and first-page title candidate extraction.

Two responsibilities live here:

* :func:`is_meaningful_title` decides whether a *given* string is usable as an
  academic title. It rejects obviously useless embedded-metadata values such as
  ``untitled`` or ``Microsoft Word - document1.docx`` — and, deterministically,
  journal/venue mastheads, proceedings template labels, production identifiers,
  proof markers, reprint notices and unmappable glyph runs — without over-cleaning
  legitimate titles.
* :func:`extract_first_page_title` proposes a title candidate from the typography
  of a first page. It is a candidate only, never authoritative, and returns
  ``None`` when it is uncertain. When the most prominent string is rejected it
  continues to the next conservative candidate.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass

from app.metadata.authors import (
    looks_like_affiliation,
    looks_like_author_line,
    split_author_list,
)
from app.metadata.doi import find_doi
from app.metadata.layout import PageLayout
from app.metadata.text import (
    SECTION_HEADINGS,
    collapse_whitespace,
    is_email,
    is_url,
    normalize_for_comparison,
)

#: Values that are never meaningful academic titles (matched case-insensitively
#: after whitespace collapse). ``document``/``untitled`` etc. are authorship or
#: scanner artefacts, not titles.
_USELESS_EXACT: frozenset[str] = frozenset(
    {
        "untitled",
        "untitled document",
        "no title",
        "document",
        "document1",
        "empty",
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
        "title",
        "pdf",
        "microsoft word",
        "microsoft powerpoint",
        "powerpoint presentation",
        "presentation",
        "slide",
        "slides",
        "scan",
        "scanned document",
        "image",
        "new document",
    }
)

#: Regexes for scanner/office artefacts that are not titles.
_USELESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^microsoft\s+(word|powerpoint|office)\b", re.IGNORECASE),
    re.compile(r"\badobe\s+acrobat\b", re.IGNORECASE),
    re.compile(r"\bacrobat\s+distiller\b", re.IGNORECASE),
    re.compile(r"^doc(ument)?\s*\d*$", re.IGNORECASE),
    re.compile(r"^image\s*\d*$", re.IGNORECASE),
    re.compile(r"^scan(ner)?\s*\d*$", re.IGNORECASE),
    re.compile(r"^\d+$"),
    re.compile(r"^[a-z0-9_-]+\.(pdf|docx?|pptx?|rtf|txt|md)$", re.IGNORECASE),
)

#: Named section headings that commonly appear in a large font on page one but
#: are not the document title.
_SECTION_HEADINGS: frozenset[str] = SECTION_HEADINGS

#: Words/phrases that signal a running journal/conference header rather than a
#: title. Matched with word boundaries -- a title containing "issues" must not be
#: rejected because "issue" happens to be a substring.
_VENUE_HEADER_RE = re.compile(
    r"\bproceedings\s+of\b"
    r"|\btransactions\s+on\b"
    r"|\bjournal\s+of\b"
    r"|\binternational\s+conference\s+on\b"
    r"|\bannual\s+meeting\b"
    r"|\bissn\b"
    r"|\bvolume\b"
    r"|\bvol\.\s*\d"
    r"|\bno\.\s*\d"
    r"|\bissue\s*\d"
    r"|\bpp\.\s*\d"
    r"|\bieee\b"
    r"|\bacm\b"
    r"|\bspringer\b"
    r"|\belsevier\b"
    r"|\bwiley\b"
    r"|\bcopyright\b"
    r"|\ball\s+rights\s+reserved\b"
    r"|\bpermission\b"
    r"|\bpublished\s+by\b"
    r"|\breceived\s*:"
    r"|\baccepted\s*:"
    r"|\bdoi\s*:",
    re.IGNORECASE,
)

_EMAIL_OR_URL_TOKENS = ("http", "www.", "@")

_ISSUE_NUMBER_RE = re.compile(r"\b(vol|volume|no|issue|pp|pages?)\.?\s*\d", re.IGNORECASE)
_YEAR_ONLY_RE = re.compile(r"^(19|20)\d{2}$")
_PAGE_NUMBER_RE = re.compile(r"^page\s*\d+(\s*of\s*\d+)?$", re.IGNORECASE)
_ISSN_RE = re.compile(r"\bissn\b", re.IGNORECASE)
#: arXiv margin stamps such as ``arXiv:1910.10045v2 [cs.AI] 26 Dec 2019``.
_ARXIV_STAMP_RE = re.compile(r"arxiv\s*:\s*\d", re.IGNORECASE)

# --- Non-title artefact classes (RAG-006 title-extraction correction) --------
#
#: Publisher reprint/preprint notices, e.g. "This is an electronic reprint of
#: the original article". Never a title.
_REPRINT_NOTICE_RE = re.compile(
    r"electronic\s+reprint"
    r"|\baccepted\s+manuscript\b"
    r"|\barticle\s+in\s+press\b"
    r"|\breprint\s+may\s+differ\b"
    r"|\bdownloaded\s+from\b"
    r"|\bthis\s+article\s+is\s+protected\s+by\s+copyright\b"
    r"|\bsee\s+discussions?,?\s+stats"
    r"|\bprovided\s+by\s+the\s+author\b"
    r"|\bpublished\s+in\s+final\s+form\b"
    r"|\bauthor'?s?\s+(?:accepted|personal)\s+(?:manuscript|copy)\b",
    re.IGNORECASE,
)

#: Publisher/format templates, e.g. "Proceedings Template - WORD",
#: "SIGCHI Conference Proceedings Format". Never a title.
_TEMPLATE_LABEL_RE = re.compile(
    r"proceedings\s+template"
    r"|conference\s+proceedings\s+format"
    r"|proceedings\s+format"
    r"|\btemplate\s*[-\u2013]\s*word\b"
    r"|\b(?:word|latex|manuscript|paper|article|ieee|acm|sigchi|chi)\s+template\b"
    r"|\bthis\s+template\b"
    r"|\bformatting\s+instructions\b"
    r"|\bsubmission\s+template\b",
    re.IGNORECASE,
)

#: A page-range like ``1..24`` / ``11..11`` only ever appears in production codes.
_PAGE_RANGE_RE = re.compile(r"\b\d{1,4}\.\.\d{1,4}\b")
#: Typesetting/production markers.
_PRODUCTION_MARKER_RE = re.compile(
    r"\bpages?\s*[-\u2013\u2014]{1,2}\s*final\b|\bgalley\b|\btypeset(?:ting)?\b"
    r"|\bin\s+production\b",
    re.IGNORECASE,
)
#: Bare production codes, e.g. ``DSJ_2300028``, ``APC873088``, ``feduc-2021-645758``,
#: ``JMI-18219SSR``, ``7543-Article``.
_PRODUCTION_CODE_RE = re.compile(
    r"\b[A-Z]{2,}[-_]?\d{4,}[A-Za-z]*\b"
    r"|\b\d{3,}[-_][A-Za-z]{2,}\b"
    r"|\b[a-z]{2,}[-_]\d{4}[-_]\d{3,}\b"
)
#: ``Article Text`` only indicates a production identifier when digits are nearby.
_ARTICLE_TEXT_RE = re.compile(r"\barticle\s+text\b(?=[^\n]*\d)", re.IGNORECASE)
#: Uppercase ``PROOF`` (``PROOF II``) is a production marker; ``proof`` in prose is not.
_PROOF_MARKER_RE = re.compile(r"\bPROOF\b")
#: Venue labels that mark a masthead even when the line is long.
_VENUE_LABEL_RE = re.compile(
    r"\b(journal|journals|proceedings|transactions|dergisi|revista|zeitschrift|"
    r"bulletin|annals)\b",
    re.IGNORECASE,
)
#: Extraction garbage that must never be a title.
_JUNK_CHARS_RE = re.compile(r"[\ufffd\ue000-\uf8ff\u00a4\u25a1\u25af\u25aa]")

#: Generic publication words. Used only as a *structural* hint that a very short
#: top-of-page line is a venue masthead -- never to reject a line on vocabulary
#: alone (a longer article title must also sit below it).
_VENUE_WORD_RE = re.compile(
    r"\b(?:journals?|proceedings|transactions|letters|bulletin|annals|sciences?|"
    r"studies|research|reports?|reviews?|quarterly|magazine|newsletter|frontiers)\b",
    re.IGNORECASE,
)

#: Tokens that indicate publisher/venue masthead context on page one.
_VENUE_CONTEXT_TOKENS = (
    "issn",
    "journal homepage",
    "contents lists available",
    "available online at",
    "sciencedirect",
    "elsevier",
    "springer",
    "wiley",
    "taylor & francis",
    "mdpi",
    "licensee",
    "frontiersin",
    "www.",
)
_MASTHEAD_TOP_FRACTION = 0.35
_MAX_MASTHEAD_WORDS = 8
#: A separate candidate group already differs from its neighbours by >12% in
#: font size (grouping merges similar sizes), so a real title sitting under a
#: large branded masthead may be noticeably smaller than it.
_MASTHEAD_TITLE_MIN_SIZE_RATIO = 0.8
#: The shorter, weaker masthead signals (venue word / "&" / all-caps) apply only
#: to genuinely short banners.
_MAX_GENERIC_MASTHEAD_WORDS = 5
_RUNNING_HEADER_BAND = 0.06
_TRAILING_NUMBER_RE = re.compile(r"\s\d{1,4}$")


def _looks_like_production_string(text: str) -> bool:
    """Return ``True`` for code-dominated strings (mostly digits and separators)."""
    compact = re.sub(r"\s+", "", text)
    if not compact or not any(character.isdigit() for character in compact):
        return False
    letters = sum(character.isalpha() for character in compact)
    return letters / len(compact) < 0.5


def _is_non_title_artefact(text: str) -> bool:
    """Return ``True`` for reprint notices, template labels and production codes."""
    if _JUNK_CHARS_RE.search(text):
        return True
    if _REPRINT_NOTICE_RE.search(text):
        return True
    if _TEMPLATE_LABEL_RE.search(text):
        return True
    if _PAGE_RANGE_RE.search(text):
        return True
    if _PROOF_MARKER_RE.search(text):
        return True
    if _PRODUCTION_MARKER_RE.search(text):
        return True
    if _ARTICLE_TEXT_RE.search(text):
        return True
    # A bare production code is short; a longer legitimate sentence that merely
    # contains a code-like token is not rejected on that token alone.
    if len(text.split()) <= 6 and _PRODUCTION_CODE_RE.search(text):
        return True
    if _looks_like_production_string(text):
        return True
    return False


def is_meaningful_title(text: str) -> bool:
    """Return ``True`` when ``text`` is usable as an academic title.

    The check rejects empty, obviously-artefactual and boilerplate values while
    leaving legitimate titles (including short ones) untouched.
    """
    if text is None:
        return False
    stripped = collapse_whitespace(text)
    if len(stripped) < 8 or len(stripped) > 300:
        return False

    lowered = stripped.casefold()
    if lowered in _USELESS_EXACT:
        return False
    if lowered.rstrip(".:").strip() in _SECTION_HEADINGS:
        return False

    for pattern in _USELESS_PATTERNS:
        if pattern.match(stripped):
            return False

    words = stripped.split()
    if len(words) < 2 or len(words) > 50:
        return False

    if is_email(stripped) or is_url(stripped):
        return False
    if find_doi(stripped) is not None:
        return False
    if any(token in stripped for token in _EMAIL_OR_URL_TOKENS):
        return False
    if _ISSN_RE.search(stripped):
        return False
    if _ARXIV_STAMP_RE.search(stripped):
        return False
    if _ISSUE_NUMBER_RE.search(stripped):
        return False
    if _YEAR_ONLY_RE.match(stripped) or _PAGE_NUMBER_RE.match(stripped):
        return False
    if _is_non_title_artefact(stripped):
        return False

    return True


def looks_like_boilerplate_line(text: str) -> bool:
    """Return ``True`` when a first-page line is clearly not a title."""
    stripped = collapse_whitespace(text)
    if not stripped:
        return True
    lowered = stripped.casefold()
    if lowered.rstrip(".:").strip() in _SECTION_HEADINGS:
        return True
    if _VENUE_HEADER_RE.search(stripped):
        return True
    if "all rights reserved" in lowered or "©" in stripped:
        return True
    if _ARXIV_STAMP_RE.search(stripped):
        return True
    return False


def _estimate_body_size(sizes: list[float]) -> float:
    """Estimate the page's body-text size.

    Body text is usually both the most common size and among the smallest. The
    most common size is used when it is genuinely repeated; otherwise the median
    is a safer estimate for pages with only a couple of lines.
    """
    counts = Counter(round(size, 1) for size in sizes)
    most_common_size, most_common_count = counts.most_common(1)[0]
    if most_common_count >= 2:
        return most_common_size
    return statistics.median(sizes)


def _group_consecutive(lines: list) -> list[list]:
    """Group adjacent prominent lines that share a font size.

    A new group starts when the reading order breaks or when the font size
    changes noticeably, so an author line in visibly smaller type is kept out of
    the title group while a title that wraps in the same size stays together.
    """
    groups: list[list] = []
    current: list = []
    for line in sorted(lines, key=lambda item: item.index):
        if current:
            adjacent = line.index == current[-1].index + 1
            same_size = current[0].size * 0.88 <= line.size <= current[0].size * 1.12
            if not adjacent or not same_size:
                groups.append(current)
                current = []
        current.append(line)
    if current:
        groups.append(current)
    return groups


def _is_author_list_line(text: str) -> bool:
    """Return ``True`` for an unambiguous author-list line (not a title list).

    Requires three or more name parts together with an ampersand, four or more
    parts, or a series of two-word names, so a title written as a comma list
    ("Machine Learning, Deep Learning, Reinforcement Learning") is not mistaken
    for an author line.
    """
    parts = split_author_list(text)
    if len(parts) < 3:
        return False
    if not looks_like_author_line(text, allow_all_caps=False):
        return False
    if "&" in text or len(parts) >= 4:
        return True
    return all(len(part.split()) == 2 for part in parts)


def _group_text(group: list) -> str:
    """Join a group's lines, trimming author lines merged into the title."""
    lines = list(group)
    while (
        len(lines) > 1
        and looks_like_author_line(lines[-1].text)
        and lines[-1].size < lines[0].size * 0.97
    ):
        lines.pop()
    # Some layouts place the author line directly above the title at the same
    # size; when a following line is not an author line, the leading line is the
    # author list rather than part of the title.
    while (
        len(lines) > 1
        and _is_author_list_line(lines[0].text)
        and not _is_author_list_line(lines[1].text)
    ):
        lines.pop(0)
    return collapse_whitespace(" ".join(line.text for line in lines))


@dataclass
class _Candidate:
    """One typographic title candidate on the first page."""

    text: str
    size: float
    top: float
    bottom: float
    words: int
    is_author_line: bool
    is_affiliation: bool


def _page_has_venue_context(page: PageLayout) -> bool:
    """Return ``True`` if the page carries publisher/venue masthead context."""
    for line in page.lines:
        lowered = collapse_whitespace(line.text).casefold()
        if any(token in lowered for token in _VENUE_CONTEXT_TOKENS):
            return True
        if _ISSUE_NUMBER_RE.search(line.text) or _ISSN_RE.search(line.text):
            return True
    return False


def _looks_like_masthead(candidate: _Candidate, page_has_venue_context: bool, others: list[_Candidate]) -> bool:
    """Return ``True`` when a candidate is a venue/journal masthead, not a title.

    Structural signals only: a short line near the top, a clearly longer
    article-like title immediately below it at comparable or larger prominence,
    and masthead context (venue/volume/ISSN/publisher text, or a short generic
    publication name / all-caps banner). No journal-name list is used.
    """
    if candidate.words > _MAX_MASTHEAD_WORDS and not _VENUE_LABEL_RE.search(candidate.text):
        return False
    if candidate.top > _MASTHEAD_TOP_FRACTION:
        return False

    min_title_size = candidate.size * _MASTHEAD_TITLE_MIN_SIZE_RATIO - 1e-6
    longer_title_below = False
    for other in others:
        if other is candidate or other.is_author_line or other.is_affiliation:
            continue
        # "More title-like": clearly longer text anywhere on the page, or a
        # comparable-prominence line below the candidate that carries more words.
        if other.words >= candidate.words + 2:
            longer_title_below = True
            break
        if (
            other.top > candidate.top
            and other.words > candidate.words
            and other.size >= min_title_size
        ):
            longer_title_below = True
            break
    if not longer_title_below:
        return False

    if page_has_venue_context or _VENUE_LABEL_RE.search(candidate.text):
        return True

    text = candidate.text
    # A very short generic publication name, or an all-caps banner.
    if candidate.words <= _MAX_GENERIC_MASTHEAD_WORDS and (
        _VENUE_WORD_RE.search(text) or "&" in text or text == text.upper()
    ):
        return True
    return False


def looks_like_venue_masthead(text: str, *, page_title: str | None) -> bool:
    """Return ``True`` when ``text`` is a venue masthead rather than a title.

    Some title candidates carry no page geometry -- notably the embedded PDF
    metadata title, which journal templates routinely fill with the journal or
    publisher name. The same conjunction of signals used for rendered lines is
    applied here: a publication label inside the string, an accepted first-page
    article title that does not share its wording, and a first-page title that is
    not *less* prominent than the metadata string.
    """
    stripped = collapse_whitespace(text)
    if not stripped or page_title is None:
        return False
    if not _VENUE_LABEL_RE.search(stripped):
        return False
    accepted = collapse_whitespace(page_title)
    if normalize_for_comparison(stripped) == normalize_for_comparison(accepted):
        return False
    return len(accepted) >= len(stripped) or len(stripped.split()) <= _MAX_MASTHEAD_WORDS


def _looks_like_running_header(candidate: _Candidate) -> bool:
    """Return ``True`` for a page-edge line ending in a bare page number."""
    in_margin_band = (
        candidate.top <= _RUNNING_HEADER_BAND
        or candidate.bottom >= 1.0 - _RUNNING_HEADER_BAND
    )
    return in_margin_band and bool(_TRAILING_NUMBER_RE.search(candidate.text))


def extract_first_page_title(page: PageLayout) -> str | None:
    """Return a conservative title candidate from page one, or ``None``.

    The candidate must be typographically prominent (larger than the page's body
    text), sit in the upper part of the page, and survive the artefact,
    boilerplate and :func:`is_meaningful_title` checks. Consecutive prominent
    lines are joined so multi-line titles are recovered. When the most prominent
    string is rejected as a masthead, production identifier or running header,
    the extractor continues to the next plausible candidate rather than giving up.
    """
    if not page.lines:
        return None

    sizes = [line.size for line in page.lines if line.size > 0]
    if not sizes:
        return None
    body_size = _estimate_body_size(sizes)
    if body_size <= 0:
        return None

    prominent = [
        line
        for line in page.lines
        if line.size >= body_size * 1.15 and line.top_fraction <= 0.60
    ]
    if not prominent:
        return None

    candidates: list[_Candidate] = []
    for group in _group_consecutive(prominent):
        text = _group_text(group)
        if not is_meaningful_title(text):
            continue
        if looks_like_boilerplate_line(text):
            continue
        candidates.append(
            _Candidate(
                text=text,
                size=sum(line.size for line in group) / len(group),
                top=min(line.top_fraction for line in group),
                bottom=max(line.bottom_fraction for line in group),
                words=len(text.split()),
                is_author_line=looks_like_author_line(text),
                is_affiliation=looks_like_affiliation(text),
            )
        )
    if not candidates:
        return None

    venue_context = _page_has_venue_context(page)
    viable = [
        candidate
        for candidate in candidates
        if not _looks_like_running_header(candidate)
        and not _looks_like_masthead(candidate, venue_context, candidates)
    ]
    if not viable:
        return None

    viable.sort(key=lambda candidate: (candidate.size, -candidate.top, len(candidate.text)), reverse=True)
    best = viable[0]
    if len(viable) > 1:
        second = viable[1]
        # Two equally-prominent, similarly-positioned candidates make the title
        # genuinely ambiguous, so decline to choose one.
        size_close = second.size >= best.size * 0.95
        top_close = abs(second.top - best.top) < 0.05
        if size_close and top_close and second.text != best.text:
            return None
    return best.text
