"""Conservative author extraction (RAG-006).

Author *ordering* is preserved. ``full_name`` is always authoritative and
given/family parts are only split when the split is reliable; otherwise the
parts stay ``None``. Affiliations are never misinterpreted as authors.
"""

from __future__ import annotations

import re

from app.metadata.layout import PageLayout
from app.metadata.models import Author
from app.metadata.text import SECTION_HEADINGS, collapse_whitespace, normalize_for_comparison

#: Institution markers that mark a line as an affiliation rather than an author
#: list. Matched with word boundaries; the generic nouns (school/college/center/
#: academy) additionally require "of"/"for", so a *title* such as "A Single
#: Center Student Survey" is not mistaken for an affiliation.
_AFFILIATION_RE = re.compile(
    r"\buniversit[a-zäöüß]+\b"
    r"|\bdepartments?\b|\bdept\.?\b"
    r"|\binstitut(?:e|es|ion|ions|ional)?\b"
    r"|\blaborator(?:y|ies)\b"
    r"|\bschool\s+of\b|\bfaculty\s+of\b|\bcollege\s+of\b|\bacademy\s+of\b"
    r"|\bcent(?:er|re)\s+(?:for|of)\b"
    r"|\bhospital\b"
    r"|\bcorporation\b|\binc\.?\b|\bltd\.?\b|\bgmbh\b"
    r"|\bresearch\s+group\b"
    r"|@",
    re.IGNORECASE,
)

#: A leading conjunction or "by" before a name ("and Jane Doe", "by J. Smith").
_LEADING_CONJUNCTION_RE = re.compile(
    r"^(?:and|or|&|by)\s+", re.IGNORECASE
)

#: Trailing markers such as footnote digits or asterisks attached to a name.
_TRAILING_MARKER_RE = re.compile(r"[\d*†‡§¶]+$")
_NAME_ALLOWED_RE = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.'\- ]*$")
_HONORIFICS = frozenset(
    {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "sir", "phd"}
)


def looks_like_affiliation(text: str) -> bool:
    """Return ``True`` when ``text`` reads like an institution, not a person."""
    return bool(_AFFILIATION_RE.search(text))


def _clean_name_token(token: str) -> str:
    """Strip footnote markers and surrounding punctuation from a name token."""
    cleaned = token.strip().strip(",;:·•-–—").strip()
    cleaned = _TRAILING_MARKER_RE.sub("", cleaned).strip()
    cleaned = _LEADING_CONJUNCTION_RE.sub("", cleaned).strip()
    return collapse_whitespace(cleaned)


#: Repository/cover-sheet boilerplate that sits next to author blocks on
#: first pages but never names a person.
_BOILERPLATE_NAME_RE = re.compile(
    r"\b(?:document|accepted|published|final|submitted)\s+version\b"
    r"|\b(?:author\s+)?accepted\s+(?:author\s+)?manuscript\b"
    r"|\bversion\s+of\s+record\b|\bpeer[-\s]?reviewed\b|\bpre[-\s]?prints?\b"
    r"|\bto\s+cite\s+this\s+version\b|\bcitation\s+for\s+published\s+version\b"
    r"|\bpublished\s+in\b|\bopen\s+access\b|\bcreative\s+commons\b"
    r"|\ball\s+rights\s+reserved\b|\busage\s+rights\b|\bin\s+copyright\b"
    r"|\bpublisher'?s\s+pdf\b|\balso\s+known\s+as\b"
    r"|\breview\s+article\b|\bresearch\s+article\b|\boriginal\s+article\b"
    r"|\bsystematic\s+review\b|\beditorial\b|\bcopyright\b|\bunknown\b"
    r"|\b(?:original|proceeding|special\s+issue|full|short)\s+paper\b"
    r"|\bedited\s+by\b|\bpublication\s+date\b|\bciting\s+this\s+paper\b"
    r"|\bbriefing\s+no\b|\bdoi\b|\blicense[sd]?\b"
    r"|\bpublished\s+as\s+\w+",
    re.IGNORECASE,
)

#: Name particles that legitimately appear lower-cased inside a name.
_NAME_PARTICLES = frozenset(
    {
        "de", "del", "della", "den", "der", "di", "da", "dos", "du", "el", "al",
        "la", "le", "van", "von", "bin", "binti", "bte", "ter", "ten", "op",
        "ben", "abd", "abu", "e", "y", "i", "st", "mac", "mc", "o",
    }
)

#: Function words and generic publication words that never form a person's name.
_NON_NAME_RE = re.compile(
    r"^(?:this|that|these|those|the|there|it|we|our|their|its|in|on|of|for|with|from|"
    r"as|at|to|a|an)\b"
    r"|\b(?:journal|journals|publishing|proceedings|conference|congress|symposium|"
    r"workshop|meeting|edition|volume|chapter|university|college|school|department|"
    r"institute|academy|faculty|laboratory|hospital|education|medical|medicine|"
    r"sciences?|research|studies)\b",
    re.IGNORECASE,
)

#: Roman-numeral section labels ("I. INTRODUCTION") that precede an all-caps
#: heading; a person's name is never written this way.
_SECTION_LABEL_RE = re.compile(
    r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z][A-Z\s]*$"
)

#: A sentence boundary inside a string ("knowledge. Generally") marks prose.
_SENTENCE_BREAK_RE = re.compile(r"[a-z]\.\s+[A-Z]")


#: Parenthetical annotations attached to author names ("(Burciu)", "(B)"); they
#: are ignored when validating a name so a real author line is still recognised,
#: but are kept in the stored value.
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")


def is_plausible_author_name(name: str) -> bool:
    """Return ``True`` when ``name`` is plausibly a single person's name."""
    cleaned = _clean_name_token(name)
    if not cleaned or any(ch.isdigit() for ch in cleaned):
        return False
    checks = collapse_whitespace(_PARENTHETICAL_RE.sub(" ", cleaned)).strip()
    if not checks or any(ch.isdigit() for ch in checks):
        return False
    if _BOILERPLATE_NAME_RE.search(checks):
        return False
    if _NON_NAME_RE.search(checks):
        return False
    if _SECTION_LABEL_RE.match(checks):
        return False
    if checks.casefold().rstrip(".:").strip() in SECTION_HEADINGS:
        return False
    if not _NAME_ALLOWED_RE.match(checks):
        return False
    if looks_like_affiliation(checks):
        return False
    tokens = checks.split()
    if not (1 <= len(tokens) <= 6):
        return False
    if len(tokens) == 1 and len(checks) < 3:
        return False
    # Reject all-lowercase multi-word strings (unlikely to be a name).
    if len(tokens) > 1 and checks == checks.casefold():
        return False
    # Multi-token names are capitalised words (or initials) plus known
    # particles; two or more lower-cased content words indicate prose rather
    # than a name. A sentence boundary inside the string is prose too.
    if len(tokens) > 1:
        lowercase_content = 0
        for token in tokens:
            core = token.strip(".'-()[]")
            if not core or core.casefold() in _NAME_PARTICLES:
                continue
            if core[0].isupper() or core.isupper():
                continue
            lowercase_content += 1
        if lowercase_content >= 2:
            return False
    if _SENTENCE_BREAK_RE.search(checks):
        return False
    return True


def _to_author(name: str) -> Author:
    """Build an :class:`Author`, splitting given/family only when reliable."""
    cleaned = _clean_name_token(name)
    given: str | None = None
    family: str | None = None

    if "," in cleaned:
        family_part, _, given_part = cleaned.partition(",")
        family_part = family_part.strip()
        given_part = given_part.strip()
        if (
            family_part
            and given_part
            and not given_part.isdigit()
            and "," not in given_part
        ):
            family = family_part
            given = given_part or None
    else:
        tokens = cleaned.split()
        if len(tokens) == 2 and all(
            token.casefold() not in _HONORIFICS for token in tokens
        ):
            given, family = tokens[0], tokens[1]

    return Author(full_name=cleaned, given_name=given, family_name=family)


def _token_count(name: str) -> int:
    return len(_clean_name_token(name).split())


def split_author_list(raw: str) -> list[str]:
    """Split a raw author string into individual names, preserving order.

    Semicolons are an unambiguous separator. ``and``/``&`` are treated as
    separators too. A comma only separates authors when *every* comma-separated
    piece looks like a full ``Given Family`` name (two or more tokens); a
    ``Family, Given`` pair is therefore kept as one author rather than split into
    a bare surname and a bare given name.
    """
    text = collapse_whitespace(raw.replace("\n", " "))
    if not text:
        return []

    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]

    normalized = re.sub(r",?\s+and\s+", ", ", text, flags=re.IGNORECASE)
    normalized = re.sub(r",?\s*&\s*", ", ", normalized)
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) > 1 and all(_token_count(part) >= 2 for part in parts):
        return parts
    return [text]


def parse_author_string(raw: str) -> list[Author]:
    """Parse a raw author string into ordered :class:`Author` objects.

    A token that is not a plausible name (for example an affiliation) causes the
    whole string to be rejected, returning an empty list. This keeps the
    extraction conservative rather than half-correct.
    """
    names = split_author_list(raw)
    if not names:
        return []

    authors: list[Author] = []
    for name in names:
        if not is_plausible_author_name(name):
            return []
        authors.append(_to_author(name))
    return authors


def extract_authors_from_first_page(
    page: PageLayout, title: str | None
) -> list[Author] | None:
    """Return author candidates from page one, or ``None`` when uncertain.

    Only the first consecutive run of author-looking lines below the title is
    considered. If any line in that run fails validation the whole extraction is
    abandoned, so a partially-correct author list is never returned.
    """
    if not page.lines:
        return None
    title_key = collapse_whitespace(title).casefold() if title else None
    title_norm = ""
    if title_key:
        title_norm = f" {normalize_for_comparison(title_key)} "

    def is_title_line(text: str) -> bool:
        if title_norm and f" {normalize_for_comparison(text)} " in title_norm:
            return True
        return title_key is not None and text.casefold() == title_key

    candidates: list[str] = []
    for line in sorted(page.lines, key=lambda item: item.index):
        text = collapse_whitespace(line.text)
        if not text or line.top_fraction > 0.75:
            continue
        # The title usually spans several lines; skip any line that is part of
        # the title rather than emitting a title fragment as an author. Such a
        # line is not an affiliation stop signal either, because the title
        # extractor may have grouped a university name into a multi-line title.
        if is_title_line(text):
            continue
        if looks_like_affiliation(text):
            break
        if looks_like_author_line(text, allow_all_caps=True):
            candidates.append(text)
            if len(candidates) >= 3:
                break
        elif candidates:
            break
    if not candidates:
        return None

    authors: list[Author] = []
    for candidate in candidates:
        parsed = parse_author_string(candidate)
        if not parsed:
            return None
        authors.extend(parsed)
    return authors or None


def _is_all_caps_banner(text: str) -> bool:
    """Return ``True`` for a multi-word all-caps string (heading-like)."""
    tokens = text.split()
    return len(tokens) >= 3 and text == text.upper()


#: Function words that appear in headings and titles but essentially never in a
#: person's name; used only to separate all-caps names from all-caps headings.
_HEADING_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "based", "between", "by", "for",
        "from", "in", "into", "is", "its", "of", "on", "or", "the", "their",
        "these", "this", "to", "toward", "towards", "using", "via", "with",
    }
)


def _all_caps_banner_is_name_like(text: str) -> bool:
    """Return ``True`` for an all-caps string that reads as names, not a heading."""
    tokens = text.split()
    if any(len(token) == 1 for token in tokens):
        return False
    return not any(token.casefold() in _HEADING_STOPWORDS for token in tokens)


def looks_like_author_line(text: str, *, allow_all_caps: bool = False) -> bool:
    """Return ``True`` when a line looks like a list of author names.

    ``allow_all_caps`` is used where the surrounding context already rules out a
    heading (author extraction below a known title), so all-caps author lists are
    accepted when they read as names. Title-side callers leave it ``False``
    because an all-caps banner is more likely to be a title line.
    """
    if len(text) > 200 or text.endswith((".", "!", "?")):
        return False
    parts = split_author_list(text)
    if not parts:
        return False
    if _is_all_caps_banner(text) and not (
        allow_all_caps and _all_caps_banner_is_name_like(text)
    ):
        return False
    if len(parts) == 1:
        # A single author is only accepted as ``Given Family`` (at least two
        # tokens); a bare single word is far more often a section label.
        return 2 <= _token_count(parts[0]) <= 4 and is_plausible_author_name(parts[0])
    return all(is_plausible_author_name(part) for part in parts)
