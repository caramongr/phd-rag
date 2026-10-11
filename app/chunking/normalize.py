"""Normalization for retrieval, without changing meaning (RAG-007).

Every rule here is deterministic and reversible in principle, and the raw
extraction is never destroyed: the parsers keep ``ParsedDocument.raw_text`` and
each page keeps its ``native_text`` / ``ocr_text``, while chunks carry the
``source_locator`` of every parsed item that produced them. This module only
produces the *normalized* text used for chunking and retrieval.

Allowed (per the RAG-007 task): Unicode normalization, line-ending normalization,
whitespace normalization, dehyphenation of line-wrap hyphens, joining wrapped
prose lines, removal of confidently detected running headers/footers, and
normalization of non-breaking spaces.

Deliberately **not** allowed here: rewriting sentences, grammar or spelling
correction, summarisation, paraphrasing, removing citations, changing
terminology, translating, or any LLM involvement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

#: Bump when a rule changes, so stored chunks can be traced to a rule set.
NORMALIZATION_VERSION = "rag007-normalize-v1"

#: Typographic ligatures that PDF text layers routinely contain. Folding them is
#: a spelling-preserving normalization ("deﬁnition" -> "definition").
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u0132": "IJ",
    "\u0133": "ij",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00c6": "AE",
    "\u00e6": "ae",
}

#: Space-like characters that must become a plain space, and zero-width
#: characters that carry no content at all.
_SPACE_LIKE = {
    "\u00a0",  # no-break space
    "\u1680",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",  # figure space
    "\u2008",
    "\u2009",
    "\u200a",
    "\u202f",  # narrow no-break space
    "\u205f",
    "\u3000",  # ideographic space
}
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}

_TRANSLATION = {ord(character): " " for character in _SPACE_LIKE}
_TRANSLATION.update({ord(character): "" for character in _ZERO_WIDTH})
_TRANSLATION.update({ord(character): replacement for character, replacement in _LIGATURES.items()})

#: Hyphen characters a line-wrap break may use.
_WRAP_HYPHENS = "-\u2010\u2011"

#: A line ending a wrapped word: a letter, then a wrap hyphen, then end of line.
_WRAP_HYPHEN_END_RE = re.compile(rf"(?<=[^\W\d_])([{_WRAP_HYPHENS}])$")
_WORD_BEFORE_HYPHEN_RE = re.compile(rf"([^\W\d_]+)[{_WRAP_HYPHENS}]$")
_FIRST_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Lines that start a new list item and must not be glued to the previous line.
_LIST_MARKER_RE = re.compile(
    r"^(?:[\-\u2010\u2011\u2022\u25cf\u25aa\u25a0*+]\s|\d{1,3}[.)]\s|[a-zA-Z][.)]\s)"
)

#: A standalone page number, or a "Page 3 of 12" style running line. Digit runs
#: are masked before repetition counting, so these match without a special rule.
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$", re.IGNORECASE
)

#: Minimum prefix length before a wrap hyphen is considered a split word.
_MIN_HYPHEN_PREFIX = 3

_WHITESPACE_RUN_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")
_DIGITS_RE = re.compile(r"\d+")


def normalize_unicode(text: str) -> str:
    """Apply NFC and fold ligatures and space-like/zero-width characters."""
    return unicodedata.normalize("NFC", text.translate(_TRANSLATION))


def normalize_line_endings(text: str) -> str:
    """Normalise CRLF and CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace and trim each line."""
    lines = [_WHITESPACE_RUN_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINE_RUN_RE.sub("\n\n", "\n".join(lines)).strip()


def normalize_inline(text: str) -> str:
    """Normalize a single-line value (heading, table cell, title)."""
    return collapse_whitespace(normalize_unicode(normalize_line_endings(text)))


def dehyphenate_lines(lines: Sequence[str]) -> list[str]:
    """Join lines broken by a line-wrap hyphen, conservatively.

    A hyphen is treated as a wrap break only when it is the last character of a
    line, is preceded by a letter, the fragment before it is at least
    ``_MIN_HYPHEN_PREFIX`` characters long, and the next line starts with a
    lowercase letter. Genuine compounds are therefore preserved unless they
    happen to be split exactly at their hyphen.
    """
    joined: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if index + 1 < len(lines) and _WRAP_HYPHEN_END_RE.search(current):
            following = lines[index + 1]
            prefix_match = _WORD_BEFORE_HYPHEN_RE.search(current)
            prefix = prefix_match.group(1) if prefix_match else ""
            if (
                len(prefix) >= _MIN_HYPHEN_PREFIX
                and following
                and following[0].islower()
            ):
                joined.append(current[:-1] + following.lstrip())
                index += 2
                continue
        joined.append(current)
        index += 1
    return joined


def join_wrapped_lines(lines: Sequence[str]) -> str:
    """Join wrapped prose lines into paragraphs.

    Lines are merged with a single space; a newline is preserved only before a
    list marker, which keeps list structure intact instead of flattening it.
    """
    paragraphs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if paragraphs and not _LIST_MARKER_RE.match(stripped):
            paragraphs[-1] = f"{paragraphs[-1]} {stripped}"
        else:
            paragraphs.append(stripped)
    return "\n".join(paragraphs)


def normalize_block_text(text: str) -> str:
    """Normalize one prose block: endings, Unicode, dehyphenation, line wrapping."""
    if not text:
        return ""
    normalized = normalize_unicode(normalize_line_endings(text))
    lines = dehyphenate_lines([line.strip() for line in normalized.split("\n")])
    return collapse_whitespace(join_wrapped_lines(lines))


@dataclass(frozen=True)
class RunningLineReport:
    """The running headers/footers detected for one document."""

    lines: frozenset[str]
    pages_considered: int
    threshold: int

    @property
    def detected(self) -> bool:
        """Whether any running line was detected."""
        return bool(self.lines)


def _comparison_form(line: str) -> str:
    """Normalize a candidate line for repetition counting (digits masked)."""
    return _DIGITS_RE.sub("#", collapse_whitespace(normalize_unicode(line)).casefold())


def detect_running_lines(
    page_texts: Sequence[tuple[int, str]],
    *,
    min_pages: int = 3,
    ratio: float = 0.5,
    max_chars: int = 160,
) -> RunningLineReport:
    """Detect repeated first/last page lines (running headers and footers).

    Only the first and last non-empty line of each page are candidates, which
    keeps the detection conservative; a candidate must repeat on at least
    ``max(min_pages, ratio * pages)`` pages to count. Digit runs are masked, so
    "Page 3" and "Page 4" are recognised as the same running line.
    """
    pages = [(number, text) for number, text in page_texts if text and text.strip()]
    pages_considered = len(pages)
    if pages_considered < min_pages:
        return RunningLineReport(frozenset(), pages_considered, 0)

    threshold = max(min_pages, int(round(ratio * pages_considered)))
    counts: dict[str, int] = {}
    original: dict[str, str] = {}
    for _number, text in pages:
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            continue
        for line in {lines[0], lines[-1]}:
            stripped = line.strip()
            if not stripped or len(stripped) > max_chars:
                continue
            key = _comparison_form(stripped)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            original.setdefault(key, stripped)

    running = {
        original[key]
        for key, count in counts.items()
        if count >= threshold
    }
    return RunningLineReport(frozenset(running), pages_considered, threshold)


def is_running_line(line: str, report: RunningLineReport) -> bool:
    """Whether ``line`` is one of the detected running headers/footers."""
    if not report.lines:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in report.lines:
        return True
    comparison = _comparison_form(stripped)
    return any(_comparison_form(candidate) == comparison for candidate in report.lines)


def strip_running_lines(text: str, report: RunningLineReport) -> str:
    """Remove leading and trailing running header/footer lines from a block."""
    if not report.lines or not text:
        return text
    lines = text.split("\n")
    start = 0
    end = len(lines)
    while start < end and is_running_line(lines[start], report):
        start += 1
    while end > start and is_running_line(lines[end - 1], report):
        end -= 1
    if start == 0 and end == len(lines):
        return text
    return "\n".join(lines[start:end]).strip()
