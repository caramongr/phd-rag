"""Small, dependency-free text helpers for metadata extraction.

These helpers deliberately avoid any clever normalisation that could damage a
legitimate value. They only fold whitespace, case, accents and punctuation for
*comparison* purposes; the stored value is always the extracted original.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

#: Section headings that are structurally prominent but are never a title and
#: never a person's name.
SECTION_HEADINGS: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "methods",
        "methodology",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "keywords",
        "index terms",
        "contents",
        "table of contents",
        "summary",
        "literature review",
        "related work",
        "related works",
        "theoretical background",
        "state of the art",
        "research questions",
    }
)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) into single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def fold_accents(text: str) -> str:
    """Return ``text`` with combining accents removed (for comparison only)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_for_comparison(text: str) -> str:
    """Return a folded, punctuation-free, lower-cased key for comparison."""
    folded = fold_accents(text).lower()
    folded = _NON_ALNUM_RE.sub(" ", folded)
    return collapse_whitespace(folded)


def is_email(text: str) -> bool:
    """Return ``True`` when ``text`` contains an e-mail address."""
    return bool(_EMAIL_RE.search(text))


def is_url(text: str) -> bool:
    """Return ``True`` when ``text`` contains a URL."""
    return bool(_URL_RE.search(text))


def title_similarity(left: str, right: str) -> float:
    """Return a 0..1 similarity between two titles.

    The score blends a whole-string ratio with a token-set ratio so that word
    reordering or a dropped subtitle does not collapse the score. It is only
    used to decide whether an external record is a plausible match, never to
    alter a stored value.
    """
    a = normalize_for_comparison(left)
    b = normalize_for_comparison(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    sequence = SequenceMatcher(None, a, b).ratio()

    tokens_a = set(a.split())
    tokens_b = set(b.split())
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0

    return max(sequence, jaccard)
