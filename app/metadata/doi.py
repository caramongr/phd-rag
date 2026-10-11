"""DOI detection and normalization (RAG-006).

The extractor is deliberately conservative: it recognises the canonical DOI
grammar (``10.<registrant>/<suffix>``), strips common wrappers such as
``https://doi.org/`` or ``doi:``, and validates the result before accepting it.
It never treats an arbitrary string beginning with ``10.`` as a DOI.
"""

from __future__ import annotations

import re

#: Canonical DOI shape: a ``10.`` prefix, a 4-9 digit registrant, a slash and a
#: non-empty suffix of printable-ASCII DOI characters. Restricting the suffix to
#: ASCII keeps trailing extraction garbage (e.g. a replacement character from a
#: bad PDF encoding) from being swallowed into the DOI.
_DOI_RE = re.compile(
    r"10\.\d{4,9}/[A-Za-z0-9._;()/:\-\[\]+%#?&=~*!,@$]+",
    re.IGNORECASE,
)

#: Wrappers that may prefix a DOI string and must be removed before validation.
_WRAPPER_RE = re.compile(
    r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*|doi\b\s*)",
    re.IGNORECASE,
)

#: Trailing characters that are almost always sentence punctuation, not part of
#: the DOI suffix.
_TRAILING_CHARS = ".,;:)\\]}>\"'"

_VALID_SUFFIX_RE = re.compile(r"[A-Za-z0-9]")


def normalize_doi(raw: str) -> str | None:
    """Normalize ``raw`` to a canonical DOI, or return ``None`` if invalid.

    Wrappers (``https://doi.org/``, ``http://dx.doi.org/``, ``doi:``) are
    removed, surrounding whitespace is stripped, and trailing sentence
    punctuation is trimmed. The returned value preserves the original casing of
    the suffix because DOI casing is not meaningful but the published form is
    what readers expect to see.
    """
    if raw is None:
        return None
    candidate = raw.strip()
    candidate = _WRAPPER_RE.sub("", candidate)
    candidate = candidate.strip()
    candidate = candidate.rstrip(_TRAILING_CHARS)

    match = _DOI_RE.fullmatch(candidate)
    if match is None:
        return None

    doi = match.group(0)
    registrant, _, suffix = doi.partition("/")
    if "." not in registrant or not suffix:
        return None
    # Reject a suffix that carries no alphanumeric content (e.g. only dots).
    if _VALID_SUFFIX_RE.search(suffix) is None:
        return None
    return doi


def find_doi(text: str) -> str | None:
    """Return the first valid DOI in ``text``, or ``None``.

    Scanning is left-to-right so the result is deterministic for a given input.
    """
    if not text:
        return None
    for match in _DOI_RE.finditer(text):
        normalized = normalize_doi(match.group(0))
        if normalized is not None:
            return normalized
    return None


def find_all_dois(text: str) -> list[str]:
    """Return every valid DOI in ``text`` in first-seen order (de-duplicated)."""
    if not text:
        return []
    seen: list[str] = []
    for match in _DOI_RE.finditer(text):
        normalized = normalize_doi(match.group(0))
        if normalized is not None and normalized not in seen:
            seen.append(normalized)
    return seen


def doi_cache_key(doi: str) -> str:
    """Return a case-insensitive cache key for a DOI."""
    return doi.strip().casefold()
