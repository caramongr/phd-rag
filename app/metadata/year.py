"""Conservative publication-year extraction (RAG-006).

A four-digit number is only accepted as a publication year when its context
supports it, and when several distinct years appear the extractor returns
``None`` rather than guessing. A PDF creation date is never treated as a
publication year here; that weaker signal is handled separately with ``LOW``
confidence and a lower resolution priority.
"""

from __future__ import annotations

import re

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_COPYRIGHT_RE = re.compile(r"(?:©|\(c\)|copyright)\s*((?:19|20)\d{2})", re.IGNORECASE)
_PUBLISHED_RE = re.compile(
    r"\b(?:published|publication|appeared|presented at)\b[^\n]{0,40}?((?:19|20)\d{2})",
    re.IGNORECASE,
)
_PDF_DATE_RE = re.compile(r"D:(\d{4})")

_MIN_YEAR = 1900
_MAX_YEAR = 2100


def _valid(year: int) -> bool:
    return _MIN_YEAR <= year <= _MAX_YEAR


def extract_year_from_text(text: str) -> int | None:
    """Return an unambiguous publication year from ``text``, or ``None``.

    A single distinct year is accepted. When several distinct years are present
    the extractor only accepts a year that is explicitly anchored to a
    copyright or publication marker; otherwise it declines to guess.
    """
    if not text:
        return None

    anchored = _COPYRIGHT_RE.search(text) or _PUBLISHED_RE.search(text)
    if anchored is not None:
        year = int(anchored.group(1))
        if _valid(year):
            return year

    years = {int(match.group(1)) for match in _YEAR_RE.finditer(text)}
    years = {year for year in years if _valid(year)}
    if len(years) == 1:
        return years.pop()
    return None


def year_from_pdf_date(raw: str | None) -> int | None:
    """Extract the year from a PDF ``creationDate``/``modDate`` string."""
    if not raw:
        return None
    match = _PDF_DATE_RE.search(raw)
    if match is None:
        return None
    year = int(match.group(1))
    return year if _valid(year) else None


def year_from_datetime(value) -> int | None:
    """Extract the year from a ``datetime``-like value, if any."""
    if value is None:
        return None
    year = getattr(value, "year", None)
    if isinstance(year, int) and _valid(year):
        return year
    return None
