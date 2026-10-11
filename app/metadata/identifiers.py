"""Identifier helpers: arXiv ids and filename-derived hints (RAG-006).

Filename metadata is a weak fallback only. It can suggest an author-like token,
a year-like token or an arXiv identifier, but it can never become an
authoritative title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: New-style arXiv identifiers (e.g. ``1703.10593`` or ``2308.04079``), with an
#: optional ``vN`` revision suffix that is not part of the normalized id.
_ARXIV_NEW_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d.])")
#: Old-style arXiv identifiers (e.g. ``hep-th/9901001``).
_ARXIV_OLD_RE = re.compile(r"\b([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
"""Regex matching a token that is exactly a plausible year."""


def _is_id_like(segment: str) -> bool:
    """Return ``True`` for a year, arXiv id, or long numeric identifier."""
    if _YEAR_RE.fullmatch(segment):
        return True
    if find_arxiv_id(segment) is not None:
        return True
    return re.fullmatch(r"\d{4,}", segment) is not None


def _author_token_from_stem(stem: str) -> str | None:
    """Return an author-like token only for ``name-year``/``name-id`` filenames.

    The documented academic filename schemes are ``kerbl-2023.pdf`` and
    ``Zhu-1703.10593v7.pdf``: an author token followed by a separator and a
    year or identifier. A filename that simply spells a title
    (``Attention_Is_All_You_Need``) therefore yields no author token.
    """
    parts = re.split(r"[-_ ]+", stem)
    for index, part in enumerate(parts[:-1]):
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\u2019]{1,29}", part):
            continue
        if part.casefold() in _FILENAME_STOPWORDS:
            continue
        if _is_id_like(parts[index + 1]):
            return part
    return None

#: Filename words that are not author-like tokens.
_FILENAME_STOPWORDS = frozenset(
    {
        "paper",
        "draft",
        "final",
        "thesis",
        "chapter",
        "article",
        "book",
        "report",
        "review",
        "copy",
        "version",
        "preprint",
        "manuscript",
        "fig",
        "figure",
        "table",
        "notes",
        "summary",
        "index",
        "main",
        "text",
        "document",
        "doc",
        "new",
        "old",
        "data",
        "appendix",
        "supplement",
        "enriched",
        "full",
        "part",
        "vol",
        "issue",
    }
)


@dataclass
class FilenameMetadata:
    """Weak, low-confidence hints derived from a filename."""

    stem: str
    author_token: str | None = None
    years: list[int] = field(default_factory=list)
    arxiv_id: str | None = None


def normalize_arxiv_id(raw: str | None) -> str | None:
    """Return a normalized arXiv id (without the ``vN`` suffix), or ``None``."""
    if not raw:
        return None
    text = raw.strip()
    new_match = _ARXIV_NEW_RE.search(text)
    if new_match is not None:
        return new_match.group(1)
    old_match = _ARXIV_OLD_RE.search(text)
    if old_match is not None:
        return old_match.group(1)
    return None


def find_arxiv_id(text: str) -> str | None:
    """Return the first arXiv id found in ``text``, or ``None``."""
    return normalize_arxiv_id(text) if text else None


def parse_filename_metadata(filename: str) -> FilenameMetadata:
    """Derive low-confidence hints from a filename.

    Any arXiv id embedded in the name is removed before year extraction so that
    an arXiv number is never mistaken for a publication year.
    """
    stem = Path(filename).stem
    arxiv_id = find_arxiv_id(stem)

    remaining = stem
    if arxiv_id is not None:
        remaining = remaining.replace(arxiv_id, " ")

    years = sorted(
        {
            int(match.group(1))
            for match in _YEAR_RE.finditer(remaining)
            if 1900 <= int(match.group(1)) <= 2100
        }
    )

    return FilenameMetadata(
        stem=stem,
        author_token=_author_token_from_stem(stem),
        years=years,
        arxiv_id=arxiv_id,
    )
