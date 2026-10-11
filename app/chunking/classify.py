"""Content classification for chunks (RAG-007).

Classification uses the *structural* signals the parsers already provide — the
heading trail, block types, table items — plus a small set of conservative text
patterns. No prose semantics are inferred: a section is a conclusion because its
heading says so, never because its sentences sound conclusive.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Sequence

from app.chunking.normalize import normalize_inline

#: Heading numbering such as "4", "4.2", "4.2.1" or "A.3" at the start of a title.
_NUMBER_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z]\.\d+(?:\.\d+)*)[.)]?\s+")

_REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "reference",
        "references cited",
        "references and notes",
        "bibliography",
        "bibliographies",
        "works cited",
        "literature cited",
        "literature",
        "r\u00e9f\u00e9rences",
        "bibliographie",
        "literaturverzeichnis",
        "referencias",
        "bibliograf\u00eda",
        "refer\u00eancias",
    }
)

_ABSTRACT_HEADINGS = frozenset(
    {
        "abstract",
        "abstract and keywords",
        "summary",
        "executive summary",
        "zusammenfassung",
        "r\u00e9sum\u00e9",
        "resumen",
        "resumo",
    }
)

_CONCLUSION_HEADINGS = frozenset(
    {
        "conclusion",
        "conclusions",
        "concluding remarks",
        "conclusions and future work",
        "conclusion and future work",
        "conclusions and recommendations",
        "conclusions and implications",
        "discussion and conclusion",
        "discussion and conclusions",
        "discussion, conclusion and future work",
        "summary and conclusion",
        "summary and conclusions",
        "conclusions and outlook",
    }
)

_APPENDIX_HEADING_RE = re.compile(r"^(?:appendix|appendices|annex)\b", re.IGNORECASE)

#: "Figure 3: caption text", "Table 2. caption text", "Fig. 4 - caption text".
_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?|tab(?:le)?|scheme|chart|exhibit|box)\s*\.?\s*"
    r"(?:\d+[a-z]?|[ivxlc]+)\s*[:.\u2013\u2014-]\s+\S",
    re.IGNORECASE,
)

#: A block whose own first line introduces a reference list ("References" alone,
#: in a block that also contains the first entries).
_REFERENCE_INTRO_RE = re.compile(
    r"^(?:references|bibliography|works cited|literature cited)\s*:?\s*$",
    re.IGNORECASE,
)

_MAX_CAPTION_CHARS = 500


class SectionKind(StrEnum):
    """What a heading trail says about the content beneath it."""

    ABSTRACT = "abstract"
    REFERENCES = "references"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    BODY = "body"


def heading_title(heading: str) -> str:
    """Strip numbering and punctuation from a heading for comparison."""
    stripped = normalize_inline(heading)
    stripped = _NUMBER_PREFIX_RE.sub("", stripped)
    return stripped.strip().rstrip(":.").strip()


def heading_kind(heading: str) -> SectionKind | None:
    """Classify one heading, or ``None`` when it is ordinary body structure."""
    title = heading_title(heading).casefold()
    if not title:
        return None
    if title in _REFERENCE_HEADINGS:
        return SectionKind.REFERENCES
    if title in _ABSTRACT_HEADINGS:
        return SectionKind.ABSTRACT
    if title in _CONCLUSION_HEADINGS:
        return SectionKind.CONCLUSION
    if _APPENDIX_HEADING_RE.match(title):
        return SectionKind.APPENDIX
    return None


def section_kind(heading_path: Sequence[str]) -> SectionKind:
    """Classify a heading trail (innermost heading first)."""
    for heading in reversed(list(heading_path)):
        kind = heading_kind(heading)
        if kind is not None:
            return kind
    return SectionKind.BODY


def reference_intro_line(text: str) -> str | None:
    """Return the reference-list heading when a block opens with one.

    PDFs sometimes leave "References" glued to the first entry in one paragraph
    block; that line is still a reliable structural signal.
    """
    first_line = text.split("\n", 1)[0].strip()
    if _REFERENCE_INTRO_RE.match(first_line):
        return first_line.rstrip(":").strip()
    return None


def looks_like_caption(text: str) -> bool:
    """Whether a block looks like a reliably identified figure/table caption."""
    stripped = normalize_inline(text)
    if not stripped or len(stripped) > _MAX_CAPTION_CHARS:
        return False
    return bool(_CAPTION_RE.match(stripped))


def caption_label(text: str) -> str | None:
    """Return the caption label (``"Figure 3"``) when the block is a caption."""
    if not looks_like_caption(text):
        return None
    match = re.match(
        r"^(fig(?:ure)?|tab(?:le)?|scheme|chart|exhibit|box)\s*\.?\s*"
        r"(\d+[a-z]?|[ivxlc]+)",
        normalize_inline(text),
        re.IGNORECASE,
    )
    if match is None:
        return None
    word = match.group(1)
    word = {"fig": "Figure", "figure": "Figure", "tab": "Table", "table": "Table"}.get(
        word.casefold(), word[0].upper() + word[1:]
    )
    return f"{word} {match.group(2)}"
