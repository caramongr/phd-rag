"""Deterministic text hygiene for *external* metadata values only (RAG-006).

Provider records occasionally carry JATS/XML presentation markup, HTML entities or
double-encoded UTF-8. Because accepted provider values are stored as metadata
candidates, those artefacts would otherwise end up in the merged title or venue.

The rules here are deliberately narrow: they remove markup while keeping its text,
decode entities, and repair mojibake *only* when the repair is unambiguous. No
wording, case or punctuation is rewritten, and a value that matches none of the
rules is returned unchanged. The provider's original text is preserved by the
clients (``ExternalRecord.raw_title`` / ``normalised``) so provenance stays honest.
"""

from __future__ import annotations

import html
import re

from app.metadata.text import collapse_whitespace

_TAG_NAME = r"[A-Za-z][A-Za-z0-9:._-]*"
#: Optional namespace prefix, as in ``<jats:p>``.
_NAMESPACE = r"(?:[A-Za-z][A-Za-z0-9._-]*:)?"

#: Tags that separate words/blocks: removing them must not join two words.
_BLOCK_TAGS = (
    "p",
    "br",
    "div",
    "sec",
    "title",
    "abstract",
    "caption",
    "label",
    "list",
    "list-item",
    "item",
    "table",
    "tbody",
    "thead",
    "tr",
    "td",
    "th",
    "fig",
    "body",
    "front",
    "back",
    "ack",
    "disp-quote",
)

_BLOCK_TAG_RE = re.compile(
    rf"</?{_NAMESPACE}(?:{'|'.join(_BLOCK_TAGS)})(?:\s[^<>]{{0,200}})?/?>",
    re.IGNORECASE,
)
#: Any remaining tag with a real tag name (inline presentation such as ``scp``/``italic``).
_INLINE_TAG_RE = re.compile(rf"</?{_TAG_NAME}(?:\s[^<>]{{0,200}})?/?>")

#: A named or numeric HTML entity, e.g. ``&amp;`` or ``&#8217;``.
_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")

#: Sequences that only occur when UTF-8 bytes were decoded as latin-1/cp1252.
_MOJIBAKE_SEQUENCES = (
    "â€",
    "â€™",
    "â€œ",
    "â€",
    "â€“",
    "â€”",
    "â€¦",
    "Ã",
    "Â",
    "ðŸ",
)


def _build_single_byte_table() -> dict[str, int]:
    """Map a character back to the single byte it was decoded from.

    Latin-1 characters map to their own code point; the printable cp1252
    characters (``€``, ``™``, ``’``, ``–`` …) map to their cp1252 byte. Characters
    that have no single-byte form are left out, so text containing them is not
    touched at all.
    """
    table = {chr(code): code for code in range(0x100)}
    for byte in range(0x80, 0x100):
        try:
            char = bytes([byte]).decode("cp1252")
        except UnicodeDecodeError:
            continue
        if ord(char) > 0xFF:
            table[char] = byte
    return table


_SINGLE_BYTE_TABLE = _build_single_byte_table()


def _decode_utf8_read_as_single_bytes(text: str) -> str | None:
    """Recover UTF-8 text that was read as latin-1/cp1252, or ``None``."""
    try:
        raw = bytes(_SINGLE_BYTE_TABLE[char] for char in text)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _mojibake_score(text: str) -> int:
    return sum(text.count(sequence) for sequence in _MOJIBAKE_SEQUENCES)


def _repair_mojibake(text: str) -> str:
    """Undo latin-1/cp1252-decoded UTF-8, but only when the result is clearly better."""
    if _mojibake_score(text) == 0:
        return text
    candidate = text
    for _ in range(2):  # at most two rounds: some values are encoded twice
        repaired = _decode_utf8_read_as_single_bytes(candidate)
        if (
            repaired is None
            or "\ufffd" in repaired
            or _mojibake_score(repaired) >= _mojibake_score(candidate)
        ):
            return candidate
        candidate = repaired
    return candidate


def clean_external_text(value: object) -> str | None:
    """Normalise one external metadata value, or return it unchanged."""
    if value is None:
        return None
    text = str(value)
    if not text.strip():
        return None
    text = html.unescape(text)
    text = _BLOCK_TAG_RE.sub(" ", text)
    text = _INLINE_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _repair_mojibake(text)
    return collapse_whitespace(text) or None


def has_artefacts(value: object) -> bool:
    """Whether a provider value carried markup, entities or mojibake."""
    if value is None:
        return False
    text = str(value)
    if (
        _BLOCK_TAG_RE.search(text)
        or _INLINE_TAG_RE.search(text)
        or _ENTITY_RE.search(text)
    ):
        return True
    return _repair_mojibake(text) != text
