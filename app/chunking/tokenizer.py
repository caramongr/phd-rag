"""Deterministic, offline token counting for chunk sizing (RAG-007).

Chunk sizes must not depend on the eventual generation or embedding model
(RAG-008 plans BGE-M3), and counting must work with no network access, so the
default counter is a deterministic *subword estimator* rather than a downloaded
BPE vocabulary:

* text is split into units — letter words, numbers, punctuation runs, whitespace;
* a word unit becomes ``ceil(len / 4)`` pieces and a punctuation run
  ``ceil(len / 2)`` pieces, mirroring the way BPE vocabularies break long words
  into subword pieces;
* whitespace is attached to the following piece, exactly as BPE does, so it adds
  no token of its own.

The estimator is deliberately slightly conservative (it over-counts typical
English prose relative to a GPT-2-class BPE), which is the safe direction for a
hard maximum. ``split_tokens`` is the primitive: it returns the pieces in order
and ``"".join(split_tokens(text)) == text``, so a hard split can cut at an exact
token budget without guessing.

The :class:`Tokenizer` protocol keeps the counter pluggable: a later stage may
inject a real vocabulary for *measurement*, but chunk boundaries in RAG-007 are
defined by this deterministic counter.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

#: Letter words (internal apostrophes kept), numbers, underscore runs, punctuation
#: symbols, whitespace. ``\w`` also contains ``_``, which no word/number rule above
#: matches, so underscore runs get their own alternative.
_UNIT_RE = re.compile(
    r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*"  # words such as "author's"
    r"|\d+(?:[.,]\d+)*"  # numbers such as "1,250.5"
    r"|_+"  # underscore runs used as rules/separators
    r"|[^\s\w]+"  # punctuation / symbol runs
    r"|\s+"  # whitespace
)

#: Characters per subword piece, used to approximate a BPE split.
_WORD_CHARS_PER_PIECE = 4
_SYMBOL_CHARS_PER_PIECE = 2


class Tokenizer(Protocol):
    """A deterministic text-to-token counter."""

    name: str

    def split(self, text: str) -> list[str]:
        """Return the token pieces of ``text`` in order."""
        ...

    def count(self, text: str) -> int:
        """Return how many tokens ``text`` contains."""
        ...

    def truncate(self, text: str, *, max_tokens: int) -> str:
        """Return the longest prefix of ``text`` within ``max_tokens`` tokens."""
        ...


def _pieces_for(unit: str) -> int:
    """Return how many subword pieces ``unit`` contributes."""
    if unit[0].isalpha() or unit[0].isdigit():
        return max(1, math.ceil(len(unit) / _WORD_CHARS_PER_PIECE))
    return max(1, math.ceil(len(unit) / _SYMBOL_CHARS_PER_PIECE))


class SubwordEstimator:
    """The default offline tokenizer: a deterministic subword estimator."""

    name = "subword-estimator-v1"

    def split(self, text: str) -> list[str]:
        """Split ``text`` into token pieces, preserving every character."""
        if not text:
            return []
        pieces: list[str] = []
        pending_whitespace = ""
        for match in _UNIT_RE.finditer(text):
            unit = match.group()
            if unit.isspace():
                pending_whitespace += unit
                continue
            pieces_for_unit = _pieces_for(unit)
            size = math.ceil(len(unit) / pieces_for_unit)
            for offset in range(0, len(unit), size):
                piece = unit[offset : offset + size]
                if offset == 0:
                    piece = pending_whitespace + piece
                    pending_whitespace = ""
                pieces.append(piece)
        if pending_whitespace:
            if pieces:
                pieces[-1] += pending_whitespace
            else:
                pieces.append(pending_whitespace)
        return pieces

    def count(self, text: str) -> int:
        """Return the number of token pieces in ``text``."""
        return len(self.split(text))

    def truncate(self, text: str, *, max_tokens: int) -> str:
        """Return the longest prefix of ``text`` within ``max_tokens`` tokens."""
        if max_tokens <= 0:
            return ""
        pieces = self.split(text)
        if len(pieces) <= max_tokens:
            return text
        return "".join(pieces[:max_tokens])


DEFAULT_TOKENIZER: Tokenizer = SubwordEstimator()


def count_tokens(text: str, *, tokenizer: Tokenizer | None = None) -> int:
    """Count the tokens in ``text`` with the default or supplied tokenizer."""
    return (tokenizer or DEFAULT_TOKENIZER).count(text)


def split_tokens(text: str, *, tokenizer: Tokenizer | None = None) -> list[str]:
    """Split ``text`` into token pieces with the default or supplied tokenizer."""
    return (tokenizer or DEFAULT_TOKENIZER).split(text)
