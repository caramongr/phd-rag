"""Claim, numeric and quotation extraction for citation validation (RAG-012 §2, §6, §7).

The answer produced by RAG-011 is a plain string with internal ``[S1]`` markers. This
module turns it into the units RAG-012 checks, without touching the text:

* :func:`sentence_spans` — the sentence boundaries, as spans into the original string, so
  ``validated_answer`` can be rebuilt by exact substitution and the paragraph layout is
  preserved;
* :func:`classify_sentence` — claim, non-evidence report, unmarked claim or non-claim;
* :func:`extract_quotations` — the direct quotations, with elision handling;
* :func:`extract_numeric_mentions` — percentages, p-values, ranges, years, counts and bare
  numbers, in a fixed priority order so one number is reported once.

Sentence splitting and marker parsing are RAG-011's: :func:`~app.generation.grounding.
split_sentences` (abbreviation-protected) and :func:`~app.generation.grounding.
extract_markers`. Reusing them means the two stages can never disagree about where a
sentence ends or which source a marker names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.citations.models import ClaimKind, NumericKind, NumericMention
from app.generation.grounding import (
    MARKER_RE,
    extract_markers,
    sentence_has_claim_signal,
    sentence_is_trivial,
    split_sentences,
)

#: Function words and other terms that carry no claim content.
STOPWORDS = frozenset(
    """
    a about above after again against all also am among an and another any are as at be
    because been before being below between both but by can cannot could did do does
    doing done down during each either else even ever every few for from further had has
    have having he her here hers herself him himself his how however i if in into is it
    its itself just like me might more must my myself no nor not now of off on once one
    only or other ought our ours ourselves out over own said same she should since so
    some such than that the their theirs them themselves then there these they this those
    though through thus to too under until up upon us very was we were what when where
    whether which while who whom why will with within without would you your yours
    yourself yourselves et al eg ie etc per via being there
    """.split()
)

#: Suffixes removed by the deliberately small, deterministic stemmer.
_SUFFIXES = ("ingly", "edly", "ing", "ed", "ly", "es", "s")

#: Words as they appear in the text, keeping case so proper nouns can be recognised.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

#: Double quotations (straight or curly), the same shape RAG-011 recognises.
QUOTE_RE = re.compile(r"[\u201c\u201d\"]([^\u201c\u201d\"]{12,})[\u201c\u201d\"]")
#: Ellipsis markers used inside a quotation.
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|\u2026)\s*")

_NUMBER = r"\d+(?:[.,]\d+)?"
_PVALUE_RE = re.compile(rf"\bp\s*(?:<|>|=|\u2264|\u2265)\s*0?\.\d+", re.IGNORECASE)
_PERCENT_RE = re.compile(rf"\b{_NUMBER}\s*(?:%|per\s*cent|percent)", re.IGNORECASE)
_RANGE_RE = re.compile(rf"\b{_NUMBER}\s*(?:-|\u2013|\u2014|to)\s*{_NUMBER}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
#: Nouns that make a number a sample size, a count or a study size rather than a value.
COUNT_NOUNS = (
    "participants",
    "students",
    "undergraduates",
    "postgraduates",
    "respondents",
    "teachers",
    "educators",
    "learners",
    "faculty",
    "papers",
    "articles",
    "studies",
    "reviews",
    "cases",
    "institutions",
    "universities",
    "schools",
    "courses",
    "endorsements",
    "images",
    "scenes",
    "items",
    "chunks",
    "authors",
    "journals",
    "records",
    "sources",
    "citations",
    "references",
    "interviews",
    "questionnaires",
    "experiments",
    "trials",
    "metaanalyses",
    "chapters",
    "documents",
    "respondent",
    "student",
    "study",
)
_COUNT_RE = re.compile(
    rf"\b({_NUMBER})\s+(?:[a-z][a-z-]*\s+){{0,3}}(?:{'|'.join(COUNT_NOUNS)})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(rf"\b{_NUMBER}\b")

#: Extraction order: the first pattern to claim a span wins, so ``p < 0.05`` is one
#: p-value mention rather than a p-value plus a bare ``0.05``.
_PATTERNS: tuple[tuple[NumericKind, re.Pattern[str]], ...] = (
    (NumericKind.PVALUE, _PVALUE_RE),
    (NumericKind.PERCENT, _PERCENT_RE),
    (NumericKind.RANGE, _RANGE_RE),
    (NumericKind.COUNT, _COUNT_RE),
    (NumericKind.YEAR, _YEAR_RE),
    (NumericKind.NUMBER, _NUMBER_RE),
)


def stem(token: str) -> str:
    """A tiny deterministic stemmer, so ``findings``/``finding`` match."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def term(token: str) -> str:
    """The canonical form of a content word, applied identically to both sides.

    The stem, with a trailing ``e`` removed, so ``scene``/``scenes``,
    ``image``/``images``, ``increase``/``increases`` and ``produce``/``produced`` all
    compare equal.
    """
    stemmed = stem(token)
    if len(stemmed) > 4 and stemmed.endswith("e"):
        return stemmed[:-1]
    return stemmed


#: A citation in the running text: optional author-name run plus a parenthesised year
#: (``Mikropoulos and Natsis (2011)``, ``(Merchant et al., 2014)``).
_CITATION_RE = re.compile(
    r"(?:\b[A-Z][A-Za-z\u2019'\-]+(?:\s+(?:and|&)\s+|\s+et\s+al\.?\s+|\s+)){0,4}"
    r"\(\s*[^()]{0,40}?(?:1[5-9]\d{2}|20\d{2})[a-z]?[^()]{0,40}?\)"
)


def content_terms(text: str) -> frozenset[str]:
    """The content words of ``text``, canonicalised and stripped of two kinds of token.

    * **Citations in the running text** (``Merchant et al. (2014)``) name the work rather
      than describe its content; a claim is not penalised for naming the author it cites.
    * **Numerals** are checked by the numeric layer against the evidence, so counting them
      here as well would charge a claim twice for the same figure.

    Mid-sentence capitalised words are also skipped: they are proper nouns and author
    names, which the citation rule above may not catch on its own. The same rules apply to
    the claim and to the source text, so the comparison stays symmetric.
    """
    return frozenset(canonical for canonical, _ in content_tokens(text))


def content_tokens(text: str) -> tuple[tuple[str, str], ...]:
    """``(canonical term, surface token)`` for every content word of ``text``.

    Comparisons use the canonical form; a report shows the surface token, so a reader sees
    ``analysis`` rather than the stem ``analysi``.
    """
    subject = _CITATION_RE.sub(" ", text or "")
    tokens: list[tuple[str, str]] = []
    for match in _WORD_RE.finditer(subject):
        token = match.group(0)
        lowered = token.casefold().replace("-", "")
        if len(lowered) < 3 or lowered in STOPWORDS:
            continue
        if any(character.isdigit() for character in lowered):
            continue
        if token[:1].isupper() and not _is_sentence_initial(subject, match.start()):
            continue
        tokens.append((term(lowered), lowered))
    return tuple(tokens)


def _is_sentence_initial(text: str, index: int) -> bool:
    """Whether the token at ``index`` is the first word of its sentence or heading."""
    position = index - 1
    while position >= 0 and text[position] in " \t":
        position -= 1
    if position < 0:
        return True
    return text[position] in ".!?:;\n\u2022*-([/"


def coverage(claim_terms: Iterable[str], text: str) -> float:
    """Share of ``claim_terms`` that occurs in ``text`` (0.0 when there are none)."""
    terms = frozenset(claim_terms)
    if not terms:
        return 0.0
    present = terms & content_terms(text)
    return len(present) / len(terms)


def source_sentences(text: str) -> list[str]:
    """The sentences of a *source passage*, with hard-wrapped lines re-joined.

    PDF extraction inserts a line break wherever the page was narrow, so ``split_sentences``
    alone would turn one source sentence into several fragments and make a claim look as if
    it were assembled from separate passages. A line is joined to the previous one unless it
    ends with terminal punctuation, which is what a genuine line break looks like.
    """
    merged: list[str] = []
    buffer = ""
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue
        buffer = f"{buffer} {line}".strip() if buffer else line
        if line.endswith((".", "!", "?", ".\u201d", '."', ".)", ":", ";")):
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    sentences: list[str] = []
    for chunk in merged:
        split = split_sentences(chunk) or [chunk]
        sentences.extend(sentence for sentence in split if sentence.strip())
    return sentences


def best_sentence_coverage(claim_terms: Iterable[str], text: str) -> float:
    """The highest :func:`coverage` over the single sentences of ``text``."""
    terms = frozenset(claim_terms)
    if not terms:
        return 0.0
    sentences = source_sentences(text) or [text]
    return max((coverage(terms, sentence) for sentence in sentences), default=0.0)


@dataclass(frozen=True)
class SentenceSpan:
    """One sentence of the answer, with its exact span in the original string."""

    text: str
    start: int
    end: int


def sentence_spans(answer: str) -> tuple[SentenceSpan, ...]:
    """Locate every sentence of ``answer`` as a span, in order, exactly once."""
    spans: list[SentenceSpan] = []
    cursor = 0
    for sentence in split_sentences(answer or ""):
        start = answer.find(sentence, cursor)
        if start < 0:  # pragma: no cover - defensive: split_sentences preserves text
            start = answer.find(sentence)
        if start < 0:  # pragma: no cover - defensive
            continue
        end = start + len(sentence)
        spans.append(SentenceSpan(text=sentence, start=start, end=end))
        cursor = end
    return tuple(spans)


def classify_sentence(
    sentence: str,
    *,
    supplied_ids: Iterable[str],
    citable_ids: Iterable[str],
    non_evidence_reports: Iterable[str] = (),
) -> ClaimKind:
    """What a sentence is, using RAG-011's own claim/triviality heuristics.

    A sentence whose markers all point at non-citable blocks is a *claim* unless RAG-011
    already classified it as a report about that material: a citation of the author's
    draft or of a metadata-only record must be judged by the validator, not waved through.
    """
    citable = {source_id.upper() for source_id in citable_ids}
    markers = extract_markers(sentence)
    if markers:
        if sentence_is_trivial(sentence) and not sentence_has_claim_signal(sentence):
            # A heading, bullet or question carrying a marker is not a claim: the marker
            # is a formatting artifact and is removed rather than cited.
            return ClaimKind.NON_CLAIM
        if any(marker in citable for marker in markers):
            return ClaimKind.CLAIM
        if sentence in set(non_evidence_reports):
            return ClaimKind.NON_EVIDENCE_REPORT
        return ClaimKind.CLAIM
    if sentence_is_trivial(sentence):
        return ClaimKind.NON_CLAIM
    if sentence_has_claim_signal(sentence):
        return ClaimKind.UNMARKED_CLAIM
    return ClaimKind.NON_CLAIM


def marker_groups(sentence: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Each marker occurrence in the sentence, with its inner text and source ids."""
    groups: list[tuple[str, tuple[str, ...]]] = []
    for match in MARKER_RE.finditer(sentence or ""):
        ids = tuple(part.strip().upper() for part in match.group(1).split(",") if part.strip())
        groups.append((match.group(0), ids))
    return tuple(groups)


def strip_markers(text: str) -> str:
    """``text`` with every ``[Sx]`` marker removed (used before numeric scanning)."""
    return MARKER_RE.sub(" ", text or "")


# --- numerals -----------------------------------------------------------------


def parse_number(raw: str) -> float | None:
    """Parse a numeral, tolerating thousands separators and decimal commas."""
    text = (raw or "").strip().replace(" ", "").replace("%", "")
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        head, _, tail = text.rpartition(",")
        if len(tail) == 3 and head and head.lstrip("-").isdigit():
            text = head + tail
        else:
            text = head + "." + tail
    try:
        return float(text)
    except ValueError:  # pragma: no cover - defensive
        return None


def extract_numeric_mentions(text: str, *, strip_source_markers: bool = True) -> tuple[NumericMention, ...]:
    """Every numeric value in ``text``, one mention per value, in reading order.

    Markers are removed first: the ``1`` in ``[S1]`` is a source id, not a claim about a
    quantity.
    """
    subject = strip_markers(text) if strip_source_markers else (text or "")
    consumed = [False] * len(subject)
    mentions: list[tuple[int, NumericMention]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(subject):
            start, end = match.span()
            if any(consumed[start:end]):
                continue
            for position in range(start, end):
                consumed[position] = True
            mentions.append((start, _mention(kind, subject[start:end])))
    mentions.sort(key=lambda item: item[0])
    return tuple(mention for _start, mention in mentions)


def _mention(kind: NumericKind, raw: str) -> NumericMention:
    """Build a :class:`NumericMention` for a matched span."""
    stripped = raw.strip()
    if kind is NumericKind.PVALUE:
        number = _NUMBER_RE.search(stripped)
        return NumericMention(
            kind=kind,
            raw=stripped,
            value=parse_number(number.group(0)) if number else None,
            unit="p",
        )
    if kind is NumericKind.PERCENT:
        value = parse_number(re.sub(r"(?i)(per\s*cent|percent|%)", "", stripped))
        return NumericMention(kind=kind, raw=stripped, value=value, unit="%")
    if kind is NumericKind.RANGE:
        parts = re.split(r"\s*(?:-|\u2013|\u2014|to)\s*", stripped, maxsplit=1)
        first = parse_number(parts[0]) if parts else None
        second = parse_number(parts[1]) if len(parts) > 1 else None
        return NumericMention(
            kind=kind, raw=stripped, value=first, second_value=second, unit="range"
        )
    if kind is NumericKind.YEAR:
        value = parse_number(stripped)
        return NumericMention(
            kind=kind, raw=stripped, value=value, unit="year"
        )
    if kind is NumericKind.COUNT:
        value = parse_number(_NUMBER_RE.search(stripped).group(0)) if _NUMBER_RE.search(stripped) else None
        return NumericMention(kind=kind, raw=stripped, value=value, unit="count")
    value = parse_number(stripped)
    return NumericMention(kind=kind, raw=stripped, value=value, unit="number")


@dataclass(frozen=True)
class NumericIndex:
    """The numeric content of one source, prepared for fast comparison."""

    text: str
    numbers: frozenset[float] = frozenset()
    percents: frozenset[float] = frozenset()
    pvalues: frozenset[float] = frozenset()
    ranges: frozenset[tuple[float, float]] = frozenset()
    years: frozenset[int] = frozenset()
    counts: frozenset[float] = frozenset()

    def all_values(self) -> frozenset[float]:
        """Every numeric value mentioned, whatever its kind."""
        return self.numbers | self.percents | self.pvalues | self.counts


def build_numeric_index(text: str) -> NumericIndex:
    """Index every numeric value in ``text``."""
    numbers: set[float] = set()
    percents: set[float] = set()
    pvalues: set[float] = set()
    ranges: set[tuple[float, float]] = set()
    years: set[int] = set()
    counts: set[float] = set()
    for mention in extract_numeric_mentions(text, strip_source_markers=False):
        value = mention.value
        if mention.kind is NumericKind.PVALUE and value is not None:
            pvalues.add(value)
        elif mention.kind is NumericKind.PERCENT and value is not None:
            percents.add(value)
        elif mention.kind is NumericKind.RANGE:
            if value is not None and mention.second_value is not None:
                ranges.add((value, mention.second_value))
        elif mention.kind is NumericKind.YEAR and value is not None:
            years.add(int(value))
        elif mention.kind is NumericKind.COUNT and value is not None:
            counts.add(value)
        elif value is not None:
            numbers.add(value)
    return NumericIndex(
        text=text or "",
        numbers=frozenset(numbers),
        percents=frozenset(percents),
        pvalues=frozenset(pvalues),
        ranges=frozenset(ranges),
        years=frozenset(years),
        counts=frozenset(counts),
    )


def _close(left: float | None, right: float | None, tolerance: float) -> bool:
    return left is not None and right is not None and abs(left - right) <= tolerance


def match_numeric(
    mention: NumericMention,
    index: NumericIndex,
    *,
    tolerance: float = 1e-6,
    extra_years: Sequence[int] = (),
) -> tuple[bool, str]:
    """Whether ``index`` contains ``mention``, and how it was derived.

    Returns ``(found, derivation)``. An empty derivation means the value is present as
    the same kind of quantity; a named derivation means it is directly derivable (a
    percentage written as its decimal equivalent, or a year taken from the cited work's
    metadata rather than its text).
    """
    kind, value = mention.kind, mention.value
    if value is None:
        return False, ""
    if kind is NumericKind.PVALUE:
        return (_any_close(value, index.pvalues, tolerance), "")
    if kind is NumericKind.PERCENT:
        if _any_close(value, index.percents, tolerance):
            return True, ""
        if _any_close(value / 100.0, index.numbers, tolerance):
            return True, "percent_written_as_decimal"
        return False, ""
    if kind is NumericKind.RANGE:
        for first, second in index.ranges:
            if _close(first, value, tolerance) and _close(
                second, mention.second_value, tolerance
            ):
                return True, ""
        return False, ""
    if kind is NumericKind.YEAR:
        if int(value) in index.years:
            return True, ""
        if int(value) in {int(year) for year in extra_years}:
            return True, "citation_metadata_year"
        return False, ""
    if kind is NumericKind.COUNT:
        if _any_close(value, index.counts, tolerance):
            return True, ""
        return (_any_close(value, index.numbers, tolerance), "")
    if _any_close(value, index.numbers, tolerance) or _any_close(
        value, index.counts, tolerance
    ):
        return True, ""
    if _any_close(value * 100.0, index.percents, tolerance):
        return True, "decimal_written_as_percent"
    return False, ""


def _any_close(value: float, values: Iterable[float], tolerance: float) -> bool:
    return any(_close(value, other, tolerance) for other in values)


# --- quotations ---------------------------------------------------------------


@dataclass(frozen=True)
class Quotation:
    """A quotation found in the answer, with its elision fragments."""

    text: str
    fragments: tuple[str, ...]
    elided: bool


def extract_quotations(text: str) -> tuple[Quotation, ...]:
    """Every double-quoted span long enough to be a quotation, in reading order."""
    quotations: list[Quotation] = []
    for match in QUOTE_RE.finditer(text or ""):
        quoted = match.group(1).strip()
        if not quoted:
            continue
        fragments = tuple(
            fragment.strip() for fragment in _ELLIPSIS_RE.split(quoted) if fragment.strip()
        )
        quotations.append(
            Quotation(
                text=quoted,
                fragments=fragments or (quoted,),
                elided=len(fragments) > 1,
            )
        )
    return tuple(quotations)
