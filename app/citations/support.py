"""Claim support validation (RAG-012 §2-§8).

The job is narrow and deliberate: take one answer sentence, the sources it cites, and
decide whether those sources actually contain the claim.

```
sentence + cited source chunks
        |
        +-- deterministic checks (never skipped)     §3
        |     marker exists | source is evidence-eligible | text present
        |     bibliography-like chunk rejected | quotation verbatim §7
        |     every numeral present or derivable §6
        |
        +-- support layer (only if the deterministic checks pass)   §4, §5
              one source sentence carries the claim?  -> SUPPORTED
              content spread over passages/sources?   -> PARTIALLY_SUPPORTED
              almost no overlap with the evidence?    -> UNSUPPORTED
              weak overlap, no decision possible?     -> UNCERTAIN
```

Four rules shape the code:

* **Deterministic first.** Invalid markers, non-evidence sources, missing source text,
  bibliography-like chunks, broken quotations and absent numbers are decided without any
  model. A language model is never allowed to override them.
* **Numbers are strict** (§6): semantic similarity cannot make a missing ``p < 0.05``
  acceptable, so a numeric claim fails when a value is not in the cited evidence.
* **Partial is a first-class outcome** (§5, §15): the RAG-011 failure mode is two findings
  compressed into one sentence, which shows up as a claim whose content only holds across
  separate source sentences. Recording that as ``PARTIALLY_SUPPORTED`` is the point of
  the stage.
* **Uncertainty is never upgraded silently** (§4). A weak lexical result stays
  ``UNCERTAIN``; only an explicit, recorded embedding-based decision may lift it, and it
  may never lift partial support to full support.

The optional semantic layer is an *addition*, not a dependency: the default validator is
purely deterministic, and the embedding checker loads BGE-M3 lazily and only when asked
for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.citations import harvard
from app.citations.claims import (
    Quotation,
    SentenceSpan,
    best_sentence_coverage,
    build_numeric_index,
    classify_sentence,
    content_terms,
    coverage,
    extract_numeric_mentions,
    extract_quotations,
    marker_groups,
    match_numeric,
    sentence_spans,
    source_sentences,
    strip_markers,
)
from app.citations.models import (
    ClaimDecision,
    ClaimKind,
    ClaimValidation,
    NumericMention,
    QuotationCheck,
    SourceSupport,
    SupportVerdict,
    ValidationPolicy,
    ValidatorMethod,
    worst_verdict,
)
from app.generation.models import ContextPackage, SourceBlock
from app.retrieval.bibliography import is_bibliography_like

_WHITESPACE_RE = re.compile(r"\s+")
_TYPOGRAPHIC = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


# --- support decision ---------------------------------------------------------


@dataclass(frozen=True)
class EntailmentResult:
    """One support decision, with the method and evidence behind it."""

    verdict: SupportVerdict
    method: ValidatorMethod
    confidence: float = 0.0
    reason: str = ""
    #: Coverage or similarity behind the verdict, for the audit trail.
    score: float | None = None
    single_sentence_coverage: float | None = None
    union_coverage: float | None = None


class EntailmentChecker(Protocol):
    """A support checker: the claim text, and the passages that should support it."""

    name: str

    def check(
        self,
        claim: str,
        passages: Sequence[str],
        *,
        policy: ValidationPolicy | None = None,
    ) -> EntailmentResult:  # pragma: no cover - protocol definition
        ...


def lexical_result(
    claim: str,
    passages: Sequence[str],
    *,
    policy: ValidationPolicy | None = None,
) -> EntailmentResult:
    """The deterministic support decision for ``claim`` against ``passages``.

    The decision rests on two measurable things:

    * **how much of the claim the evidence contains** (``union_coverage``): the share of
      the claim's content terms that occur anywhere in the cited passages. This is the
      band that decides support, because a paraphrase legitimately spreads its wording over
      several sentences of a source.
    * **whether the claim joins separate quantified findings** (``merged``): a claim whose
      "a majority ... with many ..." style units rest on *different* source sentences is
      stating two findings as one, which is the failure mode RAG-011 recorded and this stage
      must catch.

    ``single_sentence_coverage`` (the best coverage by any one sentence) is reported as
    well, so an auditor can see whether a supported claim was stated in one place or
    assembled from several passages.
    """
    policy = policy or ValidationPolicy()
    terms = content_terms(claim)
    texts = [text for text in passages if (text or "").strip()]
    if not terms:
        return EntailmentResult(
            verdict=SupportVerdict.UNCERTAIN,
            method=ValidatorMethod.LEXICAL_WEAK_OVERLAP,
            confidence=0.0,
            reason="the claim has no content words to compare against the evidence",
        )
    union = coverage(terms, " ".join(texts))
    single = max((best_sentence_coverage(terms, text) for text in texts), default=0.0)
    if len(terms) < policy.min_content_terms:
        # Too short to judge by coverage: accept only when the source shares its content.
        if union >= policy.no_support_coverage:
            return EntailmentResult(
                verdict=SupportVerdict.SUPPORTED,
                method=ValidatorMethod.LEXICAL_SINGLE_SENTENCE,
                confidence=round(min(policy.max_lexical_confidence, 0.55 + 0.30 * union), 3),
                reason="short claim; its content words all occur in the cited evidence",
                score=union,
                single_sentence_coverage=single,
                union_coverage=union,
            )
        return EntailmentResult(
            verdict=SupportVerdict.UNCERTAIN,
            method=ValidatorMethod.LEXICAL_WEAK_OVERLAP,
            confidence=0.25,
            reason="short claim with little lexical overlap; too short to decide",
            score=union,
            single_sentence_coverage=single,
            union_coverage=union,
        )
    merged = merged_findings(claim, texts, policy=policy)
    if merged is not None and union >= policy.support_coverage:
        homes = ", ".join(
            f"{unit[:48]!r} matches passage {home + 1}"
            for unit, home in zip(merged.units, merged.homes)
        )
        return EntailmentResult(
            verdict=SupportVerdict.PARTIALLY_SUPPORTED,
            method=ValidatorMethod.LEXICAL_MERGED_FINDINGS,
            confidence=0.5,
            reason=(
                "two separately-sourced findings are joined in one sentence: "
                f"{homes}, and no single passage states the whole claim "
                f"({merged.single_sentence_coverage:.0%} at best)"
            ),
            score=union,
            single_sentence_coverage=single,
            union_coverage=union,
        )
    if union >= policy.support_coverage:
        detail = (
            "a single cited sentence carries it"
            if single >= policy.support_coverage
            else f"its content is spread over the cited passages ({single:.0%} best sentence)"
        )
        return EntailmentResult(
            verdict=SupportVerdict.SUPPORTED,
            method=ValidatorMethod.LEXICAL_SINGLE_SENTENCE
            if single >= policy.support_coverage
            else ValidatorMethod.LEXICAL_MULTI_SENTENCE,
            confidence=round(min(policy.max_lexical_confidence, 0.55 + 0.30 * union), 3),
            reason=f"the cited evidence contains {union:.0%} of the claim's content terms; {detail}",
            score=union,
            single_sentence_coverage=single,
            union_coverage=union,
        )
    if union >= policy.partial_coverage:
        return EntailmentResult(
            verdict=SupportVerdict.PARTIALLY_SUPPORTED,
            method=ValidatorMethod.LEXICAL_MULTI_SENTENCE,
            confidence=0.5,
            reason=(
                f"only {union:.0%} of the claim's content terms occur in the cited evidence"
            ),
            score=union,
            single_sentence_coverage=single,
            union_coverage=union,
        )
    if union >= policy.no_support_coverage:
        return EntailmentResult(
            verdict=SupportVerdict.UNCERTAIN,
            method=ValidatorMethod.LEXICAL_WEAK_OVERLAP,
            confidence=0.25,
            reason=f"only weak lexical overlap with the evidence ({union:.0%})",
            score=union,
            single_sentence_coverage=single,
            union_coverage=union,
        )
    return EntailmentResult(
        verdict=SupportVerdict.UNSUPPORTED,
        method=ValidatorMethod.LEXICAL_NO_OVERLAP,
        confidence=0.6,
        reason=f"the cited evidence shares almost no content with the claim ({union:.0%})",
        score=union,
        single_sentence_coverage=single,
        union_coverage=union,
    )


#: Phrases that quantify a finding, used by the merging test.
_QUANTIFIER_RE = re.compile(
    r"\b(?:a majority of|the majority of|majority|most|many|several|some|few|numerous|"
    r"half of|most of)\b|\b\d+(?:[.,]\d+)?\s*%",
    re.IGNORECASE,
)
#: Boundaries between the units of a sentence, for the merging test.
_UNIT_BOUNDARY_RE = re.compile(
    r",\s+with\s+|,\s+while\s+|,\s+whereas\s+|,\s+which\s+|;\s+|,\s+and\s+|\bwith\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MergedFindings:
    """A claim whose quantified units rest on different source sentences."""

    units: tuple[str, ...]
    homes: tuple[int, ...]
    coverage: tuple[float, ...]
    #: How much of the whole claim the single best source sentence carries.
    single_sentence_coverage: float

    @property
    def distinct_passages(self) -> int:
        """How many different source sentences the units rest on."""
        return len(set(self.homes))


def claim_units(claim: str) -> tuple[str, ...]:
    """The clause-like units of a claim, split at its connectives."""
    units = [unit.strip(" ,;.") for unit in _UNIT_BOUNDARY_RE.split(claim or "")]
    return tuple(unit for unit in units if unit)


def merged_findings(
    claim: str,
    passages: Sequence[str],
    *,
    policy: ValidationPolicy | None = None,
) -> MergedFindings | None:
    """Whether a claim joins quantified findings that rest on different sentences.

    This is the RAG-011 failure mode in testable form (§5, §15): the claim says "a majority
    of faculty consider X important, with many advocating Y", where X and Y come from
    different findings of the same work. Two quantified units are required, each grounded in
    the evidence, and their best-matching passages must differ. Returns ``None`` when the
    claim is not of that shape, or when a single sentence states the whole claim.
    """
    policy = policy or ValidationPolicy()
    units = [unit for unit in claim_units(claim) if _QUANTIFIER_RE.search(unit)]
    if len(units) < 2:
        return None
    sentences = [sentence for passage in passages for sentence in source_sentences(passage)]
    if not sentences:
        return None
    whole = content_terms(claim)
    single = max((coverage(whole, sentence) for sentence in sentences), default=0.0)
    if single >= policy.support_coverage:
        # One sentence states the whole claim, so nothing was merged.
        return None
    homes: list[int] = []
    coverages: list[float] = []
    for unit in units:
        terms = content_terms(unit)
        if not terms:
            return None
        best_index, best_score = 0, 0.0
        for index, sentence in enumerate(sentences):
            score = coverage(terms, sentence)
            if score > best_score:
                best_index, best_score = index, score
        if best_score < policy.unit_coverage_floor:
            # A unit that no passage grounds at all is a different problem, not this one.
            return None
        homes.append(best_index)
        coverages.append(round(best_score, 4))
    if len(set(homes)) < 2:
        return None
    return MergedFindings(
        units=tuple(units),
        homes=tuple(homes),
        coverage=tuple(coverages),
        single_sentence_coverage=round(single, 4),
    )


class LexicalEntailmentChecker:
    """The default checker: deterministic content overlap, no model involved."""

    name = "lexical"

    def __init__(self, policy: ValidationPolicy | None = None) -> None:
        self.policy = policy or ValidationPolicy()

    def check(
        self,
        claim: str,
        passages: Sequence[str],
        *,
        policy: ValidationPolicy | None = None,
    ) -> EntailmentResult:
        """Decide support from content-word coverage."""
        return lexical_result(claim, passages, policy=policy or self.policy)


class EmbeddingEntailmentChecker:
    """Optional semantic layer over the local BGE-M3 embedder (§4).

    Only used for claims the deterministic layer could not decide, and only when a caller
    asks for it: the embedder is imported and loaded lazily on first use, so nothing in the
    default path depends on it.
    """

    name = "embedding_bge_m3"

    def __init__(
        self,
        embedder: object | None = None,
        *,
        supported: float = 0.70,
        partial: float = 0.55,
        weak: float = 0.40,
        support_confidence: float = 0.75,
    ) -> None:
        self._embedder = embedder
        self.supported = supported
        self.partial = partial
        self.weak = weak
        self.support_confidence = support_confidence

    @property
    def embedder(self) -> object:
        """The BGE-M3 embedder, built on first use."""
        if self._embedder is None:
            from app.embedding.bge_m3 import BgeM3Embedder

            self._embedder = BgeM3Embedder()
        return self._embedder

    @staticmethod
    def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right)))

    def check(
        self,
        claim: str,
        passages: Sequence[str],
        *,
        policy: ValidationPolicy | None = None,
    ) -> EntailmentResult:
        """Decide support from cosine similarity between the claim and the evidence."""
        texts = [text for text in passages if (text or "").strip()]
        if not texts:
            return EntailmentResult(
                verdict=SupportVerdict.UNCERTAIN,
                method=ValidatorMethod.EMBEDDING_SEMANTIC,
                reason="no evidence text to compare against",
            )
        candidates = list(texts)
        for text in texts:
            candidates.extend(sentence for sentence in text.split(". ") if sentence.strip())
        vectors = self.embedder.embed_documents([claim] + candidates)
        claim_vector, candidate_vectors = vectors[0], vectors[1:]
        best = max(
            (self._similarity(claim_vector, vector) for vector in candidate_vectors),
            default=0.0,
        )
        if best >= self.supported:
            verdict, confidence = SupportVerdict.SUPPORTED, self.support_confidence
        elif best >= self.partial:
            verdict, confidence = SupportVerdict.PARTIALLY_SUPPORTED, 0.5
        elif best >= self.weak:
            verdict, confidence = SupportVerdict.UNCERTAIN, 0.25
        else:
            verdict, confidence = SupportVerdict.UNSUPPORTED, 0.6
        return EntailmentResult(
            verdict=verdict,
            method=ValidatorMethod.EMBEDDING_SEMANTIC,
            confidence=confidence,
            reason=f"BGE-M3 cosine similarity to the closest evidence passage is {best:.3f}",
            score=round(best, 4),
        )


# --- quotation matching -------------------------------------------------------


def _verbatim(text: str) -> str:
    """Whitespace-collapsed text with typographic characters normalised."""
    for source, target in _TYPOGRAPHIC.items():
        text = text.replace(source, target)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _loose(text: str) -> str:
    """Casefolded text with punctuation removed, for quotation-mark differences."""
    return _WHITESPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", _verbatim(text).casefold())).strip()


def quotation_level(text: str, quotation: Quotation) -> str | None:
    """How ``quotation`` occurs in ``text``: ``verbatim``, ``normalized`` or nothing.

    A quotation elided with ``...`` is accepted when every fragment is found, and the
    level is then reported as ``elided``. There is no similarity threshold here: the words
    either occur in that order or the quotation fails.
    """
    if _verbatim(quotation.text) in _verbatim(text):
        return "elided" if quotation.elided else "verbatim"
    if _loose(quotation.text) in _loose(text):
        return "elided" if quotation.elided else "normalized"
    if quotation.elided:
        levels = [
            "verbatim" if _verbatim(fragment) in _verbatim(text) else (
                "normalized" if _loose(fragment) in _loose(text) else None
            )
            for fragment in quotation.fragments
        ]
        if levels and all(level is not None for level in levels):
            return "elided"
    return None


def _find_quotation(
    quotation: Quotation, candidates: Sequence[tuple[str, str]]
) -> tuple[str, str] | None:
    """The first ``(source_id, level)`` whose text contains the quotation."""
    for source_id, text in candidates:
        level = quotation_level(text or "", quotation)
        if level is not None:
            return source_id, level
    return None


# --- claim validation ---------------------------------------------------------


def validate_claim(
    sentence: str,
    *,
    context: ContextPackage,
    index: int = 0,
    policy: ValidationPolicy | None = None,
    checker: EntailmentChecker | None = None,
    kind: ClaimKind = ClaimKind.CLAIM,
) -> ClaimValidation:
    """Validate one answer sentence against the sources it cites (§2-§8)."""
    policy = policy or ValidationPolicy()
    groups = marker_groups(sentence)
    source_ids = tuple(dict.fromkeys(sid for _group, ids in groups for sid in ids))
    blocks = [context.block_by_id(source_id) for source_id in source_ids]
    invalid_ids = tuple(
        source_id for source_id, block in zip(source_ids, blocks) if block is None
    )
    valid_blocks = [block for block in blocks if block is not None]
    non_evidence_ids = tuple(block.source_id for block in valid_blocks if not block.citable)
    evidence_blocks = [block for block in valid_blocks if block.citable]

    if kind is ClaimKind.NON_CLAIM:
        return _plain(
            sentence,
            index=index,
            kind=kind,
            source_ids=source_ids,
            reason="not a claim sentence",
            markers=bool(groups),
        )
    if kind is ClaimKind.UNMARKED_CLAIM:
        return _plain(
            sentence,
            index=index,
            kind=kind,
            source_ids=(),
            reason="a claim sentence with no source marker; RAG-011 reports these separately",
        )
    if kind is ClaimKind.NON_EVIDENCE_REPORT:
        return _plain(
            sentence,
            index=index,
            kind=kind,
            source_ids=source_ids,
            reason="reports on non-citable material (the author's own draft or the guidance)",
            method=ValidatorMethod.NON_EVIDENCE_SOURCE,
            decision=ClaimDecision.NON_CITABLE_REPORT,
        )

    risk_reasons: list[str] = []
    for block in evidence_blocks:
        if block.ocr_involved:
            risk_reasons.append(f"OCR text in [{block.source_id}]")
        if block.is_table:
            risk_reasons.append(f"table content in [{block.source_id}]")
    elevated = bool(risk_reasons)
    effective = policy.for_risk(elevated)
    if elevated:
        risk_reasons.append(
            "stricter thresholds applied (single-sentence coverage "
            f">= {effective.support_coverage:.0%})"
        )

    claim_text = strip_markers(sentence).strip()
    warnings: list[str] = []
    if invalid_ids:
        warnings.append(
            "invalid_source_marker: " + ", ".join(f"[{source_id}]" for source_id in invalid_ids)
        )
    if non_evidence_ids:
        warnings.append(
            "non_evidence_source: " + ", ".join(f"[{source_id}]" for source_id in non_evidence_ids)
        )

    textless = [block for block in evidence_blocks if not (block.text or "").strip()]
    bibliography_like = [
        block
        for block in evidence_blocks
        if (block.text or "").strip() and is_bibliography_like(block.text)
    ]
    support_blocks = [
        block
        for block in evidence_blocks
        if (block.text or "").strip() and block not in bibliography_like
    ]
    if textless:
        warnings.append(
            "no source text supplied for " + ", ".join(f"[{b.source_id}]" for b in textless)
        )
    if bibliography_like:
        warnings.append(
            "bibliography-like chunk cited: "
            + ", ".join(f"[{b.source_id}]" for b in bibliography_like)
        )

    quotations, quote_failures = _check_quotations(sentence, context, support_blocks)
    numeric_mentions = _check_numbers(sentence, support_blocks, policy=policy)
    missing_numbers = [mention for mention in numeric_mentions if mention.found is False]

    failure: tuple[ValidatorMethod, str] | None = None
    if not evidence_blocks:
        if invalid_ids:
            failure = (
                ValidatorMethod.INVALID_MARKER,
                "cites "
                + ", ".join(f"[{source_id}]" for source_id in invalid_ids)
                + ", which do not exist in the supplied context",
            )
        elif non_evidence_ids:
            failure = (
                ValidatorMethod.NON_EVIDENCE_SOURCE,
                "cites only non-citable material: "
                + ", ".join(f"[{source_id}]" for source_id in non_evidence_ids)
                + " (the author's own work, guidance or a metadata-only record)",
            )
        else:
            failure = (ValidatorMethod.NON_EVIDENCE_SOURCE, "cites no citable source")
    elif not support_blocks:
        if bibliography_like:
            failure = (
                ValidatorMethod.BIBLIOGRAPHY_LIKE_SOURCE,
                "the cited chunk reads as a reference list, not as evidence: "
                + ", ".join(f"[{b.source_id}]" for b in bibliography_like),
            )
        else:
            failure = (
                ValidatorMethod.MISSING_SOURCE_TEXT,
                "no source text was supplied for "
                + ", ".join(f"[{b.source_id}]" for b in textless),
            )
    elif quote_failures:
        failure = (
            ValidatorMethod.QUOTATION_MISMATCH,
            "the quotation does not occur in the cited source text: "
            + " | ".join(failures for failures in quote_failures),
        )
    elif missing_numbers:
        failure = (
            ValidatorMethod.NUMERIC_MISSING,
            "the value "
            + ", ".join(mention.display for mention in missing_numbers)
            + " does not appear in the cited evidence and is not derivable from it",
        )

    if failure is not None:
        method, reason = failure
        warnings.append(reason)
        per_source = _source_records(
            support_blocks or evidence_blocks,
            claim_text,
            policy=effective,
            numeric=numeric_mentions,
            quotations=quotations,
            failure_method=method,
            failure_reason=reason,
        )
        return ClaimValidation(
            index=index,
            sentence=sentence,
            kind=kind,
            source_ids=source_ids,
            invalid_source_ids=invalid_ids,
            non_evidence_source_ids=non_evidence_ids,
            verdict=SupportVerdict.UNSUPPORTED,
            validator_method=method,
            confidence=policy.deterministic_confidence,
            reason=reason,
            per_source=per_source,
            numeric_mentions=numeric_mentions,
            quotations=quotations,
            elevated_validation_risk=elevated,
            risk_reasons=tuple(risk_reasons),
            warnings=tuple(warnings),
            decision=ClaimDecision.WITHHELD,
            validated_sentence=sentence,
        )

    combined = lexical_result(claim_text, [block.text for block in support_blocks], policy=effective)
    per_source = _source_records(
        support_blocks,
        claim_text,
        policy=effective,
        numeric=numeric_mentions,
        quotations=quotations,
    )
    verdict = combined.verdict
    method = combined.method
    confidence = combined.confidence
    reason = combined.reason
    checker_note: str | None = None
    if checker is not None and verdict in (
        SupportVerdict.PARTIALLY_SUPPORTED,
        SupportVerdict.UNCERTAIN,
    ):
        escalated = checker.check(
            claim_text, [block.text for block in support_blocks], policy=effective
        )
        if (
            escalated.verdict is SupportVerdict.SUPPORTED
            and verdict is SupportVerdict.UNCERTAIN
            and escalated.confidence >= 0.75
        ):
            # An explicit, recorded decision is the only way UNCERTAIN may become SUPPORTED.
            verdict = SupportVerdict.SUPPORTED
            method, confidence, reason = (
                escalated.method,
                escalated.confidence,
                escalated.reason,
            )
            checker_note = (
                f"semantic layer ({checker.name}) resolved this claim: {escalated.reason}"
            )
        else:
            if escalated.verdict is not SupportVerdict.SUPPORTED:
                verdict = worst_verdict((verdict, escalated.verdict))
                method, confidence, reason = (
                    escalated.method,
                    escalated.confidence,
                    escalated.reason,
                )
            checker_note = (
                f"semantic layer ({checker.name}) consulted: the verdict stays "
                f"{verdict.value} ({escalated.reason})"
            )
    combination_required = (
        len(support_blocks) > 1
        and not any(record.verdict is SupportVerdict.SUPPORTED for record in per_source)
        and verdict in (SupportVerdict.SUPPORTED, SupportVerdict.PARTIALLY_SUPPORTED)
    )
    if combination_required:
        warnings.append(
            "no single cited source supports the claim on its own; the combination does "
            "(per-source support is recorded)"
        )
    if verdict is SupportVerdict.SUPPORTED and quotations:
        # A verified quotation is the strongest evidence available for a claim.
        method = ValidatorMethod.QUOTATION_EXACT
    if verdict is SupportVerdict.PARTIALLY_SUPPORTED:
        warnings.append("partially supported: the evidence does not state the whole claim")
    elif verdict is SupportVerdict.UNCERTAIN:
        warnings.append("uncertain: the deterministic checks could not confirm the claim")
    if missing_numbers and verdict is not SupportVerdict.UNSUPPORTED:
        warnings.append(
            "numeric values not located in this source: "
            + ", ".join(mention.display for mention in missing_numbers)
        )
    if checker_note is not None:
        warnings.append(checker_note)
    return ClaimValidation(
        index=index,
        sentence=sentence,
        kind=kind,
        source_ids=source_ids,
        invalid_source_ids=invalid_ids,
        non_evidence_source_ids=non_evidence_ids,
        verdict=verdict,
        validator_method=method,
        confidence=confidence,
        reason=reason,
        per_source=per_source,
        combination_required=combination_required,
        numeric_mentions=numeric_mentions,
        quotations=quotations,
        elevated_validation_risk=elevated,
        risk_reasons=tuple(risk_reasons),
        warnings=tuple(warnings),
        decision=(
            ClaimDecision.WITHHELD
            if verdict is SupportVerdict.UNSUPPORTED
            else ClaimDecision.ANNOTATED
            if verdict.needs_annotation
            else ClaimDecision.CITATION_REPLACED
        ),
        validated_sentence=sentence,
    )


def _plain(
    sentence: str,
    *,
    index: int,
    kind: ClaimKind,
    source_ids: tuple[str, ...],
    reason: str,
    method: ValidatorMethod = ValidatorMethod.LEXICAL_WEAK_OVERLAP,
    decision: ClaimDecision = ClaimDecision.UNCHANGED,
    markers: bool = False,
) -> ClaimValidation:
    """A sentence that is not a citable claim: nothing to validate, nothing to change."""
    if markers and decision is ClaimDecision.UNCHANGED:
        decision = ClaimDecision.MARKER_REMOVED
    return ClaimValidation(
        index=index,
        sentence=sentence,
        kind=kind,
        source_ids=source_ids,
        verdict=SupportVerdict.UNCERTAIN,
        validator_method=method,
        confidence=0.0,
        reason=reason,
        decision=decision,
        validated_sentence=sentence,
    )


def _check_quotations(
    sentence: str,
    context: ContextPackage,
    support_blocks: Sequence[SourceBlock],
) -> tuple[tuple[QuotationCheck, ...], tuple[str, ...]]:
    """Verify every direct quotation against the cited source text (§7)."""
    cited = [(block.source_id, block.text) for block in support_blocks]
    everything = [(block.source_id, block.text) for block in context.blocks]
    checks: list[QuotationCheck] = []
    failures: list[str] = []
    for quotation in extract_quotations(sentence):
        cited_hit = _find_quotation(quotation, cited)
        elsewhere = None if cited_hit is not None else _find_quotation(quotation, everything)
        if cited_hit is not None:
            source_id, level = cited_hit
            checks.append(
                QuotationCheck(
                    text=quotation.text,
                    source_ids=_ids_of(sentence),
                    found_in_source=True,
                    matched_source_id=source_id,
                    match_level=level,
                    elided=quotation.elided,
                    page_provenance=context.block_by_id(source_id).pages
                    if context.block_by_id(source_id)
                    else None,
                    note=(
                        f"quotation found {level} in [{source_id}]"
                        if level != "elided"
                        else f"elided quotation; every fragment found in [{source_id}]"
                    ),
                )
            )
        elif elsewhere is not None:
            source_id, level = elsewhere
            checks.append(
                QuotationCheck(
                    text=quotation.text,
                    source_ids=_ids_of(sentence),
                    found_in_source=False,
                    matched_source_id=source_id,
                    match_level=level,
                    elided=quotation.elided,
                    outside_cited_sources=True,
                    note=(
                        f"quotation occurs in [{source_id}], which this sentence does not "
                        f"cite"
                    ),
                )
            )
            failures.append(f"{quotation.text[:80]!r} (found only in [{source_id}])")
        else:
            checks.append(
                QuotationCheck(
                    text=quotation.text,
                    source_ids=_ids_of(sentence),
                    found_in_source=False,
                    match_level="not_found",
                    elided=quotation.elided,
                    note="quotation not found in any supplied source text",
                )
            )
            failures.append(f"{quotation.text[:80]!r} (not found)")
    return tuple(checks), tuple(failures)


def _ids_of(sentence: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(sid for _group, ids in marker_groups(sentence) for sid in ids)
    )


def _check_numbers(
    sentence: str,
    support_blocks: Sequence[SourceBlock],
    *,
    policy: ValidationPolicy,
) -> tuple[NumericMention, ...]:
    """Verify every numeric value in the claim against the cited evidence (§6)."""
    mentions = extract_numeric_mentions(sentence)
    if not mentions:
        return ()
    extra_years = [block.year for block in support_blocks if block.year]
    indices = {
        block.source_id: build_numeric_index(block.text) for block in support_blocks
    }
    checked: list[NumericMention] = []
    for mention in mentions:
        matched: list[str] = []
        derivation = ""
        for source_id, index in indices.items():
            found, how = match_numeric(
                mention,
                index,
                tolerance=policy.numeric_tolerance,
                extra_years=extra_years,
            )
            if found:
                matched.append(source_id)
                derivation = derivation or how
        checked.append(
            mention.model_copy(
                update={
                    "found": bool(matched),
                    "matched_source_ids": tuple(matched),
                    "derivation": derivation,
                }
            )
        )
    return tuple(checked)


def _source_records(
    blocks: Sequence[SourceBlock],
    claim_text: str,
    *,
    policy: ValidationPolicy,
    numeric: Sequence[NumericMention],
    quotations: Sequence[QuotationCheck],
    failure_method: ValidatorMethod | None = None,
    failure_reason: str = "",
) -> tuple[SourceSupport, ...]:
    """Per-source support records for one claim (§5)."""
    records: list[SourceSupport] = []
    for block in blocks:
        if failure_method is not None:
            records.append(
                SourceSupport(
                    source_id=block.source_id,
                    source_class=block.source_class.value,
                    verdict=SupportVerdict.UNSUPPORTED,
                    method=failure_method,
                    confidence=policy.deterministic_confidence,
                    reason=failure_reason,
                    metadata_warnings=harvard.source_metadata_warnings(block),
                )
            )
            continue
        result = lexical_result(claim_text, [block.text], policy=policy)
        matched = tuple(
            mention.display for mention in numeric if block.source_id in mention.matched_source_ids
        )
        missing = tuple(
            mention.display
            for mention in numeric
            if mention.found is not None and block.source_id not in mention.matched_source_ids
        )
        records.append(
            SourceSupport(
                source_id=block.source_id,
                source_class=block.source_class.value,
                verdict=result.verdict,
                method=result.method,
                confidence=result.confidence,
                reason=result.reason,
                best_sentence_coverage=round(result.single_sentence_coverage or 0.0, 4),
                union_coverage=round(result.union_coverage or 0.0, 4),
                matched_numeric=matched,
                missing_numeric=missing,
                quotations_found=sum(
                    1
                    for check in quotations
                    if check.found_in_source and check.matched_source_id == block.source_id
                ),
                quotations_missing=sum(
                    1
                    for check in quotations
                    if not check.found_in_source
                    and check.source_ids
                    and block.source_id in check.source_ids
                ),
                metadata_warnings=harvard.source_metadata_warnings(block),
            )
        )
    return tuple(records)


def validate_answer(
    answer: str,
    *,
    context: ContextPackage,
    policy: ValidationPolicy | None = None,
    checker: EntailmentChecker | None = None,
    spans: Sequence[SentenceSpan] | None = None,
    non_evidence_reports: Sequence[str] = (),
) -> tuple[ClaimValidation, ...]:
    """Validate every sentence of ``answer``, in order (§2).

    Uses the sentence spans from :func:`~app.citations.claims.sentence_spans` so the
    validated answer is rebuilt from exactly the sentences that were checked.
    """
    policy = policy or ValidationPolicy()
    supplied = context.source_ids
    citable = context.citable_source_ids
    validations: list[ClaimValidation] = []
    for index, span in enumerate(spans or sentence_spans(answer)):
        kind = classify_sentence(
            span.text,
            supplied_ids=supplied,
            citable_ids=citable,
            non_evidence_reports=non_evidence_reports,
        )
        validations.append(
            validate_claim(
                span.text,
                context=context,
                index=index,
                policy=policy,
                checker=checker,
                kind=kind,
            )
        )
    return tuple(validations)
