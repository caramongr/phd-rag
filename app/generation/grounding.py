"""Grounding checks on the generated answer (RAG-011 §11-§17, §26).

The model is asked to cite every substantive claim with a marker such as ``[S1]`` or
``[S2,S4]``. This module checks that request mechanically and never rewrites the answer:

* every marker must exist in the supplied context (§13) — an unknown ``S7`` is recorded
  as ``invalid_source_marker``, not deleted as if the answer were fine;
* a sentence that cites only non-citable material in an evidence answer is flagged
  (§18);
* a sentence that reads like a finding but cites a metadata-only external record is
  flagged (§5, §28);
* quotations are located in the source text where possible and recorded with their page
  provenance, so RAG-012 can validate them (§15).

The checks are deliberately conservative and explainable: a text pattern either matches
or it does not, and no language model is involved.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from app.generation.models import (
    ContextPackage,
    QuotationEvidence,
    SourceBlock,
    SourceClass,
    SourceMarkerAudit,
)

#: ``[S1]``, ``[S2,S4]`` and ``[S2, S4, S7]``.
MARKER_RE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")
#: Verbs and quantities that mark a sentence as reporting a finding. Single-word entries
#: are matched as whole words. The list is generous on purpose: for an unmarked-claim
#: check, flagging a framing sentence is cheaper than missing a finding claim.
_CLAIM_SIGNALS = (
    "found that",
    "found a",
    "finds that",
    "shows that",
    "show that",
    "reports that",
    "report that",
    "suggests that",
    "suggest that",
    "indicates that",
    "indicate that",
    "demonstrates that",
    "demonstrate that",
    "argues that",
    "argue that",
    "concludes that",
    "conclude that",
    "states that",
    "claims that",
    "evidence that",
    "results show",
    "the study",
    "the analysis",
    "the survey",
    "the review",
    "indicate",
    "suggests",
    "propose",
    "proposes",
    "demonstrate",
    "reveal",
    "reveals",
)
#: Strong signals that a sentence states a *finding*, used for the stricter
#: metadata-only external check where an allowed existence mention must stay allowed.
_CONTENT_SIGNALS = (
    "found that",
    "finds that",
    "shows that",
    "show that",
    "reports that",
    "report that",
    "suggests that",
    "indicates that",
    "demonstrates that",
    "concludes that",
    "concluded",
    "the findings",
    "the results",
    "the study found",
    "the paper",
    "the report",
    "according to",
    "evidence shows",
    "significantly",
    "improved",
    "increased",
    "decreased",
    "reduced",
    "outperformed",
    "suggest",
    "suggests",
    "indicate",
    "indicates",
    "propose",
    "proposes",
    "demonstrate",
    "demonstrates",
    "reveal",
    "reveals",
    "report",
    "reports",
    "shows",
    "showed",
    "conclude",
    "concludes",
)
#: Phrases that only say a record exists or is worth investigating, which RAG-011 §28
#: explicitly allows for metadata-only records.
_EXISTENCE_FRAMING = (
    "indicate the existence",
    "indicates the existence",
    "indicating the existence",
    "the existence of",
    "may be relevant",
    "might be relevant",
    "could be relevant",
    "worth investigating",
    "for investigation",
    "was found",
    "were found",
    "exists",
    "is available",
    "appears in",
    "is titled",
    "listed here",
    "can be found",
    "are titled",
    "recent scholarship",
)
_QUANTITY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|percentage|points?|times|fold|participants|"
    r"students|respondents|cases|papers|articles|studies)\b",
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(
    r"[\u201c\u201d\"]([^\u201c\u201d\"]{12,})[\u201c\u201d\"]"
)
#: Words that make a sentence a report *about* supplied non-evidence material rather
#: than a claim about the world. Describing the author's own draft, or saying what the
#: guidance asks for, is the intended use of those blocks (RAG-011 §18-§19).
_OWN_WORK_PHRASES = (
    "your draft",
    "your text",
    "your writing",
    "your own",
    "your section",
    "your chapter",
    "your argument",
    "your work",
    "in your ",
    "according to your",
)
#: "you describe", but also "you further describe", "you also note", ...
_OWN_WORK_VERBS = re.compile(
    r"\byou\s+(?:(?:also|further|furthermore|then|additionally|finally|next|"
    r"subsequently|instead|therefore)\s+)*"
    r"(?:outline|identify|state|note|describe|record|attribute|argue|define|present|"
    r"report|explain|introduce|propose|suggest)\b",
    re.IGNORECASE,
)
_GUIDANCE_FRAMING = (
    "the guidance",
    "the supervision",
    "the supervisor",
    "your supervisor",
    "the notes",
    "the plan",
    "the rules",
    "the writing guidelines",
    "the guidelines",
    "the policy document",
)
#: Words that make a sentence a transition rather than a claim.
_TRANSITION_PREFIXES = (
    "in summary",
    "in short",
    "overall",
    "taken together",
    "in conclusion",
    "in other words",
    "to conclude",
    "in practice",
)
#: The sentence the prompt asks the model to use when the evidence is insufficient.
INSUFFICIENT_MARKERS = (
    "does not provide enough support",
    "insufficient evidence",
    "not enough evidence",
    "does not contain",
    "do not provide enough",
    "no evidence in the sources",
    "cannot be answered",
    "not supported by the retrieved evidence",
)
#: House style: the em dash character is not used in answers.
FORBIDDEN_CHARACTERS = {"\u2014": "em dash"}


def _normalize_for_match(text: str) -> str:
    """Casefold, collapse whitespace and drop punctuation (for quote matching)."""
    lowered = text.casefold()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


#: Abbreviations whose full stop must not end a sentence. "Research by Merchant et al.
#: (2020) found ..." is one sentence, and splitting it there invents a fake fragment.
_ABBREVIATIONS = (
    "et al.",
    "e.g.",
    "i.e.",
    "cf.",
    "vs.",
    "fig.",
    "figs.",
    "no.",
    "pp.",
    "ed.",
    "eds.",
    "dr.",
    "prof.",
    "approx.",
    "etc.",
    "inc.",
    "al.",
)
#: Temporary stand-in for a protected full stop (never appears in model output).
_ABBREVIATION_PLACEHOLDER = "\x00"


def _protect_abbreviations(text: str) -> str:
    protected = text
    for abbreviation in _ABBREVIATIONS:
        pattern = re.compile(re.escape(abbreviation), re.IGNORECASE)
        protected = pattern.sub(
            lambda match: match.group(0)[:-1] + _ABBREVIATION_PLACEHOLDER, protected
        )
    return protected


def _restore_abbreviations(text: str) -> str:
    return text.replace(_ABBREVIATION_PLACEHOLDER, ".")


def split_sentences(answer: str) -> list[str]:
    """Split an answer into sentences, keeping each one intact for reporting."""
    sentences: list[str] = []
    for block in answer.split("\n"):
        stripped = block.strip()
        if not stripped:
            continue
        protected = _protect_abbreviations(stripped)
        for sentence in _SENTENCE_SPLIT_RE.split(protected):
            candidate = _restore_abbreviations(sentence).strip()
            if candidate:
                sentences.append(candidate)
    return sentences


def extract_markers(text: str) -> tuple[str, ...]:
    """Every source id mentioned in ``text``, in first-seen order, normalised to ``S<n>``."""
    found: list[str] = []
    for match in MARKER_RE.finditer(text or ""):
        for part in match.group(1).split(","):
            source_id = part.strip().upper()
            if source_id and source_id not in found:
                found.append(source_id)
    return tuple(found)


def marker_counts(text: str) -> dict[str, int]:
    """How often each source is cited (a marker listing two sources counts for both)."""
    counts: dict[str, int] = {}
    for _match in MARKER_RE.finditer(text or ""):
        for part in _match.group(1).split(","):
            source_id = part.strip().upper()
            if source_id:
                counts[source_id] = counts.get(source_id, 0) + 1
    return counts


def _sentence_of(text: str, marker: str) -> str | None:
    for sentence in split_sentences(text):
        if marker in {item.upper() for item in extract_markers(sentence)}:
            return sentence
    return None


def _reports_content(sentence: str) -> bool:
    """Whether a sentence states a finding rather than that a record exists.

    Explicit existence phrasing is removed first, so "records indicate the existence of
    recent scholarship" is not mistaken for "records indicate that engagement improved".
    """
    text = sentence.casefold()
    for phrase in _EXISTENCE_FRAMING:
        text = text.replace(phrase, " ")
    tokens = set(re.findall(r"[a-z]+", text))
    for signal in _CONTENT_SIGNALS:
        if " " in signal:
            if signal in text:
                return True
        elif signal in tokens:
            return True
    return bool(_QUANTITY_RE.search(text))


def _is_existence_mention(sentence: str) -> bool:
    """Whether a sentence only says a record exists or is worth investigating."""
    lowered = sentence.casefold()
    return any(phrase in lowered for phrase in _EXISTENCE_FRAMING)


def _has_claim_signal(sentence: str) -> bool:
    """Whether a sentence reads as reporting something rather than merely framing.

    Single-word signals are matched as whole words; the list is deliberately generous,
    because flagging a sentence that turns out to be framing is cheaper than missing a
    finding claim that rests on a metadata-only record.
    """
    lowered = sentence.casefold()
    tokens = set(re.findall(r"[a-z]+", lowered))
    for signal in _CLAIM_SIGNALS:
        if " " in signal:
            if signal in lowered:
                return True
        elif signal in tokens:
            return True
    return bool(_QUANTITY_RE.search(sentence))


def _refers_to_own_work(sentence: str) -> bool:
    """Whether the sentence describes the author's own material rather than the world."""
    lowered = sentence.casefold()
    if any(phrase in lowered for phrase in _OWN_WORK_PHRASES):
        return True
    return bool(_OWN_WORK_VERBS.search(sentence))


def _refers_to_guidance(sentence: str) -> bool:
    lowered = sentence.casefold()
    return any(phrase in lowered for phrase in _GUIDANCE_FRAMING)


def _is_trivial(sentence: str) -> bool:
    """A heading, list bullet or one-word fragment makes no claim."""
    stripped = sentence.strip().strip("#*-• ")
    if len(stripped) < 25:
        return True
    if stripped.endswith("?"):
        return True
    return stripped.casefold().startswith(_TRANSITION_PREFIXES)


def _mentions_insufficient_evidence(sentence: str) -> bool:
    lowered = sentence.casefold()
    return any(phrase in lowered for phrase in INSUFFICIENT_MARKERS)


def sentence_is_trivial(sentence: str) -> bool:
    """Public wrapper for :func:`_is_trivial`, reused by RAG-012 claim extraction."""
    return _is_trivial(sentence)


def sentence_has_claim_signal(sentence: str) -> bool:
    """Public wrapper for :func:`_has_claim_signal`, reused by RAG-012."""
    return _has_claim_signal(sentence)


def find_quotations(answer: str, context: ContextPackage) -> tuple[QuotationEvidence, ...]:
    """Locate each quotation in the cited source text, when it can be located."""
    quotations: list[QuotationEvidence] = []
    for match in _QUOTE_RE.finditer(answer or ""):
        quoted = match.group(1).strip()
        if not quoted:
            continue
        sentence = _sentence_of(answer, "") or ""
        for candidate_sentence in split_sentences(answer):
            if quoted[:24] in candidate_sentence:
                sentence = candidate_sentence
                break
        source_ids = extract_markers(sentence)
        blocks = [
            block
            for block in (context.block_by_id(source_id) for source_id in source_ids)
            if block is not None
        ]
        if not blocks:
            blocks = list(context.citable_blocks)
        needle = _normalize_for_match(quoted)
        found_block: SourceBlock | None = None
        if needle:
            for block in blocks:
                if needle and needle in _normalize_for_match(block.text):
                    found_block = block
                    break
        quotations.append(
            QuotationEvidence(
                text=quoted,
                source_ids=source_ids,
                found_in_source=found_block is not None,
                page_provenance=found_block.pages if found_block else None,
                note=(
                    "quotation located in the cited source text"
                    if found_block is not None
                    else "quotation not found in the supplied source text; flagged for RAG-012"
                ),
            )
        )
    return tuple(quotations)


def _is_context_report(
    sentence: str,
    cited_blocks: Sequence[SourceBlock],
    context: ContextPackage,
) -> bool:
    """Whether a sentence legitimately cites only non-evidence material.

    Three cases are legitimate and are recorded as reports rather than failures:

    * ``my_work`` in ``evidence_plus_my_work`` mode: saying what the author's own draft
      contains is the intended use of that block (§19);
    * ``guidance`` in ``evidence_plus_guidance`` mode: saying what the guidance asks for;
    * a metadata-only *external* record where the sentence only states that a paper
      exists or is worth investigating, which RAG-011 §28 explicitly allows.

    The check requires both the right source class *and* the right wording; anything else
    stays a ``non_evidence_citation``.
    """
    if not cited_blocks:
        return False
    classes = {block.source_class for block in cited_blocks}
    if classes == {SourceClass.MY_WORK} and context.mode.includes_my_work:
        return _refers_to_own_work(sentence)
    if classes == {SourceClass.GUIDANCE} and context.mode.includes_guidance:
        return _refers_to_guidance(sentence)
    if all(block.source_class.is_metadata_only for block in cited_blocks):
        return _is_existence_mention(sentence) and not _reports_content(sentence)
    return False


def audit_markers(
    answer: str,
    context: ContextPackage,
    *,
    external_metadata_only_ids: Iterable[str] | None = None,
) -> SourceMarkerAudit:
    """Check the answer's markers, claims and quotations against the context.

    When ``external_metadata_only_ids`` is not supplied it is derived from the context,
    so a caller cannot accidentally disable the external-records check by omission.
    """
    supplied = set(context.source_ids)
    citable = set(context.citable_source_ids)
    metadata_only = (
        {block.source_id for block in context.blocks if block.source_class.is_metadata_only}
        if external_metadata_only_ids is None
        else {item.upper() for item in external_metadata_only_ids}
    )

    markers = extract_markers(answer)
    invalid = tuple(marker for marker in markers if marker not in supplied)

    unmarked_claims: list[str] = []
    unmarked_statements: list[str] = []
    non_evidence_citations: list[str] = []
    non_evidence_reports: list[str] = []
    external_findings: list[str] = []
    sentences = split_sentences(answer)
    with_markers = 0
    for sentence in sentences:
        sentence_markers = tuple(item.upper() for item in extract_markers(sentence))
        if sentence_markers:
            with_markers += 1
            if not any(marker in citable for marker in sentence_markers):
                if not _mentions_insufficient_evidence(sentence):
                    cited_blocks = [
                        context.block_by_id(marker)
                        for marker in sentence_markers
                        if context.block_by_id(marker) is not None
                    ]
                    if _is_context_report(sentence, cited_blocks, context):
                        non_evidence_reports.append(sentence)
                    else:
                        non_evidence_citations.append(sentence)
            if any(marker in metadata_only for marker in sentence_markers) and _reports_content(
                sentence
            ) and not _is_existence_mention(sentence):
                external_findings.append(sentence)
            continue
        if _is_trivial(sentence) or _mentions_insufficient_evidence(sentence):
            continue
        if _has_claim_signal(sentence):
            unmarked_claims.append(sentence)
        else:
            unmarked_statements.append(sentence)

    style_warnings: list[str] = []
    for character, name in FORBIDDEN_CHARACTERS.items():
        if character in (answer or ""):
            style_warnings.append(f"the answer contains a {name} character")

    return SourceMarkerAudit(
        markers_found=markers,
        marker_counts=marker_counts(answer),
        invalid_markers=invalid,
        unmarked_claims=tuple(unmarked_claims),
        non_evidence_citations=tuple(non_evidence_citations),
        non_evidence_reports=tuple(non_evidence_reports),
        external_finding_claims=tuple(external_findings),
        quotations=find_quotations(answer, context),
        sentences_total=len(sentences),
        sentences_with_markers=with_markers,
        unmarked_statements=tuple(unmarked_statements),
        style_warnings=tuple(style_warnings),
    )


def used_source_ids(answer: str, context: ContextPackage) -> tuple[str, ...]:
    """Source ids the answer actually cites, in first-seen order (supplied ids only)."""
    supplied = set(context.source_ids)
    return tuple(marker for marker in extract_markers(answer) if marker in supplied)


def unused_source_ids(answer: str, context: ContextPackage) -> tuple[str, ...]:
    """Supplied source ids the answer never used."""
    used = set(extract_markers(answer))
    return tuple(source_id for source_id in context.source_ids if source_id not in used)


def grounding_warnings(audit: SourceMarkerAudit) -> tuple[str, ...]:
    """Human-readable warnings derived from a marker audit (§12)."""
    warnings: list[str] = []
    for marker in audit.invalid_markers:
        warnings.append(
            f"invalid_source_marker: [{marker}] does not exist in the supplied context"
        )
    for sentence in audit.unmarked_claims:
        warnings.append(f"unmarked_claim: {sentence[:200]}")
    for sentence in audit.non_evidence_citations:
        warnings.append(f"non_evidence_citation: {sentence[:200]}")
    for sentence in audit.external_finding_claims:
        warnings.append(f"external_finding_claim: {sentence[:200]}")
    for quotation in audit.quotations:
        if not quotation.found_in_source:
            warnings.append(
                f"unverified_quotation: {quotation.text[:120]!r} was not found in the "
                f"supplied source text"
            )
    for sentence in audit.unmarked_statements:
        warnings.append(f"unmarked_statement: {sentence[:200]}")
    for warning in audit.style_warnings:
        warnings.append(f"style: {warning}")
    return tuple(warnings)


def external_discovery_notes(context: ContextPackage, answer: str) -> tuple[str, ...]:
    """What the answer did with external records, recorded for the audit (§5, §28)."""
    external = [block for block in context.blocks if block.external]
    if not external:
        return ()
    used = set(extract_markers(answer))
    notes: list[str] = []
    for block in external:
        state = "cited" if block.source_id in used else "not cited"
        kind = (
            "verified full text (external, not corpus evidence)"
            if block.fulltext_verified
            else "metadata only (no findings available)"
        )
        notes.append(f"[{block.source_id}] {state}: {kind} - {block.title or '(untitled)'}")
    return tuple(notes)
