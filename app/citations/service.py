"""Citation validation service (RAG-012 §1, §13-§15).

One call takes the RAG-011 :class:`~app.generation.models.GenerationResponse` and returns
the citations package:

```
GenerationResponse (answer, source_map, context, generation audit)
        |
        +-- claim validation      app.citations.support       -> claim_validations
        +-- reference identity    app.citations.harvard       -> references, reference_list
        +-- marker rewriting      _rewrite                    -> validated_answer
        +-- citation map, warnings, summary
```

The RAG-011 answer is never regenerated and never rewritten by a model: it is preserved
verbatim in ``original_answer``, and ``validated_answer`` is rebuilt from it by exact
substitution of the marker spans.

**Publication policy (fixed, and applied consistently).**

* ``SUPPORTED`` — the marker is replaced by its Harvard in-text citation.
* ``PARTIALLY_SUPPORTED`` — the claim is kept with its citation and a visible
  ``[partially supported]`` annotation, and listed in the warnings. It is never presented
  as fully validated.
* ``UNCERTAIN`` — kept with its citation and a ``[unverified]`` annotation, and listed in
  the warnings. Uncertainty is never upgraded to support.
* ``UNSUPPORTED`` — the sentence is **withheld**: it is replaced by
  ``[claim withheld: the cited source does not support it]`` and listed in the warnings.
* A report about the author's own draft or about the guidance keeps its wording, its
  internal marker is replaced by a non-scholarly provenance note, and it receives no
  reference entry.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from app.citations import harvard
from app.citations.claims import sentence_spans, strip_markers
from app.citations.models import (
    ClaimDecision,
    ClaimKind,
    ClaimValidation,
    CitationValidationResult,
    ReferenceEntry,
    SupportVerdict,
    ValidationPolicy,
    ValidationSummary,
)
from app.citations.support import (
    EmbeddingEntailmentChecker,
    EntailmentChecker,
    validate_answer,
)
from app.generation.models import ContextPackage, GenerationResponse, SourceBlock, SourceClass

#: Wording substituted for a claim that failed validation (§14).
WITHHELD_NOTE = "[claim withheld: the cited source does not support it]"
#: Visible caveats for retained claims that were not fully supported (§15).
PARTIAL_ANNOTATION = "[partially supported]"
UNCERTAIN_ANNOTATION = "[unverified]"
#: Provenance notes used instead of a citation for non-citable context blocks.
NON_CITABLE_LABELS = {
    SourceClass.MY_WORK: "[author's own work]",
    SourceClass.GUIDANCE: "[supervision guidance]",
}
EXTERNAL_METADATA_LABEL = "[external record: metadata only]"
MIXED_LABEL = "[non-citable context]"

#: The one-line statement of what happens to claims that fail verification.
POLICY_NOTE = (
    "unsupported claims are withheld from validated_answer and listed in the warnings; "
    "partially supported and uncertain claims are kept with their Harvard citation and a "
    "visible annotation; the RAG-011 answer is preserved unchanged in original_answer"
)


class CitationService:
    """Validate a generated answer's claims and convert its markers to Harvard citations."""

    def __init__(
        self,
        *,
        generation_service: Any | None = None,
        policy: ValidationPolicy | None = None,
        checker: EntailmentChecker | None = None,
        semantic: bool = False,
    ) -> None:
        self._generation = generation_service
        self.policy = policy or ValidationPolicy()
        self.checker = checker or (EmbeddingEntailmentChecker() if semantic else None)

    # --- components -----------------------------------------------------------
    @property
    def generation(self) -> Any:
        """The RAG-011 generation service, built on first use."""
        if self._generation is None:
            from app.generation.service import GenerationService

            self._generation = GenerationService()
        return self._generation

    @property
    def validator(self) -> str:
        """Which support layers are active."""
        return "deterministic" if self.checker is None else f"deterministic+{self.checker.name}"

    def close(self) -> None:
        """Close the generation service if this service created it."""
        if self._generation is not None:
            close = getattr(self._generation, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "CitationService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- main entry points ----------------------------------------------------
    def answer_and_validate(
        self, question: str, **generation_kwargs: Any
    ) -> tuple[GenerationResponse, CitationValidationResult]:
        """Run RAG-011 generation and validate the result (§16)."""
        response = self.generation.answer(question, **generation_kwargs)
        return response, self.validate(response)

    def validate(self, response: GenerationResponse) -> CitationValidationResult:
        """Validate one generation response and build its citations (§13)."""
        started = time.perf_counter()
        context = response.context
        spans = sentence_spans(response.answer)
        validations = validate_answer(
            response.answer,
            context=context,
            policy=self.policy,
            checker=self.checker,
            spans=spans,
            non_evidence_reports=response.marker_audit.non_evidence_reports,
        )
        cited_blocks = self._cited_blocks(validations, context)
        references = harvard.build_reference_list(cited_blocks)
        reference_by_source = {
            source_id: reference
            for reference in references
            for source_id in reference.source_ids
        }
        validations, replaced = _apply_citations(
            validations, spans, context, reference_by_source
        )
        validated_answer = _assemble(response.answer, spans, validations)
        citations = harvard.citation_map(response.source_map, reference_by_source)
        summary = _summarise(validations, references, replaced)
        warnings = _warnings(validations, references, context)
        return CitationValidationResult(
            question=response.question,
            original_answer=response.answer,
            validated_answer=validated_answer,
            reference_list=tuple(reference.formatted for reference in references),
            references=references,
            claim_validations=validations,
            citation_map=citations,
            marker_groups={
                group: replacement
                for validation in validations
                for group, replacement in validation.marker_replacements.items()
            },
            warnings=warnings,
            validation_summary=summary,
            source_map=response.source_map,
            policy=self.policy,
            policy_note=POLICY_NOTE,
            validator=self.validator,
            generation_status=response.status.value,
            generation_warnings=response.warnings,
            generation_errors=response.errors,
            retrieval_audit=response.retrieval_audit,
            validation_seconds=round(time.perf_counter() - started, 4),
        )

    # --- helpers --------------------------------------------------------------
    @staticmethod
    def _cited_blocks(
        validations: Sequence[ClaimValidation], context: ContextPackage
    ) -> tuple[SourceBlock, ...]:
        """Citable blocks that received an in-text citation, in first-cited order."""
        seen: list[SourceBlock] = []
        for validation in validations:
            if validation.kind is not ClaimKind.CLAIM:
                continue
            if validation.verdict is SupportVerdict.UNSUPPORTED:
                continue
            for source_id in validation.cited_ids:
                block = context.block_by_id(source_id)
                if block is not None and block.citable and block not in seen:
                    seen.append(block)
        return tuple(seen)


def _apply_citations(
    validations: Sequence[ClaimValidation],
    spans: Sequence[Any],
    context: ContextPackage,
    reference_by_source: dict[str, ReferenceEntry],
) -> tuple[tuple[ClaimValidation, ...], int]:
    """Rewrite each sentence and record the citation it received.

    The rewritten sentence is stored on the validation itself, so the audit and the
    ``validated_answer`` can never disagree about what a marker became.
    """
    updated: list[ClaimValidation] = []
    replaced = 0
    for validation, span in zip(validations, spans):
        sentence = span.text
        text, count, citation, replacements = _rewrite_sentence(
            sentence, validation, reference_by_source, context
        )
        replaced += count
        updated.append(
            validation.model_copy(
                update={
                    "validated_sentence": text,
                    "harvard_citation": citation,
                    "marker_replacements": replacements,
                }
            )
        )
    return tuple(updated), replaced


def _assemble(
    answer: str, spans: Sequence[Any], validations: Sequence[ClaimValidation]
) -> str:
    """Rebuild the answer from the per-sentence rewrites, preserving the layout."""
    pieces: list[str] = []
    cursor = 0
    for span, validation in zip(spans, validations):
        pieces.append(answer[cursor : span.start])
        pieces.append(validation.validated_sentence)
        cursor = span.end
    pieces.append(answer[cursor:])
    return "".join(pieces)


def _rewrite_sentence(
    sentence: str,
    validation: ClaimValidation,
    reference_by_source: dict[str, ReferenceEntry],
    context: ContextPackage,
) -> tuple[str, int, str, dict[str, str]]:
    """One sentence's replacement text, its citation, and the marker substitutions."""
    from app.citations.claims import marker_groups

    groups = marker_groups(sentence)
    citable_groups = [
        [source_id for source_id in ids if _citable(source_id, validation, reference_by_source)]
        for _group, ids in groups
    ]
    citation = " & ".join(
        harvard.citation_for_ids(ids, reference_by_source)
        for ids in citable_groups
        if ids
    )
    if validation.decision is ClaimDecision.WITHHELD:
        return WITHHELD_NOTE, 0, "", {group: WITHHELD_NOTE for group, _ids in groups}
    if validation.decision is ClaimDecision.MARKER_REMOVED:
        # A marker on a heading or bullet: dropped, so no internal marker survives.
        return " ".join(strip_markers(sentence).split()), 0, "", {
            group: "" for group, _ids in groups
        }
    if validation.decision is ClaimDecision.NON_CITABLE_REPORT:
        replacements = {
            group: _report_label(context, ids) for group, ids in groups
        }
        text = sentence
        for group, replacement in replacements.items():
            text = text.replace(group, replacement, 1)
        return " ".join(text.split()), 0, "", replacements
    if validation.decision not in (ClaimDecision.CITATION_REPLACED, ClaimDecision.ANNOTATED):
        return sentence, 0, "", {}

    replacements: dict[str, str] = {}
    text = sentence
    replaced = 0
    for group, ids in groups:
        replacement = harvard.citation_for_ids(ids, reference_by_source)
        replacements[group] = replacement
        if replacement:
            replaced += 1
        text = text.replace(group, replacement, 1)
    if validation.decision is ClaimDecision.ANNOTATED:
        annotation = (
            PARTIAL_ANNOTATION
            if validation.verdict is SupportVerdict.PARTIALLY_SUPPORTED
            else UNCERTAIN_ANNOTATION
        )
        text = f"{text} {annotation}"
    return text, replaced, citation, replacements


def _citable(
    source_id: str,
    validation: ClaimValidation,
    reference_by_source: dict[str, ReferenceEntry],
) -> bool:
    """Whether a cited source may receive a Harvard citation in this claim."""
    return (
        source_id in reference_by_source
        and source_id not in validation.invalid_source_ids
        and source_id not in validation.non_evidence_source_ids
    )


def _report_label(context: ContextPackage, ids: Sequence[str]) -> str:
    """The provenance note used instead of citing non-citable material."""
    classes = {
        block.source_class
        for block in (context.block_by_id(source_id) for source_id in ids)
        if block is not None
    }
    if len(classes) == 1:
        source_class = next(iter(classes))
        if source_class in NON_CITABLE_LABELS:
            return NON_CITABLE_LABELS[source_class]
        if source_class.is_metadata_only:
            return EXTERNAL_METADATA_LABEL
    return MIXED_LABEL


def _summarise(
    validations: Sequence[ClaimValidation],
    references: Sequence[ReferenceEntry],
    markers_replaced: int,
) -> ValidationSummary:
    """Count the outcomes of one validation run."""
    claims = [item for item in validations if item.kind is ClaimKind.CLAIM]
    invalid_markers: list[str] = []
    non_evidence: list[str] = []
    for claim in claims:
        invalid_markers.extend(claim.invalid_source_ids)
        non_evidence.extend(claim.non_evidence_source_ids)
    quotations = [check for claim in claims for check in claim.quotations]
    numerics = [
        mention
        for claim in claims
        for mention in claim.numeric_mentions
        if mention.found is not None
    ]
    return ValidationSummary(
        sentences_total=len(validations),
        claims_total=len(claims),
        supported=sum(1 for claim in claims if claim.verdict is SupportVerdict.SUPPORTED),
        partially_supported=sum(
            1 for claim in claims if claim.verdict is SupportVerdict.PARTIALLY_SUPPORTED
        ),
        unsupported=sum(1 for claim in claims if claim.verdict is SupportVerdict.UNSUPPORTED),
        uncertain=sum(1 for claim in claims if claim.verdict is SupportVerdict.UNCERTAIN),
        unmarked_claims=sum(
            1 for item in validations if item.kind is ClaimKind.UNMARKED_CLAIM
        ),
        non_evidence_reports=sum(
            1 for item in validations if item.kind is ClaimKind.NON_EVIDENCE_REPORT
        ),
        non_claim_sentences=sum(1 for item in validations if item.kind is ClaimKind.NON_CLAIM),
        withheld_claims=sum(
            1 for item in validations if item.decision is ClaimDecision.WITHHELD
        ),
        annotated_claims=sum(
            1 for item in validations if item.decision is ClaimDecision.ANNOTATED
        ),
        markers_replaced=markers_replaced,
        markers_removed=sum(
            1 for item in validations if item.decision is ClaimDecision.MARKER_REMOVED
        ),
        invalid_markers=tuple(dict.fromkeys(invalid_markers)),
        non_evidence_sources_cited=tuple(dict.fromkeys(non_evidence)),
        quotations_checked=len(quotations),
        quotations_failed=sum(1 for check in quotations if not check.found_in_source),
        numeric_mentions_checked=len(numerics),
        numeric_mentions_missing=sum(1 for mention in numerics if mention.found is False),
        elevated_risk_claims=sum(1 for claim in claims if claim.elevated_validation_risk),
        sources_cited=len({sid for ref in references for sid in ref.source_ids}),
        references_total=len(references),
        references_with_metadata_warning=sum(
            1 for reference in references if reference.metadata_warning
        ),
    )


def _warnings(
    validations: Sequence[ClaimValidation],
    references: Sequence[ReferenceEntry],
    context: ContextPackage,
) -> tuple[str, ...]:
    """Human-readable warnings, one per problem, in a fixed order (§13)."""
    warnings: list[str] = []
    for claim in validations:
        if claim.kind is ClaimKind.CLAIM:
            if claim.verdict is SupportVerdict.UNSUPPORTED:
                warnings.append(
                    f"unsupported_claim: {claim.sentence} "
                    f"(method: {claim.validator_method.value}; {claim.reason})"
                )
            elif claim.verdict is SupportVerdict.PARTIALLY_SUPPORTED:
                warnings.append(
                    f"partially_supported_claim: {claim.sentence} "
                    f"(method: {claim.validator_method.value}; {claim.reason})"
                )
            elif claim.verdict is SupportVerdict.UNCERTAIN:
                warnings.append(
                    f"uncertain_claim: {claim.sentence} "
                    f"(method: {claim.validator_method.value}; {claim.reason})"
                )
            for source_id in claim.invalid_source_ids:
                warnings.append(
                    f"invalid_source_marker: [{source_id}] does not exist in the supplied "
                    f"context"
                )
            for source_id in claim.non_evidence_source_ids:
                warnings.append(
                    f"non_evidence_source_cited: [{source_id}] "
                    f"({_source_class(context, source_id)}) cannot support a scholarly claim"
                )
            for check in claim.quotations:
                if not check.found_in_source:
                    warnings.append(
                        f"quotation_not_in_cited_source: {check.text[:120]!r} ({check.note})"
                    )
            for mention in claim.numeric_mentions:
                if mention.found is False:
                    warnings.append(
                        f"numeric_claim_not_in_evidence: {mention.display} "
                        f"({mention.kind.value}) is absent from the cited evidence"
                    )
            if claim.elevated_validation_risk:
                warnings.append(
                    "elevated_validation_risk: "
                    + "; ".join(claim.risk_reasons)
                    + f" (claim: {claim.sentence[:120]})"
                )
    for reference in references:
        if reference.metadata_warning:
            warnings.append(
                f"reference_metadata_warning: {reference.key} "
                f"({', '.join(reference.metadata_warning)})"
            )
    return tuple(warnings)


def _source_class(context: ContextPackage, source_id: str) -> str:
    block = context.block_by_id(source_id)
    return block.source_class.value if block is not None else "unknown"
