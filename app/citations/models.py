"""Data model for claim validation and Harvard citation output (RAG-012).

RAG-011 produced an answer whose every claim sentence carries an internal ``[S1]``
marker and a ``source_map`` from that marker to chunk metadata. RAG-012 adds the two
things that were deliberately deferred:

* **claim support validation** — does the cited source actually contain the claim
  (``SUPPORTED`` / ``PARTIALLY_SUPPORTED`` / ``UNSUPPORTED`` / ``UNCERTAIN``), checked
  deterministically first, with numbers and quotations verified strictly; and
* **Harvard conversion** — the internal markers become ``(Author, Year)`` citations and
  one deduplicated reference entry per cited work.

Every verdict carries the method that produced it, its confidence and a reason, so a
reader can tell a deterministic numeric check apart from a lexical overlap estimate, and
no ``UNCERTAIN`` verdict can be mistaken for a validated one.

Nothing here imports Ollama, Qwen or a retrieval service: the models stay importable and
testable without a running model server.
"""

from __future__ import annotations

import datetime as _datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SupportVerdict(str, Enum):
    """How well the cited evidence supports one claim sentence."""

    #: The cited source contains the claim.
    SUPPORTED = "SUPPORTED"
    #: Part of the claim is supported and part is not (typically two findings merged
    #: into one sentence, or several sources assembled into a single claim).
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    #: A deterministic check failed: the claim cites nothing that can support it, a
    #: number is absent from the evidence, or a quotation is not in the source.
    UNSUPPORTED = "UNSUPPORTED"
    #: The deterministic checks neither confirmed nor refuted the claim. This is never
    #: silently upgraded to ``SUPPORTED``.
    UNCERTAIN = "UNCERTAIN"

    @property
    def counts_as_supported(self) -> bool:
        """Whether this verdict may be presented as fully validated."""
        return self is SupportVerdict.SUPPORTED

    @property
    def is_failure(self) -> bool:
        """Whether the claim failed validation and must not be published as validated."""
        return self is SupportVerdict.UNSUPPORTED

    @property
    def needs_annotation(self) -> bool:
        """Whether a retained claim carries a visible caveat in ``validated_answer``."""
        return self in (SupportVerdict.PARTIALLY_SUPPORTED, SupportVerdict.UNCERTAIN)


#: Ranking used to combine per-claim verdicts; higher is worse.
VERDICT_SEVERITY: dict[SupportVerdict, int] = {
    SupportVerdict.SUPPORTED: 0,
    SupportVerdict.UNCERTAIN: 1,
    SupportVerdict.PARTIALLY_SUPPORTED: 2,
    SupportVerdict.UNSUPPORTED: 3,
}


def worst_verdict(
    verdicts: "tuple[SupportVerdict, ...] | list[SupportVerdict]",
) -> SupportVerdict:
    """The most severe verdict in ``verdicts`` (``SUPPORTED`` when empty)."""
    if not verdicts:
        return SupportVerdict.SUPPORTED
    return max(verdicts, key=lambda verdict: VERDICT_SEVERITY[verdict])


class ValidatorMethod(str, Enum):
    """Which check produced a verdict, so the guarantee stays auditable."""

    #: The marker does not exist in the supplied context.
    INVALID_MARKER = "invalid_marker"
    #: The cited block may not support scholarly claims (my_work, guidance, metadata-only).
    NON_EVIDENCE_SOURCE = "non_evidence_source"
    #: The cited block carries no source text to check against.
    MISSING_SOURCE_TEXT = "missing_source_text"
    #: The cited block reads as a reference list, not as evidence.
    BIBLIOGRAPHY_LIKE_SOURCE = "bibliography_like_source"
    #: A direct quotation was not found in the cited source text.
    QUOTATION_MISMATCH = "quotation_mismatch"
    #: A direct quotation was found in the cited source text.
    QUOTATION_EXACT = "quotation_exact"
    #: A numeric value in the claim is absent from (and not derivable from) the evidence.
    NUMERIC_MISSING = "numeric_missing"
    #: Numeric values in the claim were located in the cited evidence.
    NUMERIC_PRESENT = "numeric_present"
    #: The claim's content is spread over separate enough passages that no single one
    #: states it, and quantified units of the claim rest on different source sentences.
    LEXICAL_MERGED_FINDINGS = "lexical_merged_findings"
    #: One source sentence carries the whole claim.
    LEXICAL_SINGLE_SENTENCE = "lexical_single_sentence"
    #: The claim's content is spread over separate sentences or sources.
    LEXICAL_MULTI_SENTENCE = "lexical_multi_sentence"
    #: The claim shares almost no content with the cited sources.
    LEXICAL_NO_OVERLAP = "lexical_no_overlap"
    #: Only weak lexical overlap; the deterministic layer cannot decide.
    LEXICAL_WEAK_OVERLAP = "lexical_weak_overlap"
    #: An optional local embedding model decided the case the lexical layer could not.
    EMBEDDING_SEMANTIC = "embedding_semantic"


class ClaimKind(str, Enum):
    """What a sentence in the answer is."""

    #: A substantive sentence carrying at least one marker for a citable source.
    CLAIM = "claim"
    #: A substantive sentence whose only markers point at non-citable blocks
    #: (the author's own draft or guidance), which RAG-011 records as reports.
    NON_EVIDENCE_REPORT = "non_evidence_report"
    #: A substantive sentence with no marker (RAG-011 already flags these).
    UNMARKED_CLAIM = "unmarked_claim"
    #: A heading, bullet, question or transition: not a claim.
    NON_CLAIM = "non_claim"


class ClaimDecision(str, Enum):
    """What the citation service did with a sentence in ``validated_answer``."""

    #: Kept, markers replaced by Harvard in-text citations.
    CITATION_REPLACED = "citation_replaced"
    #: Kept with a visible caveat (partial support or unverified).
    ANNOTATED = "annotated"
    #: Withheld from ``validated_answer`` and listed in the warnings.
    WITHHELD = "withheld"
    #: Kept unchanged; a report about the author's own work or the guidance.
    NON_CITABLE_REPORT = "non_citable_report"
    #: Kept unchanged; not a claim.
    UNCHANGED = "unchanged"
    #: Kept, but a marker on a non-claim sentence (heading, bullet, question) was removed
    #: rather than converted, so no internal marker survives in ``validated_answer``.
    MARKER_REMOVED = "marker_removed"


class NumericKind(str, Enum):
    """The kind of numeric value a claim states (§6)."""

    PERCENT = "percent"
    PVALUE = "pvalue"
    RANGE = "range"
    YEAR = "year"
    COUNT = "count"
    NUMBER = "number"


class NumericMention(BaseModel):
    """One numeric value in a claim, and whether the evidence contains it."""

    model_config = ConfigDict(frozen=True)

    kind: NumericKind
    #: The text as it appears in the claim, e.g. ``"56.7%"`` or ``"p < 0.05"``.
    raw: str
    value: float | None = None
    #: Upper bound when the mention is a range.
    second_value: float | None = None
    unit: str = ""
    #: ``None`` when the mention was not checked (for example on a non-claim sentence).
    found: bool | None = None
    matched_source_ids: tuple[str, ...] = ()
    #: ``""`` when the value is present as it stands, otherwise how it was derived.
    derivation: str = ""

    @property
    def display(self) -> str:
        """The mention as a compact string, for warnings."""
        return " ".join(self.raw.split())


class QuotationCheck(BaseModel):
    """A direct quotation in a claim, and whether the cited source contains it (§7)."""

    model_config = ConfigDict(frozen=True)

    text: str
    source_ids: tuple[str, ...] = ()
    found_in_source: bool = False
    #: Which source contains the text (a cited one, or any supplied one on failure).
    matched_source_id: str | None = None
    #: ``verbatim``, ``normalized``, ``elided`` or ``not_found``.
    match_level: str = "not_found"
    #: True when the quotation was elided with ``...`` and every fragment was located.
    elided: bool = False
    #: True when the quotation exists only outside the sources the sentence cites.
    outside_cited_sources: bool = False
    page_provenance: str | None = None
    note: str = ""


class SourceSupport(BaseModel):
    """Support for one claim sentence from one cited source (§5)."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_class: str = ""
    verdict: SupportVerdict = SupportVerdict.UNCERTAIN
    method: ValidatorMethod = ValidatorMethod.LEXICAL_WEAK_OVERLAP
    confidence: float = 0.0
    reason: str = ""
    #: Share of the claim's content terms present in the best single source sentence.
    best_sentence_coverage: float = 0.0
    #: Share of the claim's content terms present anywhere in this source's text.
    union_coverage: float = 0.0
    matched_numeric: tuple[str, ...] = ()
    missing_numeric: tuple[str, ...] = ()
    quotations_found: int = 0
    quotations_missing: int = 0
    metadata_warnings: tuple[str, ...] = ()


class ClaimValidation(BaseModel):
    """Everything RAG-012 checked about one answer sentence (§2-§8, §15)."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    sentence: str = ""
    kind: ClaimKind = ClaimKind.CLAIM
    source_ids: tuple[str, ...] = ()
    invalid_source_ids: tuple[str, ...] = ()
    non_evidence_source_ids: tuple[str, ...] = ()
    verdict: SupportVerdict = SupportVerdict.UNCERTAIN
    validator_method: ValidatorMethod = ValidatorMethod.LEXICAL_WEAK_OVERLAP
    confidence: float = 0.0
    reason: str = ""
    per_source: tuple[SourceSupport, ...] = ()
    #: True when no single cited source supports the claim and the combination does.
    combination_required: bool = False
    numeric_mentions: tuple[NumericMention, ...] = ()
    quotations: tuple[QuotationCheck, ...] = ()
    #: Set when the claim rests on OCR text or a table (§8).
    elevated_validation_risk: bool = False
    risk_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    decision: ClaimDecision = ClaimDecision.UNCHANGED
    #: The Harvard in-text citation written into the sentence.
    harvard_citation: str = ""
    #: Marker text (``[S1,S4]``) to its Harvard replacement, for the citation map.
    marker_replacements: dict[str, str] = Field(default_factory=dict)
    #: The sentence as it appears in ``validated_answer``.
    validated_sentence: str = ""

    @property
    def cited_ids(self) -> tuple[str, ...]:
        """Only the source ids that may legitimately support a scholarly claim."""
        blocked = set(self.invalid_source_ids) | set(self.non_evidence_source_ids)
        return tuple(source_id for source_id in self.source_ids if source_id not in blocked)


class ReferenceEntry(BaseModel):
    """One deduplicated bibliography entry (§11-§12)."""

    model_config = ConfigDict(frozen=True)

    #: Stable key used by the citation map: ``doi:...``, ``title:...`` or ``doc:...``.
    key: str
    identity_kind: str = "document"
    identity_value: str = ""
    #: Alphabetical label used for in-text citations and ordering.
    author_label: str = ""
    #: Author list as it appears in the reference entry.
    authors_display: str = ""
    year: int | None = None
    year_label: str = ""
    title: str | None = None
    venue: str | None = None
    pages: str | None = None
    doi: str | None = None
    url: str | None = None
    source_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    metadata_warning: tuple[str, ...] = ()
    #: The formatted reference line for the reference list.
    formatted: str = ""
    #: The formatted in-text citation, e.g. ``(Doe, 2024)``.
    in_text: str = ""

    @property
    def sort_key(self) -> tuple[str, int, str]:
        """Deterministic ordering: author, then year, then the identity key."""
        return (self.author_label.casefold(), self.year or 0, self.key)


class CitationMapEntry(BaseModel):
    """``S<n>`` to Harvard in-text citation and reference key (§13)."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    in_text: str = ""
    reference_key: str = ""
    author_label: str = ""
    year_label: str = ""
    #: Documented fallbacks applied, e.g. ``missing_year`` or ``missing_author``.
    fallbacks: tuple[str, ...] = ()
    metadata_warning: tuple[str, ...] = ()
    title: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    doi: str | None = None
    pages: str | None = None


class ValidationPolicy(BaseModel):
    """The thresholds behind every verdict, recorded in the result for auditability.

    ``support_coverage`` is the share of a claim's content terms that must occur in the
    cited evidence for the claim to count as supported, ``partial_coverage`` the share
    below which only part of the claim is present, and ``no_support_coverage`` the share
    below which the evidence has essentially nothing to do with the claim.
    ``unit_coverage_floor`` is how well a quantified unit of a claim must be grounded
    before it can take part in the merged-findings test.
    """

    model_config = ConfigDict(frozen=True)

    #: Share of a claim's content terms that must occur in the cited evidence.
    support_coverage: float = 0.60
    #: Below this share, only part of the claim is present in the evidence.
    partial_coverage: float = 0.35
    #: Below this share the claim is treated as absent from the cited evidence entirely.
    no_support_coverage: float = 0.15
    #: Very short claims are never called UNSUPPORTED on overlap alone.
    min_content_terms: int = 4
    #: How well a quantified unit must be grounded to take part in the merging test.
    unit_coverage_floor: float = 0.40
    #: Stricter thresholds applied to OCR/table evidence (§8).
    risk_support_coverage: float = 0.75
    risk_partial_coverage: float = 0.50
    #: Absolute tolerance when comparing numeric values.
    numeric_tolerance: float = 1e-6
    #: Maximum confidence a lexical (probabilistic) check may claim.
    max_lexical_confidence: float = 0.80
    #: Confidence recorded for a deterministic failure (invalid marker, missing number).
    deterministic_confidence: float = 0.95

    def for_risk(self, elevated: bool) -> "ValidationPolicy":
        """This policy, with the stricter thresholds when risk is elevated."""
        if not elevated:
            return self
        return self.model_copy(
            update={
                "support_coverage": self.risk_support_coverage,
                "partial_coverage": self.risk_partial_coverage,
                "unit_coverage_floor": min(0.6, self.unit_coverage_floor + 0.1),
            }
        )


class ValidationSummary(BaseModel):
    """Counts over one validation run (§13)."""

    model_config = ConfigDict(frozen=True)

    sentences_total: int = 0
    claims_total: int = 0
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    uncertain: int = 0
    unmarked_claims: int = 0
    non_evidence_reports: int = 0
    non_claim_sentences: int = 0
    withheld_claims: int = 0
    annotated_claims: int = 0
    markers_replaced: int = 0
    markers_removed: int = 0
    invalid_markers: tuple[str, ...] = ()
    non_evidence_sources_cited: tuple[str, ...] = ()
    quotations_checked: int = 0
    quotations_failed: int = 0
    numeric_mentions_checked: int = 0
    numeric_mentions_missing: int = 0
    elevated_risk_claims: int = 0
    sources_cited: int = 0
    references_total: int = 0
    references_with_metadata_warning: int = 0

    @property
    def all_supported(self) -> bool:
        """Whether every claim sentence was fully supported."""
        return self.unsupported == 0 and self.partially_supported == 0


class CitationValidationResult(BaseModel):
    """The structured result of one citation validation run (§13)."""

    model_config = ConfigDict(frozen=True)

    question: str = ""
    #: The RAG-011 answer, unchanged, kept for auditability.
    original_answer: str = ""
    #: The answer with ``[Sx]`` replaced by Harvard citations (see the policy note).
    validated_answer: str = ""
    #: The reference list, deduplicated and deterministically ordered.
    reference_list: tuple[str, ...] = ()
    references: tuple[ReferenceEntry, ...] = ()
    claim_validations: tuple[ClaimValidation, ...] = ()
    citation_map: tuple[CitationMapEntry, ...] = ()
    #: ``"[S1,S4]"`` to the Harvard citation that replaced it.
    marker_groups: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    validation_summary: ValidationSummary = Field(default_factory=ValidationSummary)
    #: The RAG-011 ``source_map``, preserved so every citation stays traceable.
    source_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    policy: ValidationPolicy = Field(default_factory=ValidationPolicy)
    #: One line stating what happened to claims that failed verification.
    policy_note: str = ""
    #: Which support layer ran: ``deterministic`` or ``deterministic+embedding``.
    validator: str = "deterministic"
    generation_status: str = ""
    generation_warnings: tuple[str, ...] = ()
    generation_errors: tuple[str, ...] = ()
    retrieval_audit: dict[str, Any] = Field(default_factory=dict)
    validation_seconds: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: _datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
    )

    @property
    def unsupported_sentences(self) -> tuple[str, ...]:
        """The sentences withheld from ``validated_answer``."""
        return tuple(
            claim.sentence
            for claim in self.claim_validations
            if claim.verdict is SupportVerdict.UNSUPPORTED
        )

    @property
    def has_problems(self) -> bool:
        """Whether any claim failed, was only partly supported, or was left uncertain."""
        return any(
            claim.verdict is not SupportVerdict.SUPPORTED
            for claim in self.claim_validations
            if claim.kind is ClaimKind.CLAIM
        )


__all__ = [
    "ClaimDecision",
    "ClaimKind",
    "ClaimValidation",
    "CitationMapEntry",
    "CitationValidationResult",
    "NumericKind",
    "NumericMention",
    "QuotationCheck",
    "ReferenceEntry",
    "SourceSupport",
    "SupportVerdict",
    "VERDICT_SEVERITY",
    "ValidationPolicy",
    "ValidationSummary",
    "ValidatorMethod",
    "worst_verdict",
]
